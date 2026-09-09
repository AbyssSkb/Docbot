import math
import platform

import torch
import faiss
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi
import jieba


# ponytail: single-threaded FAISS avoids a macOS arm64 native crash; this corpus is small.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    faiss.omp_set_num_threads(1)
DEVICE = torch.accelerator.current_accelerator(True) or torch.device("cpu")
RERANK_BATCH_SIZE = 8


class EmbeddingModel:
    def __init__(
        self, model_path, index_path=None, query_prompt_name=None, revision=None
    ):
        from sentence_transformers import SentenceTransformer

        self.device = DEVICE
        self.model = SentenceTransformer(
            model_path,
            revision=revision,
            device=str(self.device),
        )
        self.model.max_seq_length = 512
        self.index = (
            faiss.read_index(str(index_path)) if index_path is not None else None
        )
        self.query_prompt_name = query_prompt_name

    def _embed_texts(self, texts, prompt_name=None):
        kwargs = {"prompt_name": prompt_name} if prompt_name else {}
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            **kwargs,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_text(self, text, prompt_name=None):
        return self._embed_texts([text], prompt_name=prompt_name)[0].tolist()

    def save_index(self, texts, save_path, batch_size=32):
        texts = list(texts)
        if not texts:
            raise ValueError("cannot build an index without texts")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.index = None
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            embeddings = self._embed_texts(texts[start : start + batch_size])
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        faiss.write_index(self.index, str(save_path))

    def query(self, query, k=20):
        if self.index is None or self.index.ntotal == 0 or k <= 0:
            return []

        query_embedding = np.asarray(
            [self.embed_text(query, prompt_name=self.query_prompt_name)],
            dtype="float32",
        )
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        return [
            {"index": int(index), "raw_score": float(score)}
            for index, score in zip(indices[0], scores[0])
        ]


class BM25Model:
    def __init__(self, texts):
        self.tokenized_texts = [jieba.lcut(text) for text in texts]
        self.model = BM25Okapi(self.tokenized_texts) if self.tokenized_texts else None

    def query(self, query, k=20):
        if self.model is None or k <= 0:
            return []

        query = jieba.lcut(query)
        scores = self.model.get_scores(query)
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            {"index": index, "raw_score": float(scores[index])} for index in indices
        ]


def reciprocal_rank_fusion(results_by_route, k=60, route_weights=None):
    route_weights = route_weights or {}
    fused = {}
    for route, results in results_by_route.items():
        weight = route_weights.get(route, 1.0)
        for rank, result in enumerate(results, start=1):
            index = result["index"]
            candidate = fused.setdefault(
                index, {"index": index, "rrf_score": 0.0, "routes": []}
            )
            candidate["rrf_score"] += weight / (k + rank)
            candidate["routes"].append(
                {
                    "route": route,
                    "rank": rank,
                    "raw_score": result["raw_score"],
                }
            )

    return sorted(fused.values(), key=lambda item: (-item["rrf_score"], item["index"]))


class RerankingModel:
    def __init__(self, model_path, revision=None):
        if not model_path.startswith("Qwen/Qwen3-Reranker"):
            raise ValueError("Only Qwen/Qwen3-Reranker models are supported")

        from sentence_transformers import CrossEncoder

        self.device = DEVICE
        self.model = CrossEncoder(
            model_path,
            revision=revision,
            device=str(self.device),
            prompts={
                "docbot": (
                    "Given a Chinese question, retrieve passages from a book "
                    "that directly answer the question."
                )
            },
            default_prompt_name="docbot",
            max_length=512,
            model_kwargs=(
                {"torch_dtype": torch.float32}
                if getattr(self.device, "type", None) in {"cpu", "mps"}
                else {}
            ),
        )

    def select(self, query, candidates, k=10, min_score=None):
        if not candidates or k <= 0:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        with torch.no_grad():
            scores = self.model.predict(
                pairs,
                batch_size=min(RERANK_BATCH_SIZE, len(pairs)),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        if getattr(self.device, "type", None) == "mps":
            torch.mps.synchronize()
        scores = np.asarray(scores, dtype="float32").reshape(-1).tolist()

        ranked = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
            if math.isfinite(score) and (min_score is None or score >= min_score)
        ]
        if not ranked and min_score is None:
            return [{**candidate, "rerank_score": 0.0} for candidate in candidates[:k]]
        ranked.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
        return ranked[:k]
