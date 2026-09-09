#!/usr/bin/env python3
"""Run Docbot retrieval or end-to-end benchmarks."""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows has no stdlib resource module.
    resource = None

from dotenv import load_dotenv
from tqdm import tqdm

from create_index import sha256_file
from eval import load_gold, validate_gold_mapping, write_jsonl
from pipeline import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    is_refusal,
    load_index_bundle,
    load_models,
    retrieve,
    validate_answer_citations,
)


load_dotenv()

CONFIG_ROUTES = {
    "bm25": ("bm25",),
    "embed1": ("embed1",),
    "embed2": ("embed2",),
    "dual_dense": ("embed1", "embed2"),
    "hybrid": ("embed1", "embed2", "bm25"),
    "hybrid_rerank": ("embed1", "embed2", "bm25"),
}
SCORE_SEMANTICS = {
    "bm25": "BM25 raw score; higher is better",
    "embed1": "negative FAISS squared L2 distance; higher is better",
    "embed2": "negative FAISS squared L2 distance; higher is better",
    "dual_dense": "RRF score; higher is better",
    "hybrid": "RRF score; higher is better",
    "hybrid_rerank": "reranker raw logit; higher is better",
}
RERANKER_MODEL = DEFAULT_RERANKER_MODEL
RERANKER_REVISION = DEFAULT_RERANKER_REVISION
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
BENCHMARK_RETRIEVAL_K = 50
CONTEXT_K = 10


def _top_score(result, config):
    if config == "hybrid_rerank":
        return float(result["rerank_score"])
    if config == "bm25":
        return float(result["routes"][0]["raw_score"])
    if config in {"embed1", "embed2"}:
        return -float(result["routes"][0]["raw_score"])
    return float(result["rrf_score"])


def _generate(client, model, contexts, question):
    from pipeline import build_messages, parse_agent_response

    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, contexts),
        stream=False,
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    action, answer = parse_agent_response(content) if content else ("search", "")
    if action == "search":
        answer = "无答案"
    allowed = [context["chunk_id"] for context in contexts]
    valid, invalid, citation_valid = validate_answer_citations(answer, allowed)
    citations = sorted(valid + invalid, key=answer.find)
    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return answer, citations, citation_valid, input_tokens, output_tokens


def _memory_mb():
    result = {}
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        result["peak_rss"] = round(rss / divisor, 1)
    torch = sys.modules.get("torch")
    if torch is not None and torch.accelerator.is_available():
        result["peak_accelerator_allocated"] = round(
            torch.accelerator.max_memory_allocated() / 1024**2, 1
        )
    return result


def _percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def run_benchmark(
    gold_records,
    chunks,
    retrievers,
    ranker=None,
    config="hybrid",
    split="all",
    threshold=None,
    generate=False,
    client=None,
    model=LLM_MODEL,
    input_usd_per_million=None,
    output_usd_per_million=None,
    clock=time.perf_counter,
):
    if config not in CONFIG_ROUTES:
        raise ValueError(f"unknown benchmark config: {config}")
    if split not in {"dev", "test", "all"}:
        raise ValueError(f"unknown benchmark split: {split}")
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if config == "hybrid_rerank" and ranker is None:
        raise ValueError("hybrid_rerank requires a reranker")
    if generate and client is None:
        raise ValueError("generate requires an OpenAI client")
    rates = (input_usd_per_million, output_usd_per_million)
    if (rates[0] is None) != (rates[1] is None):
        raise ValueError("input and output token rates must be provided together")
    if any(
        rate is not None and (not math.isfinite(rate) or rate < 0) for rate in rates
    ):
        raise ValueError("token rates must be finite and non-negative")
    if rates[0] is not None and not generate:
        raise ValueError("token rates require --generate")

    routes = CONFIG_ROUTES[config]
    missing = set(routes) - retrievers.keys()
    if missing:
        raise ValueError(f"missing retrievers: {', '.join(sorted(missing))}")
    selected_retrievers = {route: retrievers[route] for route in routes}
    gold_rows = [row for row in gold_records if split == "all" or row["split"] == split]
    if not gold_rows:
        raise ValueError("gold dataset has no items for the selected split")
    validate_gold_mapping(gold_records, chunks, split)
    predictions = []
    latencies = []
    end_to_end_latencies = []
    input_tokens = output_tokens = 0
    citation_validation_failures = 0
    for row in tqdm(gold_rows, desc=f"{config} {split}", unit="query"):
        started = clock()
        ranked_results = retrieve(
            row["question"],
            chunks,
            selected_retrievers,
            ranker if config == "hybrid_rerank" else None,
            candidate_k=BENCHMARK_RETRIEVAL_K,
            result_k=BENCHMARK_RETRIEVAL_K,
        )
        latencies.append((clock() - started) * 1000)
        top_score = _top_score(ranked_results[0], config) if ranked_results else 0.0
        if not math.isfinite(top_score):
            raise ValueError(f"non-finite top score for {row['question_id']}")
        results = ranked_results[:CONTEXT_K]
        if generate and config == "hybrid_rerank" and threshold is not None:
            results = [
                result for result in results if result["rerank_score"] >= threshold
            ]
        prediction = {
            "question_id": row["question_id"],
            "retrieved_chunk_ids": [
                result["chunk_id"] for result in ranked_results
            ],
            "cited_chunk_ids": [],
            "answered": bool(ranked_results)
            and (threshold is None or top_score >= threshold),
        }
        prediction["top_score"] = top_score
        if generate:
            prediction["context_chunk_ids"] = (
                [result["chunk_id"] for result in results] if prediction["answered"] else []
            )
            if prediction["answered"]:
                (
                    raw_answer,
                    citations,
                    citation_valid,
                    prompt_tokens,
                    completion_tokens,
                ) = _generate(client, model, results, row["question"])
            else:
                raw_answer, citations, citation_valid = "无答案", [], True
                prompt_tokens = completion_tokens = 0
            answer = raw_answer if citation_valid else "无答案"
            prediction.update(
                {
                    "cited_chunk_ids": citations,
                    "answered": not is_refusal(answer),
                    "answer": answer,
                    "citation_valid": citation_valid,
                }
            )
            if not citation_valid:
                prediction["raw_answer"] = raw_answer
                citation_validation_failures += 1
            input_tokens += prompt_tokens
            output_tokens += completion_tokens
            end_to_end_latencies.append((clock() - started) * 1000)
        predictions.append(prediction)

    stats = {
        "cases": len(predictions),
        "config": config,
        "split": split,
        "score_semantics": SCORE_SEMANTICS[config],
        "parameters": {
            "candidate_k_per_route": BENCHMARK_RETRIEVAL_K,
            "retrieval_k": BENCHMARK_RETRIEVAL_K,
            "context_k": CONTEXT_K,
            "threshold": threshold,
            "generate": generate,
            "input_usd_per_million": input_usd_per_million,
            "output_usd_per_million": output_usd_per_million,
        },
        "memory_mb": _memory_mb(),
        "retrieval_ms": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "mean": round(statistics.fmean(latencies), 3),
        },
    }
    if generate:
        stats["end_to_end_ms"] = {
            "p50": round(statistics.median(end_to_end_latencies), 3),
            "p95": round(_percentile(end_to_end_latencies, 0.95), 3),
            "mean": round(statistics.fmean(end_to_end_latencies), 3),
        }
        stats["token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        stats["citation_validation_failures"] = citation_validation_failures
        if rates[0] is not None:
            total_cost = (
                input_tokens * rates[0] + output_tokens * rates[1]
            ) / 1_000_000
            stats["cost_usd"] = {
                "total": round(total_cost, 8),
                "mean": round(total_cost / len(predictions), 8),
            }
    return predictions, stats


def load_config(index_dir, config):
    manifest, chunks = load_index_bundle(index_dir)
    routes = CONFIG_ROUTES[config]
    configured_manifest = {
        **manifest,
        "embedding_models": [
            entry
            for entry in manifest["embedding_models"]
            if Path(entry["index"]).stem in routes
        ],
    }
    retrievers, ranker = load_models(
        configured_manifest,
        chunks,
        index_dir,
        RERANKER_MODEL if config == "hybrid_rerank" else None,
        RERANKER_REVISION if config == "hybrid_rerank" else None,
    )
    return chunks, {route: retrievers[route] for route in routes}, ranker


def ensure_index(index_dir="index", doc_dir="doc", rebuild=False):
    from create_index import create_index

    kwargs = {"doc_dir": doc_dir, "index_dir": index_dir}
    if rebuild:
        kwargs["force"] = True
    create_index(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "top_score uses BM25 raw score for bm25, negative squared L2 distance for a "
            "single embedding, RRF score for dual_dense/hybrid, and reranker raw "
            "logit for hybrid_rerank. Calibrate thresholds separately for every "
            "config; never reuse them."
        ),
    )
    parser.add_argument("gold", help="gold JSONL")
    parser.add_argument(
        "--doc-dir",
        default="doc",
        help="documents used to build the index when it is missing (default: doc)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="rebuild the index from --doc-dir before benchmarking",
    )
    parser.add_argument("--config", required=True, choices=CONFIG_ROUTES)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--stats-output", help="write stats as atomic JSON")
    parser.add_argument(
        "--force", action="store_true", help="overwrite prediction/stats outputs"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="mark answered only at or above this config-specific top_score",
    )
    parser.add_argument(
        "--generate", action="store_true", help="also generate non-streaming answers"
    )
    parser.add_argument(
        "--input-usd-per-million",
        type=float,
        help="input-token rate; requires --generate and the output-token rate",
    )
    parser.add_argument(
        "--output-usd-per-million",
        type=float,
        help="output-token rate; requires --generate and the input-token rate",
    )
    args = parser.parse_args()
    if args.stats_output and Path(args.stats_output) == Path(args.output):
        parser.error("--output and --stats-output must be different files")
    protected_outputs = [args.output] + (
        [args.stats_output] if args.stats_output else []
    )
    if not args.force:
        existing = [path for path in protected_outputs if Path(path).exists()]
        if existing:
            parser.error("output already exists: " + ", ".join(existing))
    try:
        gold = load_gold(args.gold)
        index_dir = Path("index")
        ensure_index(index_dir, args.doc_dir, args.rebuild_index)
        chunks, retrievers, ranker = load_config(index_dir, args.config)
        client = None
        if args.generate:
            from openai import OpenAI

            client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                timeout=60,
                max_retries=2,
            )
        predictions, stats = run_benchmark(
            gold,
            chunks,
            retrievers,
            ranker=ranker,
            config=args.config,
            split=args.split,
            threshold=args.threshold,
            generate=args.generate,
            client=client,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
        )
        stats["provenance"] = {
            "gold_sha256": sha256_file(args.gold),
            "index_manifest_sha256": sha256_file(index_dir / "manifest.json"),
        }
        stats["models"] = {
            "llm": LLM_MODEL if args.generate else None,
            "reranker": (
                {"name": RERANKER_MODEL, "revision": RERANKER_REVISION}
                if args.config == "hybrid_rerank"
                else None
            ),
        }
        stats["devices"] = {
            "retrievers": {
                route: str(getattr(retriever, "device", "cpu"))
                for route, retriever in retrievers.items()
            },
            "reranker": str(ranker.device) if ranker is not None else None,
            "faiss": "cpu",
        }
        write_jsonl(predictions, args.output, args.force)
        stats["provenance"]["predictions_sha256"] = sha256_file(args.output)
        if args.stats_output:
            write_jsonl([stats], args.stats_output, args.force)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    if not args.stats_output:
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
