import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from model import EmbeddingModel, RerankingModel, reciprocal_rank_fusion


class FakeIndex:
    ntotal = 2

    def search(self, _embedding, k):
        self.k = k
        return (
            np.array([[0.1, 0.2]], dtype="float32")[:, :k],
            np.array([[1, 0]])[:, :k],
        )


class ModelTest(unittest.TestCase):
    @patch("sentence_transformers.CrossEncoder")
    def test_qwen_reranker_uses_cross_encoder(self, cross_encoder):
        ranker = RerankingModel(
            "Qwen/Qwen3-Reranker-0.6B",
            revision="rev",
        )

        cross_encoder.assert_called_once()
        self.assertEqual(
            cross_encoder.call_args.kwargs["device"], str(ranker.device)
        )

    def test_reranker_rejects_non_qwen_model(self):
        with self.assertRaises(ValueError):
            RerankingModel("unsupported/reranker")

    def test_embedding_query_caps_k_and_uses_query_prompt(self):
        model = EmbeddingModel.__new__(EmbeddingModel)
        model.index = FakeIndex()
        model.query_prompt_name = "query"
        seen = []
        model.embed_text = (
            lambda text, prompt_name=None: seen.append((text, prompt_name))
            or [1.0, 0.0]
        )

        self.assertEqual(
            model.query("问题", k=20),
            [
                {"index": 1, "raw_score": np.float32(0.1)},
                {"index": 0, "raw_score": np.float32(0.2)},
            ],
        )
        self.assertEqual(seen, [("问题", "query")])
        self.assertEqual(model.index.k, 2)

        model.index = None
        self.assertEqual(model.query("问题"), [])

    def test_rrf_keeps_routes_and_breaks_ties_by_index(self):
        fused = reciprocal_rank_fusion(
            {
                "dense": [
                    {"index": 2, "raw_score": 0.1},
                    {"index": 1, "raw_score": 0.2},
                ],
                "bm25": [
                    {"index": 1, "raw_score": 9.0},
                    {"index": 2, "raw_score": 8.0},
                ],
            }
        )

        self.assertEqual([item["index"] for item in fused], [1, 2])
        self.assertEqual(
            fused[0]["routes"],
            [
                {"route": "dense", "rank": 2, "raw_score": 0.2},
                {"route": "bm25", "rank": 1, "raw_score": 9.0},
            ],
        )

    def test_rrf_supports_fixed_route_weights(self):
        fused = reciprocal_rank_fusion(
            {
                "embed1": [{"index": 1, "raw_score": 0.1}],
                "bm25": [{"index": 2, "raw_score": 0.2}],
            },
            route_weights={"embed1": 8.0, "bm25": 1.0},
        )

        self.assertEqual([item["index"] for item in fused], [1, 2])

    def test_reranker_preserves_candidate_and_filters_calibrated_score(self):
        ranker = RerankingModel.__new__(RerankingModel)
        ranker.device = torch.device("cpu")
        ranker.model = Mock()
        ranker.model.predict.return_value = np.array([0.2, 0.8], dtype="float32")
        candidates = [
            {"text": "低分", "metadata": {"section": "低分"}},
            {"text": "高分", "metadata": {"section": "高分"}},
        ]

        result = ranker.select("问题", candidates, min_score=0.5)
        self.assertEqual(result[0]["text"], "高分")
        self.assertAlmostEqual(result[0]["rerank_score"], 0.8)
        ranker.model.predict.assert_called_once()
        self.assertEqual(ranker.model.predict.call_args.kwargs["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
