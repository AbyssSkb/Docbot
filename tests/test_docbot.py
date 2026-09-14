import unittest
from unittest.mock import patch

import docbot


class DocbotCliTest(unittest.TestCase):
    @patch("docbot.retrieve")
    @patch("docbot.load_retrieval_config")
    @patch("docbot.load_index_bundle")
    def test_search_index_returns_agent_friendly_json(
        self, load_bundle, load_config, retrieve
    ):
        chunk = {
            "chunk_id": "chunk_" + "a" * 64,
            "source": "policy.txt",
            "section": "退款",
            "text": "签收后七天内可以退款。",
        }
        manifest = {"counts": {"documents": 1, "chunks": 1}}
        load_bundle.return_value = manifest, [chunk]
        retrievers = {"bm25": object()}
        load_config.return_value = retrievers, None
        retrieve.return_value = [
            {
                **chunk,
                "rrf_score": 0.1,
                "routes": [{"route": "bm25", "rank": 1, "raw_score": 2.0}],
            }
        ]

        result = docbot.search_index("  退款期限？ ", top_k=3, candidate_k=7)

        self.assertEqual(result["query"], "退款期限？")
        self.assertEqual(result["results"][0]["chunk_id"], chunk["chunk_id"])
        self.assertEqual(result["results"][0]["text"], chunk["text"])
        retrieve.assert_called_once_with(
            "退款期限？",
            [chunk],
            retrievers,
            ranker=None,
            candidate_k=7,
            result_k=3,
        )
        load_config.assert_called_once_with(manifest, [chunk], "index", "hybrid")

    @patch("docbot.retrieve")
    @patch("docbot.load_retrieval_config", return_value=({"bm25": object()}, None))
    @patch("docbot.load_index_bundle")
    def test_search_index_passes_requested_config(
        self, load_bundle, load_config, retrieve
    ):
        chunk = {
            "chunk_id": "chunk_" + "a" * 64,
            "source": "policy.txt",
            "section": "退款",
            "text": "签收后七天内可以退款。",
        }
        manifest = {"counts": {"documents": 1, "chunks": 1}}
        load_bundle.return_value = manifest, [chunk]
        retrieve.return_value = [
            {
                **chunk,
                "rrf_score": 0.1,
                "routes": [{"route": "bm25", "rank": 1, "raw_score": 2.0}],
            }
        ]

        result = docbot.search_index("退款", retrieval_config="bm25")

        self.assertEqual(result["retrieval_config"], "bm25")
        self.assertEqual(result["results"][0]["score"], 2.0)
        load_config.assert_called_once_with(manifest, [chunk], "index", "bm25")


if __name__ == "__main__":
    unittest.main()
