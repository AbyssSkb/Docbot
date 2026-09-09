import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import benchmark
from eval import load_predictions


class BenchmarkTest(unittest.TestCase):
    def test_generation_unwraps_decision_without_changing_benchmark_citation_rules(self):
        chunk_id = "chunk_" + "a" * 64
        contexts = [{"chunk_id": chunk_id, "source": "policy.txt", "section": None, "text": "七天"}]
        for action, answer, expected, valid in [
            ("answer", f"七天 [{chunk_id}]", f"七天 [{chunk_id}]", True),
            ("search", "退款期限", "无答案", True),
            ("answer", "你好", "你好", False),
        ]:
            with self.subTest(action=action):
                client = Mock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content=json.dumps({"action": action, "query" if action == "search" else "answer": answer})
                    ))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                )
                generated, citations, passed, inputs, outputs = benchmark._generate(
                    client, "test-model", contexts, "退款期限"
                )
                self.assertEqual(generated, expected)
                self.assertEqual(passed, valid)
                self.assertEqual(citations, [chunk_id] if valid and action == "answer" else [])
                self.assertEqual((inputs, outputs), (10, 5))

    @patch("create_index.create_index")
    def test_ensure_index_delegates_incremental_check_to_indexer(
        self, create_index
    ):
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory) / "index"
            doc_dir = Path(directory) / "docs"
            benchmark.ensure_index(index_dir, doc_dir)
            create_index.assert_called_once_with(doc_dir=doc_dir, index_dir=index_dir)

            create_index.reset_mock()
            index_dir.mkdir()
            for name in ("manifest.json", "chunks.json", "embed1.index", "embed2.index"):
                (index_dir / name).write_bytes(b"artifact")
            benchmark.ensure_index(index_dir, doc_dir)
            create_index.assert_called_once_with(doc_dir=doc_dir, index_dir=index_dir)

            create_index.reset_mock()
            benchmark.ensure_index(index_dir, doc_dir, rebuild=True)
            create_index.assert_called_once_with(
                doc_dir=doc_dir,
                index_dir=index_dir,
                force=True,
            )

    @patch("benchmark.load_models")
    @patch("benchmark.load_index_bundle")
    def test_load_config_loads_only_requested_embedding(
        self, load_index_bundle, load_models
    ):
        manifest = {
            "embedding_models": [
                {"index": "embed1.index", "name": "one"},
                {"index": "embed2.index", "name": "two"},
            ]
        }
        chunks = [{"text": "chunk"}]
        embed1 = object()
        load_index_bundle.return_value = manifest, chunks
        load_models.return_value = {"embed1": embed1, "bm25": object()}, None

        loaded_chunks, retrievers, ranker = benchmark.load_config("index", "embed1")

        configured_manifest = load_models.call_args.args[0]
        self.assertEqual(
            configured_manifest["embedding_models"],
            [{"index": "embed1.index", "name": "one"}],
        )
        self.assertIsNone(load_models.call_args.args[3])
        self.assertIsNone(load_models.call_args.args[4])
        self.assertEqual(loaded_chunks, chunks)
        self.assertEqual(retrievers, {"embed1": embed1})
        self.assertIsNone(ranker)

        load_models.reset_mock()

    @patch("pipeline.build_messages", create=True)
    @patch("benchmark.retrieve")
    def test_generate_rejects_bad_citation_but_keeps_audit_and_cost(
        self, retrieve, build_messages
    ):
        valid_id = "chunk_" + "a" * 64
        invalid_id = "chunk_" + "b" * 64
        answer = f"答案 [{valid_id}]，延伸 [{invalid_id}]"
        retrieve.return_value = [
            {"chunk_id": valid_id, "rerank_score": 0.8, "text": "evidence"}
        ]
        build_messages.return_value = [{"role": "user", "content": "prompt"}]
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"action": "answer", "answer": answer})
            ))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )
        client = Mock()
        client.chat.completions.create.return_value = response
        retrievers = {route: object() for route in benchmark.CONFIG_ROUTES["hybrid"]}
        times = iter((1.0, 1.01, 1.03))

        predictions, stats = benchmark.run_benchmark(
            [{"question_id": "q1", "question": "question"}],
            [],
            retrievers,
            ranker=object(),
            config="hybrid_rerank",
            threshold=0.5,
            generate=True,
            client=client,
            input_usd_per_million=2,
            output_usd_per_million=4,
            clock=lambda: next(times),
        )

        self.assertEqual(predictions[0]["answer"], "无答案")
        self.assertEqual(predictions[0]["raw_answer"], answer)
        self.assertEqual(predictions[0]["cited_chunk_ids"], [valid_id, invalid_id])
        self.assertFalse(predictions[0]["answered"])
        self.assertFalse(predictions[0]["citation_valid"])
        self.assertEqual(stats["end_to_end_ms"]["mean"], 30.0)
        self.assertEqual(
            stats["token_usage"],
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
        self.assertEqual(stats["citation_validation_failures"], 1)
        self.assertEqual(stats["cost_usd"], {"total": 0.00028, "mean": 0.00028})
        client.chat.completions.create.assert_called_once_with(
            model=benchmark.LLM_MODEL,
            messages=build_messages.return_value,
            stream=False,
            temperature=0,
        )
        build_messages.assert_called_once_with("question", retrieve.return_value)
        self.assertNotIn("min_rerank_score", retrieve.call_args.kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.jsonl"
            benchmark.write_jsonl(predictions, path)
            self.assertEqual(load_predictions(path), predictions)

    @patch("benchmark.retrieve")
    def test_generate_skips_llm_below_threshold(self, retrieve):
        retrieve.return_value = [
            {"chunk_id": "chunk_1", "rerank_score": 0.2, "text": "evidence"}
        ]
        client = Mock()
        retrievers = {route: object() for route in benchmark.CONFIG_ROUTES["hybrid"]}
        times = iter((1.0, 1.01, 1.011))

        predictions, _ = benchmark.run_benchmark(
            [{"question_id": "q1", "question": "question"}],
            [],
            retrievers,
            ranker=object(),
            config="hybrid_rerank",
            threshold=0.5,
            generate=True,
            client=client,
            clock=lambda: next(times),
        )

        self.assertEqual(predictions[0]["answer"], "无答案")
        self.assertFalse(predictions[0]["answered"])
        self.assertTrue(predictions[0]["citation_valid"])
        self.assertEqual(predictions[0]["top_score"], 0.2)
        self.assertEqual(predictions[0]["retrieved_chunk_ids"], ["chunk_1"])
        self.assertEqual(predictions[0]["context_chunk_ids"], [])
        client.chat.completions.create.assert_not_called()
        self.assertNotIn("min_rerank_score", retrieve.call_args.kwargs)

        self.assertTrue(benchmark.is_refusal("无答案。"))

    @patch("benchmark._generate")
    @patch("benchmark.retrieve")
    def test_generation_keeps_full_ranking_separate_from_ten_chunk_context(
        self, retrieve, generate
    ):
        retrieve.return_value = [
            {
                "chunk_id": f"chunk_{i}",
                "routes": [{"raw_score": 1.0}],
                "text": "evidence",
            }
            for i in range(50)
        ]
        generate.return_value = ("无答案", [], True, 0, 0)
        rows = [{"question_id": "q1", "question": "question"}]
        retrieved, _ = benchmark.run_benchmark(
            rows, [], {"bm25": object()}, config="bm25"
        )
        generated, _ = benchmark.run_benchmark(
            rows, [], {"bm25": object()}, config="bm25",
            generate=True, client=Mock(),
        )
        self.assertEqual(generated[0]["retrieved_chunk_ids"], retrieved[0]["retrieved_chunk_ids"])
        self.assertEqual(generated[0]["context_chunk_ids"], [f"chunk_{i}" for i in range(10)])
        self.assertEqual(generate.call_args.args[2], retrieve.return_value[:10])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.jsonl"
            benchmark.write_jsonl(generated, path)
            self.assertEqual(load_predictions(path), generated)

    def test_selected_split_predictions_are_eval_compatible(self):
        gold = [
            {
                "question_id": "q1",
                "question": "dev",
                "split": "dev",
            },
            {
                "question_id": "q3",
                "question": "untouched",
                "split": "test",
            },
        ]
        retrievers = {"bm25": object(), "embed1": object()}
        result = {
            "chunk_id": "chunk_1",
            "rrf_score": 1 / 61,
            "routes": [{"route": "bm25", "rank": 1, "raw_score": 2.5}],
        }
        times = iter((1.0, 1.01))

        with patch("benchmark.retrieve", return_value=[result]) as retrieve:
            predictions, stats = benchmark.run_benchmark(
                gold,
                [],
                retrievers,
                config="bm25",
                split="dev",
                threshold=2.0,
                clock=lambda: next(times),
            )

        self.assertEqual(
            predictions,
            [
                {
                    "question_id": "q1",
                    "retrieved_chunk_ids": ["chunk_1"],
                    "cited_chunk_ids": [],
                    "answered": True,
                    "top_score": 2.5,
                }
            ],
        )
        self.assertEqual(retrieve.call_args.args[2], {"bm25": retrievers["bm25"]})
        self.assertIsNone(retrieve.call_args.args[3])
        self.assertEqual(retrieve.call_args.kwargs, {"candidate_k": 50, "result_k": 50})
        self.assertGreater(stats.pop("memory_mb")["peak_rss"], 0)
        self.assertEqual(
            stats,
            {
                "cases": 1,
                "config": "bm25",
                "split": "dev",
                "score_semantics": "BM25 raw score; higher is better",
                "parameters": {
                    "candidate_k_per_route": 50,
                    "retrieval_k": 50,
                    "context_k": 10,
                    "threshold": 2.0,
                    "generate": False,
                    "input_usd_per_million": None,
                    "output_usd_per_million": None,
                },
                "retrieval_ms": {"p50": 10.0, "p95": 10.0, "mean": 10.0},
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            benchmark.write_jsonl(predictions, path)
            self.assertEqual(load_predictions(path), predictions)
            with self.assertRaises(FileExistsError):
                benchmark.write_jsonl(predictions, path)

    def test_stale_gold_mapping_is_rejected_before_running(self):
        with self.assertRaisesRegex(ValueError, "do not match this index"):
            benchmark.run_benchmark(
                [
                    {
                        "question_id": "q1",
                        "question": "question",
                        "split": "dev",
                        "answerable": True,
                        "gold_chunk_ids": ["old"],
                    }
                ],
                [{"chunk_id": "current", "text": "evidence"}],
                {"bm25": object()},
                config="bm25",
            )


if __name__ == "__main__":
    unittest.main()
