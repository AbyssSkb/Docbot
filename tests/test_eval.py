import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import eval as evaluator


def gold(
    question_id,
    *,
    answerable,
    chunks,
    split="dev",
    group_id=None,
):
    return {
        "question_id": question_id,
        "group_id": group_id or question_id,
        "question": f"Question {question_id}",
        "category": "fact" if answerable else "unanswerable",
        "split": split,
        "answerable": answerable,
        "gold_answer": "answer" if answerable else "",
        "gold_evidence": (
            [{"source": "doc.pdf", "section": None, "quote": "supporting quote"}]
            if answerable
            else []
        ),
        "gold_chunk_ids": chunks,
    }


def prediction(
    question_id,
    retrieved,
    cited,
    answered,
    score=None,
    answer_correct=None,
    faithful=None,
    include_answer=True,
):
    result = {
        "question_id": question_id,
        "retrieved_chunk_ids": retrieved,
        "cited_chunk_ids": cited,
        "answered": answered,
    }
    if include_answer:
        result["answer"] = "answer" if answered else "无答案"
    if score is not None:
        result["top_score"] = score
    if answer_correct is not None:
        result["answer_correct"] = answer_correct
    if faithful is not None:
        result["faithful"] = faithful
    return result


class EvaluationTest(unittest.TestCase):
    def test_gold_without_metadata_keeps_prediction_matching_when_reordered(self):
        rows = [
            gold("first", answerable=True, chunks=["a"]),
            gold("second", answerable=False, chunks=[]),
        ]
        for row in rows:
            for field in ("question_id", "category", "answerable"):
                del row[field]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.jsonl"
            payload = "\n".join(json.dumps(row) for row in rows)
            path.write_text(payload, encoding="utf-8")
            loaded = evaluator.load_gold(path)
            self.assertEqual([row["answerable"] for row in loaded], [True, False])
            self.assertEqual(path.read_text(encoding="utf-8"), payload)
            predictions = [
                prediction(loaded[0]["question_id"], ["a"], ["a"], True),
                prediction(loaded[1]["question_id"], [], [], False),
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in reversed(rows)),
                encoding="utf-8",
            )
            result = evaluator.evaluate(evaluator.load_gold(path), predictions, k=2)
            self.assertEqual(result["overall"]["retrieval"]["recall@2"], 1.0)
            self.assertEqual(set(result["by_category"]), {"uncategorized"})
            path.write_text(payload + "\n" + json.dumps(rows[0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate question_id"):
                evaluator.load_gold(path)

    def test_inferred_answerability_still_requires_consistent_evidence(self):
        for answer, chunks in (("answer", []), ("", ["a"])):
            row = gold("q1", answerable=True, chunks=chunks)
            row["gold_answer"] = answer
            del row["answerable"]
            del row["category"]
            with self.subTest(answer=answer, chunks=chunks):
                with self.assertRaisesRegex(ValueError, "items (require|need)"):
                    evaluator._validate_gold(row, "gold")

    def test_metrics_slices_all_gold_rows(self):
        gold_rows = [
            gold("q1", answerable=True, chunks=["a", "b"]),
            gold("q2", answerable=False, chunks=[]),
            gold("q3", answerable=True, chunks=["c"], split="test"),
        ]
        predictions = [
            prediction(
                "q1",
                ["x", "a", "b"],
                ["a"],
                True,
                answer_correct=1,
                faithful=1,
            ),
            prediction("q2", ["y"], [], False),
            prediction("q3", ["x"], ["x"], True),
        ]

        result = evaluator.evaluate(gold_rows, predictions, k=2)

        self.assertEqual(result["counts"]["gold_cases"], 3)
        self.assertEqual(result["counts"]["out_of_split_gold_ignored"], 0)
        self.assertEqual(result["metric_version"], 2)
        self.assertEqual(result["overall"]["retrieval"]["recall@2"], 0.25)
        self.assertEqual(result["overall"]["retrieval"]["hit@2"], 0.5)
        self.assertEqual(result["overall"]["retrieval"]["recall@20"], 0.5)
        self.assertEqual(result["overall"]["retrieval"]["recall@50"], 0.5)
        self.assertEqual(result["overall"]["retrieval"]["mrr@2"], 0.25)
        self.assertEqual(result["overall"]["citation"]["f1"], 0.4)
        self.assertEqual(result["overall"]["answerable"]["f1"], 1.0)
        self.assertEqual(result["overall"]["refusal"]["f1"], 1.0)
        self.assertEqual(result["overall"]["generation"]["answer_correct"]["mean"], 1.0)
        self.assertEqual(
            result["overall"]["generation"]["answer_correct"]["coverage_rate"],
            0.333333,
        )
        self.assertEqual(
            result["overall"]["generation"]["faithful"]["eligible_cases"], 2
        )
        self.assertEqual(
            result["overall"]["generation"]["faithful"]["coverage_rate"], 0.5
        )
        self.assertEqual(set(result["by_category"]), {"fact", "unanswerable"})
        self.assertEqual(set(result["by_split"]), {"dev", "test"})

    def test_retrieval_rewards_complete_evidence_beyond_the_first_hit(self):
        rows = [gold("q1", answerable=True, chunks=["a", "b", "c"])]
        partial = evaluator.evaluate(
            rows, [prediction("q1", ["a"], [], True)], k=3
        )["overall"]["retrieval"]
        complete = evaluator.evaluate(
            rows, [prediction("q1", ["a", "b", "c"], [], True)], k=3
        )["overall"]["retrieval"]

        self.assertEqual(partial["hit@3"], 1.0)
        self.assertEqual(partial["mrr@3"], 1.0)
        self.assertEqual(partial["recall@3"], 0.333333)
        self.assertEqual(partial["ndcg@3"], 0.469279)
        for metric in ("hit", "mrr", "recall", "ndcg"):
            self.assertEqual(complete[f"{metric}@3"], 1.0)

        # IDCG is bounded by k, but recall still needs all three gold chunks.
        top_one = evaluator.evaluate(
            rows, [prediction("q1", ["a", "b", "c"], [], True)], k=1
        )["overall"]["retrieval"]
        self.assertEqual(top_one["ndcg@1"], 1.0)
        self.assertEqual(top_one["recall@1"], 0.333333)

    def test_retrieval_macro_average_includes_misses(self):
        rows = [
            gold("multi", answerable=True, chunks=["a", "b", "c"]),
            gold("single", answerable=True, chunks=["d"]),
            gold("miss", answerable=True, chunks=["e"]),
            gold("refuse", answerable=False, chunks=[]),
        ]
        predictions = [
            prediction("multi", ["a"], [], True),
            prediction("single", ["d"], [], True),
            prediction("miss", [], [], False),
            prediction("refuse", [], [], False),
        ]
        result = evaluator.evaluate(rows, predictions, k=20)
        self.assertEqual(result["overall"]["retrieval_cases"], 3)
        self.assertEqual(result["overall"]["retrieval"]["recall@20"], 0.444444)
        self.assertEqual(result["overall"]["retrieval"]["hit@20"], 0.666667)
        self.assertIsNone(result["by_category"]["unanswerable"]["retrieval"]["ndcg@20"])

    def test_citations_count_chunks_and_missing_evidence_including_refusals(self):
        rows = [
            gold("q1", answerable=True, chunks=["a", "b", "c"]),
            gold("q2", answerable=True, chunks=["d"]),
        ]
        predictions = [
            prediction("q1", ["a", "b", "c", "x"], ["a", "b", "x"], True),
            prediction("q2", [], [], False),
        ]
        citation = evaluator.evaluate(rows, predictions)["overall"]["citation"]
        # TP=2, FP=1, FN=2: two correct citations must count twice.
        self.assertEqual(citation, {"precision": 0.666667, "recall": 0.5, "f1": 0.571429})

    def test_citation_must_be_in_actual_context_and_context_must_be_retrieved(self):
        chunk_id = "chunk_" + "a" * 64
        row = prediction("q1", [chunk_id], [chunk_id], False) | {
            "context_chunk_ids": [],
            "raw_answer": f"答案 [{chunk_id}]",
            "citation_valid": False,
        }
        evaluator._validate_prediction(row, "prediction")
        result = evaluator.evaluate(
            [gold("q1", answerable=True, chunks=[chunk_id])], [row]
        )["overall"]
        self.assertEqual(result["retrieval"]["recall@10"], 1.0)
        self.assertEqual(result["citation"]["f1"], 0.0)

        with self.assertRaisesRegex(ValueError, "context_chunk_ids.*retrieved"):
            evaluator._validate_prediction(
                row | {"context_chunk_ids": ["unretrieved"]}, "prediction"
            )
        with self.assertRaisesRegex(ValueError, "context_chunk_ids requires answer"):
            evaluator._validate_prediction(
                prediction("q1", [], [], False, include_answer=False)
                | {"context_chunk_ids": []}, "prediction"
            )

    def test_removed_review_fields_are_rejected_and_unretrieved_citation_is_wrong(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gold_path = directory / "gold.jsonl"
            prediction_path = directory / "predictions.jsonl"
            for field in ("status", "reviewer"):
                gold_path.write_text(
                    json.dumps(
                        {**gold("q1", answerable=True, chunks=["a"]), field: "old"}
                    )
                    + "\n",
                    encoding="utf-8",
                )

                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, f"unknown fields: {field}"):
                        evaluator.load_gold(gold_path)

            prediction_path.write_text(
                json.dumps(prediction("q1", ["x"], ["a"], True))
                + "\n",
                encoding="utf-8",
            )

            loaded = evaluator.load_predictions(prediction_path)
            result = evaluator.evaluate(
                [gold("q1", answerable=True, chunks=["a"])], loaded
            )
            self.assertEqual(result["overall"]["citation"]["precision"], 0.0)
            self.assertEqual(result["overall"]["citation"]["recall"], 0.0)

    def test_answerable_gold_requires_complete_evidence(self):
        with self.assertRaisesRegex(ValueError, "answerable items require"):
            evaluator._validate_gold(gold("q1", answerable=True, chunks=[]), "gold")

    def test_retrieval_only_run_does_not_claim_citation_metrics(self):
        gold_rows = [gold("q1", answerable=True, chunks=["a"])]
        predictions = [
            prediction("q1", ["a"], [], True, include_answer=False)
        ]

        result = evaluator.evaluate(gold_rows, predictions)

        self.assertEqual(result["overall"]["citation_cases"], 0)
        self.assertIsNone(result["overall"]["citation"]["f1"])
        self.assertEqual(
            result["overall"]["generation"]["answer_correct"]["eligible_cases"],
            0,
        )
        self.assertEqual(
            result["overall"]["generation"]["faithful"]["eligible_cases"], 0
        )

        unranked = prediction("q1", ["a"], [], True, include_answer=False)
        unranked["ranked"] = False
        result = evaluator.evaluate(gold_rows, [unranked])
        self.assertEqual(result["overall"]["retrieval_cases"], 0)
        self.assertIsNone(result["overall"]["retrieval"]["mrr@10"])

        with self.assertRaisesRegex(ValueError, "answered does not match"):
            evaluator._validate_prediction(
                prediction("bad", [], [], False, include_answer=True)
                | {"answer": "实际回答"},
                "prediction",
            )
        with self.assertRaisesRegex(ValueError, "faithful requires an answered"):
            evaluator._validate_prediction(
                prediction("bad-faithful", [], [], False, faithful=1),
                "prediction",
            )

    def test_gold_mapping_matches_source_and_section_locations(self):
        chunks = [
            {
                "chunk_id": "pdf",
                "source": "doc.pdf",
                "section": None,
                "text": "prefix supporting\nquote suffix",
            },
            {
                "chunk_id": "section",
                "source": "guide.md",
                "section": "Intro",
                "text": "section fact",
            },
            {
                "chunk_id": "document",
                "source": "plain.txt",
                "section": None,
                "text": "document fact",
            },
        ]
        pdf = gold("pdf", answerable=True, chunks=["pdf"])
        section = gold("section", answerable=True, chunks=["section"]) | {
            "gold_evidence": [
                    {"source": "guide.md", "section": "Intro", "quote": "section fact"}
            ]
        }
        document = gold("document", answerable=True, chunks=["document"]) | {
            "gold_evidence": [
                {
                    "source": "plain.txt",
                    "section": None,
                    "quote": "document fact",
                }
            ]
        }

        evaluator.validate_gold_mapping([pdf, section, document], chunks)
        with self.assertRaisesRegex(ValueError, "gold_evidence does not match"):
            evaluator.validate_gold_mapping(
                [pdf | {"gold_evidence": [{"source": "doc.pdf", "section": None, "quote": "missing"}]}],
                chunks,
            )
        with self.assertRaisesRegex(ValueError, "not supported by gold_evidence"):
            evaluator.validate_gold_mapping(
                [pdf | {"gold_chunk_ids": ["pdf", "section"]}], chunks
            )

    def test_split_can_be_evaluated_without_other_split_predictions(self):
        gold_rows = [
            gold("dev", answerable=True, chunks=["a"]),
            gold("test", answerable=True, chunks=["b"], split="test"),
        ]
        predictions = [prediction("dev", ["a"], ["a"], True)]

        result = evaluator.evaluate(gold_rows, predictions, split="dev")

        self.assertEqual(result["overall"]["cases"], 1)
        self.assertEqual(result["counts"]["out_of_split_gold_ignored"], 1)

    def test_gold_rejects_a_group_crossing_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        gold(
                            "dev",
                            answerable=True,
                            chunks=["a"],
                            group_id="same-fact",
                        ),
                        gold(
                            "test",
                            answerable=True,
                            chunks=["b"],
                            split="test",
                            group_id="same-fact",
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "crosses dev/test"):
                evaluator.load_gold(path)

    def test_threshold_is_selected_on_dev_and_frozen_for_test(self):
        gold_rows = [
            gold("d-answer", answerable=True, chunks=["a"]),
            gold("d-refuse", answerable=False, chunks=[]),
            gold("t-answer", answerable=True, chunks=["b"], split="test"),
            gold("t-refuse", answerable=False, chunks=[], split="test"),
        ]
        predictions = [
            prediction("d-answer", ["a"], ["a"], True, 0.9),
            prediction("d-refuse", ["x"], [], False, 0.2),
            prediction("t-answer", ["b"], ["b"], True, 0.8),
            prediction("t-refuse", [], [], False, 0.3),
        ]

        result = evaluator.calibrate_threshold(gold_rows, predictions)

        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["dev"]["refusal"]["f1"], 1.0)
        self.assertNotIn("test", result)

    def test_calibration_always_refuses_empty_retrieval_even_with_high_score(self):
        rows = [
            gold("answer", answerable=True, chunks=["a"]),
            gold("refuse", answerable=False, chunks=[]),
        ]
        predictions = [
            prediction("answer", ["a"], [], True, -0.5),
            prediction("refuse", [], [], False, 0.0),
        ]
        result = evaluator.calibrate_threshold(rows, predictions)
        self.assertEqual(result["threshold"], -0.5)
        self.assertEqual(result["dev"]["refusal"]["f1"], 1.0)

    def test_index_validation_rejects_stale_gold_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc_id = "doc_" + "a" * 64
            chunk_id = "chunk_" + "b" * 64
            chunks = [
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source": "doc.pdf",
                    "section": None,
                    "text": "supporting quote",
                }
            ]
            payload = json.dumps(chunks).encode()
            (root / "chunks.json").write_bytes(payload)
            (root / "embed1.index").write_bytes(b"one")
            (root / "embed2.index").write_bytes(b"two")
            manifest = {
                "schema": {
                    "name": "docbot-index",
                    "version": 2,
                    "chunk_fields": [
                        "chunk_id",
                        "doc_id",
                        "source",
                        "section",
                        "text",
                    ],
                },
                "embedding_models": [
                    {
                        "index": "embed1.index",
                        "name": "one",
                        "revision": "rev-one",
                    },
                    {
                        "index": "embed2.index",
                        "name": "two",
                        "revision": "rev-two",
                    },
                ],
                "documents": [
                    {
                        "doc_id": doc_id,
                        "source": "doc.pdf",
                        "sha256": "c" * 64,
                    }
                ],
                "counts": {"documents": 1, "chunks": 1},
                "artifacts": {
                    "chunks": {
                        "path": "chunks.json",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    "embed1": {
                        "path": "embed1.index",
                        "sha256": hashlib.sha256(b"one").hexdigest(),
                    },
                    "embed2": {
                        "path": "embed2.index",
                        "sha256": hashlib.sha256(b"two").hexdigest(),
                    },
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "do not match this index"):
                evaluator.validate_index_mapping(
                    [gold("q1", answerable=True, chunks=["chunk_" + "d" * 64])],
                    [prediction("q1", [chunk_id], [], True)],
                    root,
                )


if __name__ == "__main__":
    unittest.main()
