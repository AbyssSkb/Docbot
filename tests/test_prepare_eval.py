import tempfile
import unittest
from pathlib import Path

import eval as evaluator
import prepare_eval


def chunk(chunk_id, source, text, section=None):
    return {
        "chunk_id": chunk_id,
        "source": source,
        "section": section,
        "text": text,
    }


class PrepareEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("a1", "a.pdf", "alpha one"),
            chunk("a2", "a.pdf", "alpha two"),
            chunk("a3", "a.pdf", "alpha three"),
            chunk("b1", "b.pdf", "beta one"),
            chunk("b2", "b.pdf", "beta two"),
            chunk("c1", "c.md", "gamma one", section="Intro"),
        ]

    def test_sampling_is_deterministic_and_round_robins_sources(self):
        first = prepare_eval.sample_chunks(self.chunks, 3, seed=7)
        second = prepare_eval.sample_chunks(list(reversed(self.chunks)), 3, seed=7)

        self.assertEqual(
            [item["chunk_id"] for item in first],
            [item["chunk_id"] for item in second],
        )
        self.assertEqual({item["source"] for item in first}, {"a.pdf", "b.pdf", "c.md"})

    def test_candidates_are_evidence_first_but_not_gold(self):
        records = prepare_eval.build_candidates(
            self.chunks, count=2, seed=11, unanswerable=1
        )

        self.assertEqual(len(records), 3)
        self.assertTrue(records[0]["question"].startswith("【待人工填写】"))
        self.assertNotIn("status", records[0])
        self.assertNotIn("reviewer", records[0])
        self.assertEqual(records[0]["gold_answer"], "")
        self.assertEqual(len(records[0]["gold_evidence"]), 1)
        self.assertEqual(records[-1]["gold_evidence"], [])
        self.assertFalse(records[-1]["answerable"])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidates.jsonl"
            prepare_eval.write_jsonl(records, output)
            with self.assertRaisesRegex(ValueError, "answerable items require"):
                evaluator.load_gold(output)
            with self.assertRaises(FileExistsError):
                prepare_eval.write_jsonl(records, output)
            prepare_eval.write_jsonl(records[:1], output, overwrite=True)

    def test_split_depends_on_fact_not_sampling_seed(self):
        candidate = prepare_eval.candidate_from_chunk(self.chunks[0])
        again = prepare_eval.candidate_from_chunk(self.chunks[0])

        self.assertEqual(candidate["question_id"], again["question_id"])
        self.assertEqual(candidate["group_id"], again["group_id"])
        self.assertEqual(candidate["split"], again["split"])
        self.assertIn(candidate["split"], {"dev", "test"})


if __name__ == "__main__":
    unittest.main()
