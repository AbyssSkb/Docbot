#!/usr/bin/env python3
"""Evaluate retrieval, citations, and refusal decisions from JSONL files."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from create_index import sha256_file


GOLD_REQUIRED = {
    "group_id",
    "question",
    "split",
    "gold_answer",
    "gold_evidence",
    "gold_chunk_ids",
}
GOLD_OPTIONAL = {"notes", "question_id", "category", "answerable"}
PREDICTION_REQUIRED = {
    "question_id",
    "retrieved_chunk_ids",
    "cited_chunk_ids",
    "answered",
}
PREDICTION_OPTIONAL = {
    "context_chunk_ids",
    "ranked",
    "top_score",
    "answer",
    "raw_answer",
    "citation_valid",
    "answer_correct",
    "faithful",
}
SPLITS = {"dev", "test"}
INDEX_DIR = Path("index")


def _nonempty_string(value, field, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty string")


def _string_list(value, field, location):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{location}: {field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{location}: {field} must not contain duplicates")


def _check_fields(record, required, optional, location):
    if not isinstance(record, dict):
        raise ValueError(f"{location}: each line must be a JSON object")
    missing = required - record.keys()
    unknown = record.keys() - required - optional
    if missing:
        raise ValueError(f"{location}: missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{location}: unknown fields: {', '.join(sorted(unknown))}")


def _validate_gold(record, location):
    _check_fields(record, GOLD_REQUIRED, GOLD_OPTIONAL, location)
    record.setdefault("category", "uncategorized")
    for field in ("group_id", "question", "category"):
        _nonempty_string(record[field], field, location)
    if "question_id" not in record:
        record["question_id"] = "q_" + hashlib.sha256(
            record["question"].encode("utf-8")
        ).hexdigest()
    _nonempty_string(record["question_id"], "question_id", location)
    if record["split"] not in SPLITS:
        raise ValueError(f"{location}: split must be dev or test")
    if not isinstance(record["gold_answer"], str):
        raise ValueError(f"{location}: gold_answer must be a string")
    record.setdefault("answerable", bool(record["gold_answer"].strip()))
    if type(record["answerable"]) is not bool:
        raise ValueError(f"{location}: answerable must be a boolean")
    if not isinstance(record["gold_evidence"], list):
        raise ValueError(f"{location}: gold_evidence must be a list")
    evidence_seen = set()
    for index, evidence in enumerate(record["gold_evidence"]):
        evidence_location = f"{location}:gold_evidence[{index}]"
        _check_fields(
            evidence, {"source", "section", "quote"}, set(), evidence_location
        )
        _nonempty_string(evidence["source"], "source", evidence_location)
        if evidence["section"] is not None and (
            not isinstance(evidence["section"], str) or not evidence["section"].strip()
        ):
            raise ValueError(
                f"{evidence_location}: section must be null or a non-empty string"
            )
        _nonempty_string(evidence["quote"], "quote", evidence_location)
        evidence_key = (evidence["source"], evidence["section"], evidence["quote"])
        if evidence_key in evidence_seen:
            raise ValueError(f"{location}: gold_evidence must not contain duplicates")
        evidence_seen.add(evidence_key)
    _string_list(record["gold_chunk_ids"], "gold_chunk_ids", location)
    if not record["answerable"] and (
        record["gold_answer"].strip()
        or record["gold_evidence"]
        or record["gold_chunk_ids"]
    ):
        raise ValueError(
            f"{location}: unanswerable items need empty gold_answer, gold_evidence, "
            "and gold_chunk_ids"
        )
    for field in (GOLD_OPTIONAL - {"answerable"}) & record.keys():
        if not isinstance(record[field], str):
            raise ValueError(f"{location}: {field} must be a string")
    if record["answerable"] and (
        not record["gold_answer"].strip()
        or not record["gold_evidence"]
        or not record["gold_chunk_ids"]
    ):
        raise ValueError(
            f"{location}: answerable items require gold_answer, gold_evidence, "
            "and gold_chunk_ids"
        )


def _validate_prediction(record, location):
    _check_fields(record, PREDICTION_REQUIRED, PREDICTION_OPTIONAL, location)
    _nonempty_string(record["question_id"], "question_id", location)
    _string_list(record["retrieved_chunk_ids"], "retrieved_chunk_ids", location)
    _string_list(record["cited_chunk_ids"], "cited_chunk_ids", location)
    if "context_chunk_ids" in record:
        _string_list(record["context_chunk_ids"], "context_chunk_ids", location)
        if "answer" not in record:
            raise ValueError(f"{location}: context_chunk_ids requires answer")
        if not set(record["context_chunk_ids"]) <= set(record["retrieved_chunk_ids"]):
            raise ValueError(f"{location}: context_chunk_ids must be retrieved")
    if type(record["answered"]) is not bool:
        raise ValueError(f"{location}: answered must be a boolean")
    if "ranked" in record and type(record["ranked"]) is not bool:
        raise ValueError(f"{location}: ranked must be a boolean")
    if "answer" in record:
        _nonempty_string(record["answer"], "answer", location)
        from pipeline import is_refusal

        if record["answered"] == is_refusal(record["answer"]):
            raise ValueError(f"{location}: answered does not match answer")
    if "citation_valid" in record:
        if type(record["citation_valid"]) is not bool:
            raise ValueError(f"{location}: citation_valid must be a boolean")
        if "answer" not in record:
            raise ValueError(f"{location}: citation_valid requires answer")
    if "raw_answer" in record:
        _nonempty_string(record["raw_answer"], "raw_answer", location)
        if record.get("citation_valid") is not False:
            raise ValueError(
                f"{location}: raw_answer requires failed citation validation"
            )
    if "citation_valid" in record:
        from pipeline import validate_answer_citations

        checked_answer = record.get("raw_answer", record["answer"])
        valid, invalid, actual = validate_answer_citations(
            checked_answer,
            record.get("context_chunk_ids", record["retrieved_chunk_ids"]),
        )
        if set(record["cited_chunk_ids"]) != set(valid + invalid):
            raise ValueError(f"{location}: cited_chunk_ids do not match the answer")
        if record["citation_valid"] != actual:
            raise ValueError(f"{location}: citation_valid does not match citations")
        if not record["citation_valid"] and record["answered"]:
            raise ValueError(f"{location}: failed citation validation must refuse")
    if "top_score" in record and (
        isinstance(record["top_score"], bool)
        or not isinstance(record["top_score"], (int, float))
        or not math.isfinite(record["top_score"])
    ):
        raise ValueError(f"{location}: top_score must be a finite number")
    for field in ("answer_correct", "faithful"):
        if field in record:
            if not (
                isinstance(record[field], bool)
                or type(record[field]) in (int, float)
                and record[field] in (0, 1)
            ):
                raise ValueError(f"{location}: {field} must be a boolean or 0/1")
            if "answer" not in record:
                raise ValueError(f"{location}: {field} requires answer")
            if field == "faithful" and not record["answered"]:
                raise ValueError(f"{location}: faithful requires an answered output")


def _load_jsonl(path, validator):
    records = []
    seen = set()
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
            validator(record, location)
            question_id = record["question_id"]
            if question_id in seen:
                raise ValueError(f"{location}: duplicate question_id {question_id!r}")
            seen.add(question_id)
            records.append(record)
    return records


def load_gold(path):
    records = _load_jsonl(path, _validate_gold)
    group_splits = {}
    for record in records:
        group_id = record["group_id"]
        previous = group_splits.setdefault(group_id, record["split"])
        if previous != record["split"]:
            raise ValueError(f"group_id {group_id!r} crosses dev/test splits")
    return records


def load_predictions(path):
    return _load_jsonl(path, _validate_prediction)


def write_jsonl(records, path, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            Path(temporary).unlink()
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _normalized_text(text):
    return " ".join(text.split())


def _evidence_matches_chunk(evidence, chunk):
    if evidence["source"] != chunk["source"]:
        return False
    location_matches = evidence["section"] == chunk.get("section")
    return location_matches and _normalized_text(evidence["quote"]) in _normalized_text(
        chunk["text"]
    )


def validate_gold_mapping(gold_records, chunks, split="all"):
    known = {chunk["chunk_id"]: chunk for chunk in chunks}
    scoped_gold = [
        record
        for record in gold_records
        if (split == "all" or record["split"] == split)
        and record.get("answerable") is True
    ]
    stale_gold = {
        chunk_id
        for record in scoped_gold
        for chunk_id in record["gold_chunk_ids"]
        if chunk_id not in known
    }
    if stale_gold:
        raise ValueError(
            "gold_chunk_ids do not match this index: " + ", ".join(sorted(stale_gold))
        )

    for record in scoped_gold:
        gold_chunks = [known[chunk_id] for chunk_id in record["gold_chunk_ids"]]
        if any(
            not any(_evidence_matches_chunk(evidence, chunk) for chunk in gold_chunks)
            for evidence in record["gold_evidence"]
        ):
            raise ValueError(
                f"{record['question_id']}: gold_evidence does not match gold_chunk_ids"
            )
        unmatched_chunks = [
            chunk["chunk_id"]
            for chunk in gold_chunks
            if not any(
                _evidence_matches_chunk(evidence, chunk)
                for evidence in record["gold_evidence"]
            )
        ]
        if unmatched_chunks:
            raise ValueError(
                f"{record['question_id']}: gold_chunk_ids are not supported by "
                "gold_evidence: " + ", ".join(unmatched_chunks)
            )


def validate_index_mapping(gold_records, predictions, index_dir, split="all"):
    from pipeline import load_index_bundle

    _, chunks = load_index_bundle(index_dir)
    validate_gold_mapping(gold_records, chunks, split)
    known = {chunk["chunk_id"] for chunk in chunks}
    scoped_ids = {
        record["question_id"]
        for record in gold_records
        if split == "all" or record["split"] == split
    }
    stale_retrieval = {
        chunk_id
        for record in predictions
        if record["question_id"] in scoped_ids
        for chunk_id in record["retrieved_chunk_ids"]
        if chunk_id not in known
    }
    if stale_retrieval:
        raise ValueError(
            "retrieved_chunk_ids do not match this index: "
            + ", ".join(sorted(stale_retrieval))
        )
    return sha256_file(Path(index_dir) / "manifest.json")


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else 0.0


def _prf(tp, fp, fn):
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _classification(gold_rows, prediction_by_id, positive):
    tp = fp = fn = 0
    for gold in gold_rows:
        predicted_answerable = prediction_by_id[gold["question_id"]]["answered"]
        actual = (
            gold["answerable"] if positive == "answerable" else not gold["answerable"]
        )
        predicted = (
            predicted_answerable
            if positive == "answerable"
            else not predicted_answerable
        )
        tp += actual and predicted
        fp += not actual and predicted
        fn += actual and not predicted
    return _prf(tp, fp, fn)


def _metrics(gold_rows, prediction_by_id, k):
    recalls = {cutoff: [] for cutoff in sorted({k, 20, 50})}
    hits = []
    reciprocal_ranks = []
    ndcgs = []
    citation_tp = citation_fp = citation_fn = citation_cases = 0

    for gold in gold_rows:
        prediction = prediction_by_id[gold["question_id"]]
        relevant = set(gold["gold_chunk_ids"])
        retrieved = prediction["retrieved_chunk_ids"][:k]
        if "answer" in prediction:
            cited = set(prediction["cited_chunk_ids"])
            context = set(
                prediction.get("context_chunk_ids", prediction["retrieved_chunk_ids"])
            )
            supported_citations = cited & relevant & context
            citation_tp += len(supported_citations)
            citation_fp += len(cited - supported_citations)
            citation_fn += len(relevant - supported_citations)
            citation_cases += 1

        if relevant and prediction.get("ranked", True):
            for cutoff in recalls:
                recalls[cutoff].append(
                    len(relevant.intersection(prediction["retrieved_chunk_ids"][:cutoff]))
                    / len(relevant)
                )
            hit_ranks = [
                rank
                for rank, chunk_id in enumerate(retrieved, 1)
                if chunk_id in relevant
            ]
            hits.append(float(bool(hit_ranks)))
            reciprocal_ranks.append(1 / hit_ranks[0] if hit_ranks else 0.0)
            dcg = sum(1 / math.log2(rank + 1) for rank in hit_ranks)
            ideal_dcg = sum(
                1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1)
            )
            ndcgs.append(dcg / ideal_dcg)

    def average(values):
        return round(sum(values) / len(values), 6) if values else None

    def human_score(field, eligible):
        values = [
            float(prediction_by_id[gold["question_id"]][field])
            for gold in eligible
            if field in prediction_by_id[gold["question_id"]]
        ]
        return {
            "mean": average(values),
            "coverage": len(values),
            "eligible_cases": len(eligible),
            "coverage_rate": _ratio(len(values), len(eligible)),
        }

    generated_rows = [
        gold for gold in gold_rows if "answer" in prediction_by_id[gold["question_id"]]
    ]
    answered_rows = [
        gold
        for gold in generated_rows
        if prediction_by_id[gold["question_id"]]["answered"]
    ]

    retrieval = {
        f"recall@{cutoff}": average(values) for cutoff, values in recalls.items()
    }
    retrieval.update(
        {
            f"hit@{k}": average(hits),
            f"mrr@{k}": average(reciprocal_ranks),
            f"ndcg@{k}": average(ndcgs),
        }
    )

    return {
        "cases": len(gold_rows),
        "retrieval_eligible_cases": sum(bool(gold["gold_chunk_ids"]) for gold in gold_rows),
        "retrieval_cases": len(hits),
        "retrieval": retrieval,
        "citation_cases": citation_cases,
        "citation": (
            _prf(citation_tp, citation_fp, citation_fn)
            if citation_tp + citation_fp + citation_fn
            else {"precision": None, "recall": None, "f1": None}
        ),
        "answerable": _classification(gold_rows, prediction_by_id, "answerable"),
        "refusal": _classification(gold_rows, prediction_by_id, "refusal"),
        "generation": {
            "answer_correct": human_score("answer_correct", generated_rows),
            "faithful": human_score("faithful", answered_rows),
        },
    }


def evaluate(gold_records, predictions, k=10, split="all"):
    if k < 1:
        raise ValueError("k must be at least 1")
    if split not in {"dev", "test", "all"}:
        raise ValueError("split must be dev, test, or all")
    all_gold_ids = {record["question_id"] for record in gold_records}
    prediction_by_id = {record["question_id"]: record for record in predictions}
    unknown = prediction_by_id.keys() - all_gold_ids
    if unknown:
        raise ValueError(
            f"predictions contain unknown question_ids: {', '.join(sorted(unknown))}"
        )

    selected = [
        record for record in gold_records if split == "all" or record["split"] == split
    ]
    if not selected:
        raise ValueError("gold dataset has no items for the selected split")
    selected_ids = {record["question_id"] for record in selected}
    missing = selected_ids - prediction_by_id.keys()
    if missing:
        raise ValueError(f"missing predictions: {', '.join(sorted(missing))}")

    categories = sorted({record["category"] for record in selected})
    splits = sorted({record["split"] for record in selected})
    return {
        "metric_version": 2,
        "k": k,
        "split": split,
        "counts": {
            "gold_records": len(gold_records),
            "gold_cases": len(selected),
            "out_of_split_gold_ignored": len(gold_records) - len(selected),
            "predictions": len(predictions),
            "out_of_scope_predictions_ignored": len(
                prediction_by_id.keys() - selected_ids
            ),
        },
        "overall": _metrics(selected, prediction_by_id, k),
        "by_category": {
            category: _metrics(
                [record for record in selected if record["category"] == category],
                prediction_by_id,
                k,
            )
            for category in categories
        },
        "by_split": {
            split: _metrics(
                [record for record in selected if record["split"] == split],
                prediction_by_id,
                k,
            )
            for split in splits
        },
    }


def _refusal_at_threshold(rows, prediction_by_id, threshold):
    tp = fp = fn = 0
    for gold in rows:
        prediction = prediction_by_id[gold["question_id"]]
        answered = bool(prediction["retrieved_chunk_ids"]) and (
            prediction["top_score"] >= threshold
        )
        actual_refusal = not gold["answerable"]
        predicted_refusal = not answered
        tp += actual_refusal and predicted_refusal
        fp += not actual_refusal and predicted_refusal
        fn += actual_refusal and not predicted_refusal
    return _prf(tp, fp, fn)


def calibrate_threshold(gold_records, predictions):
    prediction_by_id = {record["question_id"]: record for record in predictions}
    dev = [record for record in gold_records if record["split"] == "dev"]
    if not dev or {record["answerable"] for record in dev} != {False, True}:
        raise ValueError("threshold calibration needs dev items from both classes")
    missing_scores = [
        record["question_id"]
        for record in dev
        if "top_score" not in prediction_by_id.get(record["question_id"], {})
    ]
    if missing_scores:
        raise ValueError(
            f"dev predictions missing top_score: {', '.join(sorted(missing_scores))}"
        )

    scores = [prediction_by_id[record["question_id"]]["top_score"] for record in dev]
    thresholds = sorted(set(scores))
    above_max = math.nextafter(max(scores), math.inf)
    if math.isfinite(above_max):
        thresholds.append(above_max)
    threshold = thresholds[0]
    dev_refusal = _refusal_at_threshold(dev, prediction_by_id, threshold)
    for candidate in thresholds[1:]:
        metrics = _refusal_at_threshold(dev, prediction_by_id, candidate)
        if metrics["f1"] > dev_refusal["f1"]:
            threshold, dev_refusal = candidate, metrics

    return {
        "rule": "answered = bool(retrieved_chunk_ids) and top_score >= threshold",
        "selected_on": "dev",
        "objective": "refusal.f1",
        "tie_break": "lowest threshold",
        "threshold": threshold,
        "dev": {"cases": len(dev), "refusal": dev_refusal},
        "warning": (
            "Only dev was inspected. Freeze this threshold, then run and report test "
            "separately; never feed test predictions into tuning decisions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", help="gold JSONL")
    parser.add_argument("predictions", help="predictions JSONL")
    parser.add_argument(
        "--k", type=int, default=10, help="retrieval cutoff (default: 10)"
    )
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="select a refusal threshold from dev top_score values",
    )
    args = parser.parse_args()
    try:
        gold = load_gold(args.gold)
        predictions = load_predictions(args.predictions)
        result = evaluate(gold, predictions, args.k, args.split)
        result["provenance"] = {
            "gold_sha256": sha256_file(args.gold),
            "predictions_sha256": sha256_file(args.predictions),
            "index_manifest_sha256": validate_index_mapping(
                gold, predictions, INDEX_DIR, args.split
            ),
        }
        if args.calibrate_threshold:
            result["calibration"] = calibrate_threshold(gold, predictions)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
