#!/usr/bin/env python3
"""Prepare evidence-first candidate questions from index/chunks.json.

The output is annotation work, never gold data.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from eval import write_jsonl


QUESTION_PLACEHOLDER = "【待人工填写】请根据 gold_evidence 编写一个仅凭该证据可回答的问题。"
UNANSWERABLE_PLACEHOLDER = "【待人工填写】请编写一个主题相关、但冻结语料中确实无法回答的问题。"


def validate_chunks(chunks):
    if not isinstance(chunks, list):
        raise ValueError("chunks file must contain a JSON array")
    seen = set()
    for index, chunk in enumerate(chunks):
        location = f"chunk[{index}]"
        if not isinstance(chunk, dict):
            raise ValueError(f"{location} must be an object")
        missing = {"chunk_id", "source", "section", "text"} - chunk.keys()
        if missing:
            raise ValueError(f"{location} missing fields: {', '.join(sorted(missing))}")
        for field in ("chunk_id", "source", "text"):
            if not isinstance(chunk[field], str) or not chunk[field].strip():
                raise ValueError(f"{location}.{field} must be a non-empty string")
        section = chunk["section"]
        if section is not None and (
            not isinstance(section, str) or not section.strip()
        ):
            raise ValueError(f"{location}.section must be null or a non-empty string")
        if chunk["chunk_id"] in seen:
            raise ValueError(f"duplicate chunk_id: {chunk['chunk_id']}")
        seen.add(chunk["chunk_id"])
    if not chunks:
        raise ValueError("chunks file is empty")


def load_chunks(path):
    path = Path(path)
    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    validate_chunks(chunks)
    return chunks


def sample_chunks(chunks, count, seed):
    validate_chunks(chunks)
    if not 0 <= count <= len(chunks):
        raise ValueError(f"count must be between 0 and {len(chunks)}")

    groups = {}
    for chunk in chunks:
        groups.setdefault(chunk["source"], []).append(chunk)
    sources = sorted(groups)
    for group in groups.values():
        group.sort(key=lambda chunk: chunk["chunk_id"])

    randomizer = random.Random(seed)
    randomizer.shuffle(sources)
    for source in sources:
        randomizer.shuffle(groups[source])

    selected = []
    while len(selected) < count:
        for source in sources:
            if groups[source]:
                selected.append(groups[source].pop())
                if len(selected) == count:
                    return selected
    return selected


def stable_split(key, test_ratio=0.2):
    if not 0 <= test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1")
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return "test" if value / 2**64 < test_ratio else "dev"


def _fact_key(chunk):
    return json.dumps(
        [chunk["source"], chunk["section"], chunk["text"].strip()],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def candidate_from_chunk(chunk, test_ratio=0.2):
    fact_key = _fact_key(chunk)
    group_id = "group_" + hashlib.sha256(fact_key.encode("utf-8")).hexdigest()[:16]
    question_id = "candidate_" + hashlib.sha256(
        chunk["chunk_id"].encode("utf-8")
    ).hexdigest()[:16]
    return {
        "question_id": question_id,
        "group_id": group_id,
        "question": QUESTION_PLACEHOLDER,
        "category": "evidence_first",
        "split": stable_split(group_id, test_ratio),
        "answerable": True,
        "gold_answer": "",
        "gold_evidence": [
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "quote": chunk["text"].strip(),
            }
        ],
        "gold_chunk_ids": [chunk["chunk_id"]],
        "notes": "索引分层抽样候选；group_id/split 只是初值，人工合并同事实组后再复核。",
    }


def build_candidates(chunks, count, seed=2024, unanswerable=0, test_ratio=0.2):
    if unanswerable < 0:
        raise ValueError("unanswerable must be at least 0")
    candidates = [
        candidate_from_chunk(chunk, test_ratio)
        for chunk in sample_chunks(chunks, count, seed)
    ]
    for number in range(1, unanswerable + 1):
        key = f"unanswerable:{seed}:{number}"
        group_id = "group_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        candidates.append(
            {
                "question_id": "candidate_"
                    + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                "group_id": group_id,
                "question": f"{UNANSWERABLE_PLACEHOLDER}（候选 {number}）",
                "category": "unanswerable",
                "split": stable_split(group_id, test_ratio),
                "answerable": False,
                "gold_answer": "",
                "gold_evidence": [],
                "gold_chunk_ids": [],
                "notes": "库外问题占位；group_id/split 只是初值，人工须检索完整语料并复核分组。",
            }
        )
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="index/chunks.json")
    parser.add_argument("--output", default="eval/candidates.jsonl")
    parser.add_argument("--count", type=int, default=80, help="evidence candidates")
    parser.add_argument("--unanswerable", type=int, default=0, help="extra placeholders")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--force", action="store_true", help="overwrite output")
    args = parser.parse_args()
    try:
        chunks = load_chunks(args.chunks)
        candidates = build_candidates(
            chunks, args.count, args.seed, args.unanswerable, args.test_ratio
        )
        write_jsonl(candidates, args.output, args.force)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Wrote {len(candidates)} candidate rows to {args.output}; "
        "human review is still required."
    )


if __name__ == "__main__":
    main()
