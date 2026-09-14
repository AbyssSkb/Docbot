import argparse
import json
import sys
import time

from pipeline import (
    RETRIEVAL_CONFIG_ROUTES,
    RETRIEVAL_SCORE_SEMANTICS,
    load_index_bundle,
    load_retrieval_config,
    retrieve,
    retrieval_score,
)


def search_index(
    query, index_dir="index", top_k=8, candidate_k=20, retrieval_config="hybrid"
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if type(candidate_k) is not int or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer")
    if retrieval_config not in RETRIEVAL_CONFIG_ROUTES:
        raise ValueError(f"unknown retrieval config: {retrieval_config}")

    query = query.strip()
    manifest, chunks = load_index_bundle(index_dir)
    retrievers, ranker = load_retrieval_config(
        manifest, chunks, index_dir, retrieval_config
    )
    started = time.perf_counter()
    results = retrieve(
        query,
        chunks,
        retrievers,
        ranker=ranker,
        candidate_k=candidate_k,
        result_k=top_k,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "query": query,
        "retrieval_config": retrieval_config,
        "score_semantics": RETRIEVAL_SCORE_SEMANTICS[retrieval_config],
        "results": [
            {
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "section": item["section"],
                "text": item["text"],
                "score": retrieval_score(item, retrieval_config),
                "routes": item.get("routes", []),
            }
            for item in results
        ],
        "meta": {
            "documents": manifest["counts"]["documents"],
            "chunks": manifest["counts"]["chunks"],
            "retrieval_ms": elapsed_ms,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search the Docbot knowledge base")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="return JSON search results")
    search_parser.add_argument("query")
    search_parser.add_argument("--index-dir", default="index")
    search_parser.add_argument("--top-k", type=int, default=8)
    search_parser.add_argument("--candidate-k", type=int, default=20)
    search_parser.add_argument(
        "--retrieval-config",
        choices=tuple(RETRIEVAL_CONFIG_ROUTES),
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    args = parser.parse_args(argv)

    try:
        payload = search_index(
            args.query,
            index_dir=args.index_dir,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            retrieval_config=args.retrieval_config,
        )
    except Exception as error:
        print(
            json.dumps(
                {"error": str(error), "type": type(error).__name__},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
