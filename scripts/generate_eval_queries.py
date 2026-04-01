"""Generate data/eval_queries.json from real ingested documents.

Loads data/datasets/clean_docs.json, samples a balanced set of documents
with meaningful titles, and writes an eval set where:
  - query_text = the document's title
  - relevant_doc_ids = [document id]  (becomes the chunk parent_id after chunking)

Usage:
    python scripts/generate_eval_queries.py [--max 50] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DOCS_PATH = REPO_ROOT / "data" / "datasets" / "clean_docs.json"
OUTPUT_PATH = REPO_ROOT / "data" / "eval_queries.json"

MIN_TITLE_LEN = 25
MIN_CONTENT_LEN = 200


def load_docs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_good_doc(doc: dict) -> bool:
    title = (doc.get("title") or "").strip()
    content = (doc.get("content") or "").strip()
    return len(title) >= MIN_TITLE_LEN and len(content) >= MIN_CONTENT_LEN


def sample_balanced(docs: list[dict], max_total: int, seed: int) -> list[dict]:
    """Sample up to max_total docs, balancing across sources."""
    by_source: dict[str, list[dict]] = {}
    for doc in docs:
        src = doc.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    sources = list(by_source.keys())
    per_source = max(1, max_total // len(sources))

    rng = random.Random(seed)
    selected: list[dict] = []
    for src in sources:
        pool = by_source[src]
        rng.shuffle(pool)
        selected.extend(pool[:per_source])

    # Fill remaining slots from any source if total < max_total
    remaining = max_total - len(selected)
    if remaining > 0:
        used_ids = {d["id"] for d in selected}
        extras = [d for d in docs if d["id"] not in used_ids]
        rng.shuffle(extras)
        selected.extend(extras[:remaining])

    return selected[:max_total]


def build_eval_queries(docs: list[dict]) -> list[dict]:
    queries = []
    for i, doc in enumerate(docs, start=1):
        queries.append(
            {
                "query_id": f"eq-{i:03d}",
                "query_text": doc["title"].strip(),
                "relevant_doc_ids": [doc["id"]],
            }
        )
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval_queries.json from real docs")
    parser.add_argument("--max", type=int, default=50, dest="max_queries",
                        help="Maximum number of eval queries to generate (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if not CLEAN_DOCS_PATH.exists():
        raise FileNotFoundError(
            f"{CLEAN_DOCS_PATH} not found — run 'dvc repro preprocess' first"
        )

    all_docs = load_docs(CLEAN_DOCS_PATH)
    good_docs = [d for d in all_docs if is_good_doc(d)]

    sources = {}
    for d in good_docs:
        sources[d.get("source", "unknown")] = sources.get(d.get("source", "unknown"), 0) + 1

    print(f"Loaded {len(all_docs)} docs, {len(good_docs)} pass quality filter")
    print("Source distribution:", sources)

    sampled = sample_balanced(good_docs, args.max_queries, args.seed)
    queries = build_eval_queries(sampled)

    OUTPUT_PATH.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(queries)} eval queries to {OUTPUT_PATH}")

    src_counts: dict[str, int] = {}
    for d in sampled:
        src = d.get("source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1
    print("Sampled source distribution:", src_counts)


if __name__ == "__main__":
    main()
