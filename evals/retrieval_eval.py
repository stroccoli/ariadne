from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    query_id: str
    query_text: str
    relevant_doc_ids: list[str]


@dataclass
class RetrievalResult:
    query_id: str
    retrieved_texts: list[str]
    retrieved_doc_ids: list[str]


@dataclass
class EvalReport:
    recall_at_k: dict[int, float]
    mrr: float
    n_queries: int
    per_query: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Eval Report ({self.n_queries} queries)",
            f"  MRR:       {self.mrr:.4f}",
        ]
        for k, recall in sorted(self.recall_at_k.items()):
            lines.append(f"  Recall@{k}: {recall:.4f}")
        return "\n".join(lines)


def build_eval_set_from_titles(
    doc_id_title_pairs: list[tuple[str, str]],
    *,
    max_queries: int = 100,
) -> list[EvalQuery]:
    eval_queries: list[EvalQuery] = []
    for doc_id, title in doc_id_title_pairs[:max_queries]:
        if not title.strip():
            continue
        eval_queries.append(
            EvalQuery(
                query_id=doc_id,
                query_text=title.strip(),
                relevant_doc_ids=[doc_id],
            )
        )
    return eval_queries


def load_curated_eval_set(path: Path) -> list[EvalQuery]:
    """Load a curated eval set from a JSON file.

    Each entry must have: query_id, query_text, relevant_doc_ids.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    eval_queries: list[EvalQuery] = []
    for entry in raw:
        query_id = entry["query_id"]
        query_text = entry["query_text"]
        relevant_ids = entry.get("relevant_doc_ids", [])
        eval_queries.append(
            EvalQuery(
                query_id=query_id,
                query_text=query_text,
                relevant_doc_ids=relevant_ids,
            )
        )
    logger.info("Loaded %d curated eval queries from %s", len(eval_queries), path)
    return eval_queries


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


class RetrievalEvaluator:

    def __init__(
        self,
        search_fn,
        id_extractor=None,
    ) -> None:
        self.search_fn = search_fn
        self.id_extractor = id_extractor or (lambda text: text)

    def evaluate(
        self,
        eval_queries: list[EvalQuery],
        *,
        k_values: list[int] | None = None,
        verbose: bool = True,
    ) -> EvalReport:
        if k_values is None:
            k_values = [1, 3, 5]

        max_k = max(k_values)
        recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        per_query_results: list[dict] = []

        for i, eq in enumerate(eval_queries):
            try:
                retrieved_texts = self.search_fn(eq.query_text)
            except Exception as exc:
                logger.warning("Search failed for query '%s': %s", eq.query_text[:50], exc)
                retrieved_texts = []

            retrieved_ids = [self.id_extractor(t) for t in retrieved_texts]

            rr = reciprocal_rank(retrieved_ids, eq.relevant_doc_ids)
            mrr_sum += rr

            per_query_result = {
                "query_id": eq.query_id,
                "query_text": eq.query_text[:80],
                "retrieved_ids": retrieved_ids[:max_k],
                "relevant_ids": eq.relevant_doc_ids,
                "reciprocal_rank": round(rr, 4),
            }
            for k in k_values:
                r_at_k = recall_at_k(retrieved_ids, eq.relevant_doc_ids, k)
                recall_sums[k] += r_at_k
                per_query_result[f"recall_at_{k}"] = round(r_at_k, 4)

            per_query_results.append(per_query_result)

            if verbose and (i + 1) % 10 == 0:
                logger.info("Evaluated %d/%d queries...", i + 1, len(eval_queries))

        n = len(eval_queries)
        report = EvalReport(
            recall_at_k={k: round(recall_sums[k] / n, 4) for k in k_values},
            mrr=round(mrr_sum / n, 4),
            n_queries=n,
            per_query=per_query_results,
        )

        if verbose:
            logger.info(report.summary())

        return report

    def save_report(self, report: EvalReport, path: Path) -> None:
        data = {
            "recall_at_k": {f"recall@{k}": v for k, v in report.recall_at_k.items()},
            "mrr": report.mrr,
            "n_queries": report.n_queries,
            "per_query": report.per_query,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Eval report saved to %s", path)


def compare_chunking_presets(
    eval_queries: list[EvalQuery],
    preset_results: dict[str, list[str]],
) -> dict[str, EvalReport]:
    reports: dict[str, EvalReport] = {}
    for preset_name, search_fn in preset_results.items():
        evaluator = RetrievalEvaluator(search_fn)
        report = evaluator.evaluate(eval_queries, verbose=False)
        reports[preset_name] = report
        logger.info("Preset '%s': MRR=%.4f, Recall@3=%.4f", preset_name, report.mrr, report.recall_at_k.get(3, 0))
    return reports
