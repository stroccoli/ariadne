from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from ariadne.core.retrieval.document import IngestionDocument, SourceType

logger = logging.getLogger(__name__)

DEFAULT_REPOS = [
    "kubernetes/kubernetes",
    "prometheus/prometheus",
    "grafana/grafana",
    "istio/istio",
    "docker/compose",
    "hashicorp/vault",
    "elastic/elasticsearch",
    "redis/redis",
]

BUG_LABELS_BY_REPO: dict[str, list[str]] = {
    "kubernetes/kubernetes": ["kind/bug"],
    "prometheus/prometheus": ["kind/bug"],
    "grafana/grafana": ["type/bug", "bug"],
    "istio/istio": ["kind/bug"],
    "docker/compose": ["kind/bug"],
    "hashicorp/vault": ["bug"],
    "elastic/elasticsearch": ["bug", ">bug"],
    "redis/redis": ["bug"],
}

GITHUB_API_BASE = "https://api.github.com"


class GitHubIssuesCollector:

    def __init__(
        self,
        token: str,
        repos: list[str] | None = None,
        *,
        min_body_length: int = 100,
    ) -> None:
        self.repos = repos or DEFAULT_REPOS
        self.min_body_length = min_body_length
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def collect(
        self,
        max_per_repo: int = 200,
        output_path: Path | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[IngestionDocument]:
        all_docs: list[IngestionDocument] = []
        already_done_repos: set[str] = set()

        # Resume from partial checkpoint if available
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                partial = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                all_docs = [IngestionDocument.model_validate(d) for d in partial["docs"]]
                already_done_repos = set(partial.get("completed_repos", []))
                logger.info(
                    "Resuming from checkpoint: %d docs already collected, %d repos done",
                    len(all_docs),
                    len(already_done_repos),
                )
            except Exception as exc:
                logger.warning("Could not load checkpoint %s: %s — starting fresh", checkpoint_path, exc)

        for repo in self.repos:
            if repo in already_done_repos:
                logger.info("Skipping %s (already in checkpoint)", repo)
                continue

            logger.info("Collecting issues from %s (max=%d)", repo, max_per_repo)
            try:
                docs = self._collect_repo(repo, max_per_repo)
                logger.info("  → %d issues collected from %s", len(docs), repo)
                if len(docs) < max_per_repo * 0.8:
                    logger.warning(
                        "Shortfall from %s: got %d/%d requested (rate limit or low issue count)",
                        repo,
                        len(docs),
                        max_per_repo,
                    )
                all_docs.extend(docs)
                already_done_repos.add(repo)

                # Save partial checkpoint after each successful repo
                if checkpoint_path is not None:
                    self._save_partial_checkpoint(
                        all_docs, already_done_repos, checkpoint_path
                    )
            except requests.HTTPError as exc:
                logger.warning("Failed to collect %s: %s", repo, exc)

        logger.info("Total collected: %d documents from %d repos", len(all_docs), len(self.repos))

        if output_path:
            self._save(all_docs, output_path)

        return all_docs

    def _collect_repo(self, repo: str, max_issues: int) -> list[IngestionDocument]:
        docs: list[IngestionDocument] = []
        owner, repo_name = repo.split("/", 1)
        service = repo_name.lower()
        bug_labels = BUG_LABELS_BY_REPO.get(repo, ["bug"])
        page = 1
        while len(docs) < max_issues:
            remaining = max_issues - len(docs)
            per_page = min(100, remaining)

            issues = self._fetch_page(owner, repo_name, bug_labels, page, per_page)
            if not issues:
                break

            for issue in issues:
                doc = self._issue_to_document(issue, repo, service)
                if doc:
                    docs.append(doc)

            page += 1
            self._respect_rate_limit()

        return docs

    def _fetch_page(
        self,
        owner: str,
        repo: str,
        labels: list[str],
        page: int,
        per_page: int,
    ) -> list[dict]:
        labels_param = ",".join(labels)
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
        params = {
            "state": "closed",
            "labels": labels_param,
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "direction": "desc",
        }

        response = self.session.get(url, params=params, timeout=30)

        remaining = int(response.headers.get("X-RateLimit-Remaining", 9999))
        if remaining < 20:
            reset_at = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait_seconds = max(0, reset_at - int(time.time())) + 5
            logger.warning("GitHub rate limit low (%d remaining). Waiting %ds", remaining, wait_seconds)
            time.sleep(wait_seconds)

        response.raise_for_status()
        return response.json()

    def _issue_to_document(self, issue: dict, repo: str, service: str) -> IngestionDocument | None:
        body = str(issue.get("body") or "").strip()
        if len(body) < self.min_body_length:
            return None

        title = str(issue.get("title") or "").strip()
        issue_number = issue.get("number", 0)
        html_url = str(issue.get("html_url") or "")
        labels = issue.get("labels") or []
        tags = [lbl["name"].lower() for lbl in labels if isinstance(lbl, dict) and lbl.get("name")]
        severity = _infer_severity(tags)

        created_at: datetime | None = None
        raw_created = issue.get("created_at")
        if raw_created:
            try:
                created_at = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
            except ValueError:
                pass

        doc_id = f"gh-{service}-{issue_number}"

        return IngestionDocument(
            id=doc_id,
            title=title,
            content=body,
            source="github_issues",
            source_url=html_url,
            tags=tags,
            severity=severity,
            service=service,
            created_at=created_at,
        )

    def _save_partial_checkpoint(
        self,
        docs: list[IngestionDocument],
        completed_repos: set[str],
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "completed_repos": sorted(completed_repos),
            "docs": [doc.model_dump(mode="json") for doc in docs],
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.debug("Checkpoint saved: %s (%d docs, %d repos done)", path, len(docs), len(completed_repos))

    def _respect_rate_limit(self) -> None:
        time.sleep(0.5)

    def _save(self, docs: list[IngestionDocument], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [doc.model_dump(mode="json") for doc in docs]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Saved %d documents to %s", len(docs), path)


def _infer_severity(tags: list[str]) -> str:
    tags_str = " ".join(tags)
    if any(word in tags_str for word in ("critical", "blocker", "p0", "urgent")):
        return "critical"
    if any(word in tags_str for word in ("high", "important", "p1", "major")):
        return "high"
    if any(word in tags_str for word in ("medium", "normal", "p2")):
        return "medium"
    if any(word in tags_str for word in ("low", "minor", "p3", "trivial")):
        return "low"
    return "unknown"


def load_collected_documents(path: Path) -> list[IngestionDocument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [IngestionDocument.model_validate(item) for item in raw]
