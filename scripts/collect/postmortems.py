from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from ariadne.core.retrieval.document import IngestionDocument

logger = logging.getLogger(__name__)

POSTMORTEMS_README_URL = (
    "https://raw.githubusercontent.com/danluu/post-mortems/master/README.md"
)

_LINE_PATTERN = re.compile(
    r"^\s*-\s+"
    r"(?:\[([^\]]+)\]\s+)?"
    r"\[([^\]]+)\]"
    r"\(([^)]+)\)",
    re.MULTILINE,
)

MAX_CONTENT_CHARS = 15_000
FETCH_TIMEOUT = 10
FETCH_MAX_WORKERS = 5


class PostmortemsCollector:

    def __init__(self, github_token: str | None = None) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ariadne-rag-pipeline/1.0"
        if github_token:
            self.session.headers["Authorization"] = f"Bearer {github_token}"

    def collect(
        self,
        max_entries: int = 1000,
        output_path: Path | None = None,
    ) -> list[IngestionDocument]:
        logger.info("Fetching danluu/post-mortems README...")
        readme_text = self._fetch_readme()

        logger.info("Parsing post-mortem entries...")
        entries = self._parse_readme(readme_text)
        logger.info("Found %d entries in README", len(entries))

        entries = entries[:max_entries]
        docs = self._fetch_all_content(entries)

        logger.info("Created %d IngestionDocuments from post-mortems", len(docs))

        if output_path:
            self._save(docs, output_path)

        return docs

    def _fetch_all_content(
        self, entries: list[tuple[str, str, str]]
    ) -> list[IngestionDocument]:
        """Fetch real content for each entry using a thread pool, with fallback to title."""
        docs: list[IngestionDocument] = [None] * len(entries)  # type: ignore[list-item]
        fetched = 0
        failed = 0

        def _fetch_one(idx: int, company: str, title: str, url: str) -> tuple[int, IngestionDocument]:
            content = self._try_fetch_url_content(url)
            if content:
                return idx, self._build_document(idx, company, title, url, content)
            return idx, self._build_document(idx, company, title, url, content=None)

        with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_fetch_one, i, company, title, url): i
                for i, (company, title, url) in enumerate(entries)
            }
            for future in as_completed(futures):
                idx, doc = future.result()
                docs[idx] = doc
                if doc.content != doc.title and len(doc.content) > 200:
                    fetched += 1
                else:
                    failed += 1

        logger.info(
            "Content fetch complete: %d fetched, %d fell back to title",
            fetched,
            failed,
        )
        return docs

    def _try_fetch_url_content(self, url: str) -> str | None:
        """Try to fetch and extract text content from a URL. Returns None on failure."""
        try:
            resp = self.session.get(url, timeout=FETCH_TIMEOUT, allow_redirects=True)
            if resp.status_code != 200:
                return None

            content_type = resp.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "lxml")

            # Remove noisy elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            # Prefer <article> or <main>, fallback to <body>
            container = soup.find("article") or soup.find("main") or soup.find("body")
            if container is None:
                return None

            text = container.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)

            if len(text) < 100:
                return None

            return text[:MAX_CONTENT_CHARS]
        except Exception:
            return None

    def _build_document(
        self, index: int, company: str, title: str, url: str, content: str | None
    ) -> IngestionDocument:
        if content:
            full_content = f"{company + ': ' if company else ''}{title}\n\n{content}" if company else f"{title}\n\n{content}"
        elif company:
            full_content = f"{company} incident: {title}"
        else:
            full_content = title
        service = company.lower().replace(" ", "_") if company else ""
        tags = _extract_tags_from_title(title)

        return IngestionDocument(
            id=f"pm-{index:05d}",
            title=title,
            content=full_content,
            source="postmortem",
            source_url=url,
            tags=tags,
            severity="unknown",
            service=service,
            created_at=None,
        )

    def _fetch_readme(self) -> str:
        response = self.session.get(POSTMORTEMS_README_URL, timeout=30)
        response.raise_for_status()
        return response.text

    def _parse_readme(self, text: str) -> list[tuple[str, str, str]]:
        entries: list[tuple[str, str, str]] = []
        for match in _LINE_PATTERN.finditer(text):
            company = (match.group(1) or "").strip()
            title = (match.group(2) or "").strip()
            url = (match.group(3) or "").strip()

            if not title or not url:
                continue
            if not url.startswith("http"):
                continue

            entries.append((company, title, url))

        return entries

    def _save(self, docs: list[IngestionDocument], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [doc.model_dump(mode="json") for doc in docs]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Saved %d post-mortems to %s", len(docs), path)


_INCIDENT_KEYWORDS = {
    "outage": ["outage"],
    "downtime": ["downtime"],
    "degradation": ["degradation"],
    "latency": ["latency", "performance"],
    "timeout": ["timeout"],
    "memory": ["memory", "oom"],
    "database": ["database"],
    "network": ["network"],
    "dns": ["dns"],
    "ssl": ["ssl", "tls"],
    "deploy": ["deployment"],
    "crash": ["crash"],
    "overflow": ["overflow"],
    "leak": ["memory-leak"],
    "corruption": ["data-corruption"],
    "ddos": ["ddos", "security"],
    "hack": ["security", "breach"],
}


def _extract_tags_from_title(title: str) -> list[str]:
    lower = title.lower()
    tags: list[str] = []
    for keyword, tag_list in _INCIDENT_KEYWORDS.items():
        if keyword in lower:
            tags.extend(tag_list)
    return list(dict.fromkeys(tags))


def load_collected_documents(path: Path) -> list[IngestionDocument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [IngestionDocument.model_validate(item) for item in raw]
