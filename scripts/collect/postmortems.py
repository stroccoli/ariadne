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

# Each entry in the README is a paragraph of the form:
#   [Company](https://...). Description text.
# Entries in the same section share category tags derived from the ## heading.
_ENTRY_PATTERN = re.compile(
    r"^\[([^\]]+)\]\((https?://[^)]+)\)\.\s+(.{10,})",
    re.MULTILINE | re.DOTALL,
)

_SECTION_TAGS: dict[str, list[str]] = {
    "config errors": ["config"],
    "hardware": ["hardware", "infrastructure"],
    "database": ["database"],
    "time": ["time", "clock"],
    "conflicts": ["conflict"],
    "software": ["software"],
    "cascading": ["cascade"],
    "network": ["network"],
    "security": ["security"],
    "memory": ["memory"],
}

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
        self, entries: list[tuple[str, str, str, str, list[str]]]
    ) -> list[IngestionDocument]:
        """Fetch real content for each entry using a thread pool, with fallback to description."""
        docs: list[IngestionDocument] = [None] * len(entries)  # type: ignore[list-item]
        fetched = 0
        failed = 0

        def _fetch_one(
            idx: int,
            company: str,
            title: str,
            url: str,
            description: str,
            section_tags: list[str],
        ) -> tuple[int, IngestionDocument]:
            content = self._try_fetch_url_content(url)
            return idx, self._build_document(idx, company, title, url, description, content, section_tags)

        with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_fetch_one, i, *entry): i
                for i, entry in enumerate(entries)
            }
            for future in as_completed(futures):
                idx, doc = future.result()
                docs[idx] = doc
                if doc.content != doc.title and len(doc.content) > 200:
                    fetched += 1
                else:
                    failed += 1

        logger.info(
            "Content fetch complete: %d fetched, %d fell back to description",
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
        self,
        index: int,
        company: str,
        title: str,
        url: str,
        description: str,
        content: str | None,
        section_tags: list[str] | None = None,
    ) -> IngestionDocument:
        if content:
            prefix = f"{company}: " if company else ""
            full_content = f"{prefix}{title}\n\n{content}"
        elif description:
            prefix = f"{company}: " if company else ""
            full_content = f"{prefix}{description}"
        elif company:
            full_content = f"{company}: {title}"
        else:
            full_content = title

        tags = list(dict.fromkeys((section_tags or []) + _extract_tags_from_title(title)))

        return IngestionDocument(
            id=f"pm-{index:05d}",
            title=title,
            content=full_content,
            source="postmortem",
            source_url=url,
            tags=tags,
            severity="unknown",
            service=company.lower().replace(" ", "_") if company else "",
            created_at=None,
        )

    def _fetch_readme(self) -> str:
        response = self.session.get(POSTMORTEMS_README_URL, timeout=30)
        response.raise_for_status()
        return response.text

    def _parse_readme(self, text: str) -> list[tuple[str, str, str, str, list[str]]]:
        """Parse the README and return (company, title, url, description, section_tags) tuples.

        The README uses paragraph-based entries of the form:
            [Company](https://...). Description text.
        Entries within a ## section inherit category tags derived from the heading.
        """
        entries: list[tuple[str, str, str, str, list[str]]] = []
        current_section_tags: list[str] = []

        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if not para:
                continue

            # Detect section heading (## or ###)
            if re.match(r"^#{1,3}\s", para):
                heading = re.sub(r"^#{1,3}\s+", "", para).strip().lower()
                current_section_tags = next(
                    (tags[:] for key, tags in _SECTION_TAGS.items() if key in heading),
                    [],
                )
                continue

            # Detect incident entry: [Company](url). Description
            m = re.match(r"^\[([^\]]+)\]\((https?://[^)]+)\)\.\s+(.{10,})", para, re.DOTALL)
            if not m:
                continue

            company = m.group(1).strip()
            url = m.group(2).strip()
            description = re.sub(r"\s+", " ", m.group(3).strip())

            # Title = first sentence (up to first ". " boundary, max 200 chars)
            dot = description.find(". ")
            title = description[:dot].strip() if 0 < dot < 200 else description[:200].strip()
            if not title:
                continue

            entries.append((company, title, url, description, current_section_tags[:]))

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
