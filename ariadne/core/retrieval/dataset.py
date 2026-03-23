from __future__ import annotations

import json
from pathlib import Path


def load_documents(dataset_path: Path) -> list[str]:
    raw_items = json.loads(dataset_path.read_text(encoding="utf-8"))
    documents: list[str] = []

    for item in raw_items:
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        documents.append(f"{title}: {content}" if title else content)

    return documents
