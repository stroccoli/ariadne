from __future__ import annotations

import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]

from ariadne.core.agents.rag import retrieve_context
from ariadne.core.config import get_embedding_client, get_vector_store


class RagSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.getenv("RUN_RAG_SMOKE", "0") != "1":
            raise unittest.SkipTest("Set RUN_RAG_SMOKE=1 to run the Qdrant + Ollama smoke test")

        cls.compose_command = cls._resolve_compose_command()
        if cls.compose_command is None:
            raise unittest.SkipTest("Docker compose is not available")

        if shutil.which("ollama") is None:
            raise unittest.SkipTest("Ollama CLI is not available")

        cls._run_command([*cls.compose_command, "up", "-d", "qdrant"])
        cls._wait_for_qdrant()

        cls.collection_name = f"incident_knowledge_smoke_{os.getpid()}"

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "compose_command", None) is not None:
            cls._run_command([*cls.compose_command, "down"], check=False)

    @classmethod
    def _resolve_compose_command(cls) -> list[str] | None:
        docker_path = shutil.which("docker")
        if docker_path is not None:
            return [docker_path, "compose"]

        docker_compose_path = shutil.which("docker-compose")
        if docker_compose_path is not None:
            return [docker_compose_path]

        return None

    @classmethod
    def _run_command(cls, command: list[str], check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=check,
        )

    @classmethod
    def _wait_for_qdrant(cls, timeout_seconds: int = 30) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None

        while time.time() < deadline:
            try:
                with urlopen("http://localhost:6333/collections", timeout=2) as response:
                    if response.status == 200:
                        return
            except URLError as error:
                last_error = error
                time.sleep(1)

        raise RuntimeError(f"Qdrant did not become ready in time: {last_error}")

    def test_index_and_retrieve_against_qdrant(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "VECTOR_STORE": "qdrant",
                "EMBEDDING_PROVIDER": "ollama",
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_COLLECTION": self.collection_name,
                "QDRANT_SEARCH_LIMIT": "3",
                "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text:latest",
                "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            }
        )

        self._run_command(
            [sys.executable, "scripts/index_data.py", "--dataset", "data/incident_knowledge.json"],
            env=env,
        )

        os.environ.update(env)
        get_vector_store.cache_clear()
        get_embedding_client.cache_clear()

        context = retrieve_context(
            "ERROR postgres connection pool exhausted after repeated retries and active sessions are saturated"
        )

        self.assertIn("Database pool exhaustion pattern", context)
