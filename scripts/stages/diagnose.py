"""DVC stage: LLM-driven diagnosis of the pipeline health report."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ariadne.core.config import get_llm_client  # noqa: E402
from ariadne.core.retrieval.pipeline_diagnosis import diagnose_pipeline  # noqa: E402
from ariadne.core.retrieval.pipeline_report import PipelineHealthReport  # noqa: E402


logger = logging.getLogger(__name__)

_REPORT_PATH = _REPO_ROOT / "data" / "datasets" / "pipeline_report.json"
_DIAGNOSIS_PATH = _REPO_ROOT / "data" / "datasets" / "pipeline_diagnosis.json"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not _REPORT_PATH.exists():
        print(
            f"ERROR: {_REPORT_PATH} not found. Run the evaluate stage first.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = PipelineHealthReport.load(_REPORT_PATH)
    llm = get_llm_client()

    diagnosis = diagnose_pipeline(report, llm)
    diagnosis.save(_DIAGNOSIS_PATH)

    print(json.dumps(diagnosis.to_dict(), indent=2))
    logger.info("Diagnosis saved to %s", _DIAGNOSIS_PATH)


if __name__ == "__main__":
    main()
