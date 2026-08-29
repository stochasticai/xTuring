import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from xturing.evaluation.adapters.base import BaseEvalAdapter
from xturing.evaluation.schemas import EvalRunResult


def persist_eval_result(result: EvalRunResult, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def run_eval_adapter(
    adapter: BaseEvalAdapter,
    *,
    model: Any,
    dataset: Any,
    task_name: str,
    output_path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvalRunResult:
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()

    result = adapter.run(
        model=model,
        dataset=dataset,
        task_name=task_name,
        metadata=metadata,
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    duration_seconds = time.monotonic() - started_monotonic

    if result.started_at is None:
        result.started_at = started_at
    if result.finished_at is None:
        result.finished_at = finished_at
    if result.duration_seconds is None:
        result.duration_seconds = duration_seconds

    if output_path is not None:
        persist_eval_result(result, output_path)

    return result
