from typing import Any, Dict, List, Optional

from xturing.evaluation.adapters.base import BaseEvalAdapter
from xturing.evaluation.schemas import EvalRunResult


class LMEvalAdapter(BaseEvalAdapter):
    """Scaffold adapter for lm-evaluation-harness style integrations."""

    adapter_name = "lm_eval"

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        num_fewshot: int = 0,
        batch_size: int = 1,
    ):
        self.tasks = tasks or []
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size

    def run(
        self,
        *,
        model: Any,
        dataset: Any,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalRunResult:
        result_metadata = {
            "tasks": self.tasks,
            "requested_task": task_name,
            "num_fewshot": self.num_fewshot,
            "batch_size": self.batch_size,
            "integration_status": "scaffold_only",
        }
        if metadata:
            result_metadata.update(metadata)

        return EvalRunResult(
            adapter_name=self.adapter_name,
            task_name=task_name,
            status="planned",
            metrics=[],
            metadata=result_metadata,
        )
