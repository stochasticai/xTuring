from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from xturing.evaluation.schemas import EvalRunResult


class BaseEvalAdapter(ABC):
    adapter_name = "base"

    @abstractmethod
    def run(
        self,
        *,
        model: Any,
        dataset: Any,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalRunResult:
        """Run an evaluation adapter and return a standardized result object."""
