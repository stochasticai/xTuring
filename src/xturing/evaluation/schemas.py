from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalMetric:
    name: str
    value: float
    higher_is_better: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
        }


@dataclass
class EvalRunResult:
    adapter_name: str
    task_name: str
    status: str
    metrics: List[EvalMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "task_name": self.task_name,
            "status": self.status,
            "metrics": [metric.as_dict() for metric in self.metrics],
            "metadata": self.metadata,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
        }
