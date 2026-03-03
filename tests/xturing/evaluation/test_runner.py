import json

from xturing.evaluation import (
    EvalMetric,
    EvalRunResult,
    LMEvalAdapter,
    run_eval_adapter,
)
from xturing.evaluation.adapters.base import BaseEvalAdapter


class _DummyAdapter(BaseEvalAdapter):
    adapter_name = "dummy"

    def run(self, *, model, dataset, task_name, metadata=None):
        return EvalRunResult(
            adapter_name=self.adapter_name,
            task_name=task_name,
            status="completed",
            metrics=[EvalMetric(name="accuracy", value=0.75, higher_is_better=True)],
            metadata=metadata or {},
        )


def test_run_eval_adapter_persists_output(tmp_path):
    output_file = tmp_path / "eval" / "result.json"

    result = run_eval_adapter(
        _DummyAdapter(),
        model=object(),
        dataset=[{"text": "hello"}],
        task_name="smoke",
        output_path=output_file,
        metadata={"suite": "unit"},
    )

    assert result.status == "completed"
    assert result.metrics[0].name == "accuracy"
    assert result.duration_seconds is not None
    assert output_file.exists()

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["adapter_name"] == "dummy"
    assert payload["task_name"] == "smoke"
    assert payload["metadata"]["suite"] == "unit"


def test_lm_eval_adapter_scaffold_metadata():
    adapter = LMEvalAdapter(tasks=["arc_easy"], num_fewshot=2, batch_size=4)
    result = run_eval_adapter(
        adapter,
        model=object(),
        dataset=None,
        task_name="arc_easy",
    )

    assert result.status == "planned"
    assert result.adapter_name == "lm_eval"
    assert result.metadata["integration_status"] == "scaffold_only"
    assert result.metadata["num_fewshot"] == 2
