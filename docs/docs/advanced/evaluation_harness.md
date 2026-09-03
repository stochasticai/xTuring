---
title: 📏 Evaluation Harness Scaffold
description: Adapter-based evaluation runner scaffold
sidebar_position: 5
---

# 📏 Evaluation harness scaffold

xTuring now includes an evaluation scaffold in `xturing.evaluation` to standardize benchmark integration work.

## What is included

1. `BaseEvalAdapter`: adapter interface for external evaluation backends.
2. `LMEvalAdapter`: initial scaffold adapter for lm-evaluation-harness style workflows.
3. `run_eval_adapter(...)`: orchestration helper with timing metadata.
4. `persist_eval_result(...)`: JSON artifact writer for consistent output format.

## Quick usage

```python
from pathlib import Path

from xturing.evaluation import LMEvalAdapter, run_eval_adapter

adapter = LMEvalAdapter(tasks=["arc_easy"], num_fewshot=0, batch_size=1)

result = run_eval_adapter(
    adapter,
    model=object(),   # replace with your xTuring model instance
    dataset=None,     # replace with your dataset / task source
    task_name="arc_easy",
    output_path=Path("artifacts/evals/arc_easy.json"),
)

print(result.status)      # planned (scaffold)
print(result.metadata)    # adapter config + integration status
```

## Current scope

The current `LMEvalAdapter` is intentionally scaffold-only and returns a standardized planned result. This gives a stable contract for:

1. CI artifact wiring
2. result schema integration
3. future backend execution integration without changing downstream consumers
