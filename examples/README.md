# Navigating examples

This folder is organized by task first, so you can pick a workflow quickly and then
switch model keys as needed.

## Directory structure

```text
examples/
    | datasets/
    | features/
    | models/
    | notebooks/
    | playground_ui/
```

## Start here: task-based notebooks

Use the notebooks in `examples/notebooks/` as the primary entry point:

| Notebook | Purpose |
| --- | --- |
| `notebooks/finetune_lora.ipynb` | Fine-tune with LoRA |
| `notebooks/finetune_lora_int8.ipynb` | Fine-tune with LoRA + INT8 |
| `notebooks/evaluate_perplexity.ipynb` | Evaluate with perplexity |

## Model variants

Instead of maintaining one notebook per model, use one task notebook and choose a model key.

| Version | Template |
| --- | --- |
| Base | `<model_key>` |
| LoRA | `<model_key>_lora` |
| INT8 | `<model_key>_int8` |
| LoRA + INT8 | `<model_key>_lora_int8` |

For the full model-key list, see `https://xturing.stochastic.ai/overview/supported_models` in the docs.

For INT4 + LoRA:

```python
from xturing.models import GenericLoraKbitModel
model = GenericLoraKbitModel("<model_path>")
```

`<model_path>` can be a local directory or a Hugging Face model path.

## Legacy model folders

Legacy model-specific notebooks are archived in `examples/legacy/model_notebooks/`.
The `examples/models/` folders remain for scripts and dataset references.
New examples should prefer task-based notebooks and keep model differences in docs.
