# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
xTuring is a Python library for fine-tuning, evaluation and data generation for Large Language Models (LLMs). It provides fast, efficient fine-tuning of open-source LLMs like Mistral, LLaMA, GPT-J with memory-efficient methods including LoRA and quantization (INT8/INT4).

## Development Commands

### Environment Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt
```

### Code Quality & Pre-commit
```bash
pre-commit install && pre-commit install --hook-type commit-msg
pre-commit run -a  # runs black, isort, autoflake, yaml checks, gitlint, absolufy-imports
```

### Testing
```bash
pytest -q                                    # run all tests
pytest tests/xturing/models -k gpt2         # run specific tests
CUDA_VISIBLE_DEVICES=-1 pytest -q -k cpu   # CPU-only tests
```

### CLI Usage
```bash
xturing chat -m <path-to-model-dir>         # chat interface
python -c "from xturing.ui import Playground; Playground().launch()"  # UI playground
```

## Architecture Overview

### Core Components
- **Models** (`src/xturing/models/`): Registry-based system supporting 15+ LLM architectures (LLaMA, GPT-2, Falcon, etc.) with variants for LoRA, INT8, and INT4 quantization
- **Engines** (`src/xturing/engines/`): Inference engines handling model loading, generation, and quantization optimizations
- **Datasets** (`src/xturing/datasets/`): Dataset abstractions for text, instruction, and text-to-image data
- **Trainers** (`src/xturing/trainers/`): PyTorch Lightning-based training pipeline with DeepSpeed integration
- **CLI** (`src/xturing/cli/`): Command-line interface with chat, UI, and API commands

### Registry Pattern
The codebase uses a registry pattern (`src/xturing/registry.py`) where models, datasets, and engines register themselves by name:
```python
# Models register like: BaseModel.add_to_registry("llama_lora", LlamaLora)
model = BaseModel.create("llama_lora")  # Factory method access
```

### Model Variants
Models follow a naming convention:
- Base: `llama`, `gpt2`, `falcon`
- LoRA: `llama_lora`, `gpt2_lora`
- INT8: `llama_int8`, `gpt2_int8`
- Combined: `llama_lora_int8`
- INT4: Use `GenericLoraKbitModel('<model_path>')` class

### Key Directories
- `config/`: YAML configuration files for model defaults
- `preprocessors/`: Data preprocessing utilities
- `self_instruct/`: Self-instruction data generation
- `model_apis/`: Integration with OpenAI, Cohere, AI21 APIs
- `ui/`: Gradio-based UI components
- `utils/`: Shared utilities and external logger configuration

## Development Guidelines

### Code Style
- Python with 4-space indentation
- Tools: black (88 cols), isort (--profile black), autoflake, absolufy-imports
- Naming: `snake_case` for functions/modules, `PascalCase` for classes

### Testing
- Framework: pytest with markers for `slow` and `gpu` tests
- Keep tests fast and deterministic, avoid large model downloads
- Use small fixtures and CPU where possible

### Environment Variables
For API-backed features, export keys:
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `AI21_API_KEY`

### Pull Requests
- Target `dev` branch
- Run pre-commit and tests locally before submitting
- Follow conventional commit format: `feat(models): add llama2 INT4 path`
