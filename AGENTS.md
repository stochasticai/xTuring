# Repository Guidelines

## Project Structure & Module Organization
- Source code: `src/xturing/` (key packages: `models/`, `engines/`, `datasets/`, `preprocessors/`, `trainers/`, `cli/`, `utils/`, `config/`, `ui/`, `self_instruct/`, `model_apis/`, `registry.py`).
- Tests: `tests/xturing/` mirroring package layout (`test_*.py`).
- Examples & docs: `examples/`, `docs/`; assets and workflows: `.github/`.

## Build, Test, and Development Commands
- Setup (editable + dev tools):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .`
  - `pip install -r requirements-dev.txt`
- Formatting & lint (pre-commit):
  - `pre-commit install && pre-commit install --hook-type commit-msg`
  - `pre-commit run -a` (runs black, isort, autoflake, yaml checks, gitlint, absolufy-imports)
- Run tests (pytest):
  - `pytest -q` or target a subset: `pytest tests/xturing/models -k gpt2`
  - CPU-only example: `CUDA_VISIBLE_DEVICES=-1 pytest -q -k cpu`
- Local CLI/UI:
  - `xturing chat -m <path-to-model-dir>`
  - `python -c "from xturing.ui import Playground; Playground().launch()"`

## Coding Style & Naming Conventions
- Python, 4-space indent; keep functions small and typed where practical.
- Tools: black (PEP 8, 88 cols), isort (`--profile black`), autoflake (remove unused), absolufy-imports.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Prefer explicit imports from `xturing.*`; avoid unused/relative imports.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/xturing/<area>/test_*.py`.
- Keep tests fast and deterministic; avoid network and large model downloads.
- Use small fixtures (e.g., `TextDataset`, `InstructionDataset`) and CPU where possible.
- Add tests for new code paths and regressions; run `pytest -q` before pushing.

## Commit & Pull Request Guidelines
- Conventional, clear titles (<= 80 chars); avoid “WIP”. Provide context in body (wrap ~120 cols).
- Examples: `feat(models): add llama2 INT4 path`, `fix(datasets): handle missing target`.
- Run `pre-commit` and tests locally.
- Open PRs against `dev`; include description, linked issues, and screenshots/CLI snippets when UI/UX changes.

## Security & Configuration Tips
- Do not commit secrets. For API-backed features, export keys: `OPENAI_API_KEY`, `COHERE_API_KEY`, `AI21_API_KEY`.
- Tunable defaults live in `src/xturing/config/*.yaml` (generation/finetuning). Document changes impacting behavior.
