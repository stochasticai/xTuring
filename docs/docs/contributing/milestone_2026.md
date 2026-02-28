---
title: 🗺️ 2026 Modernization Milestone
description: Gap analysis and execution plan toward xTuring 2.0
sidebar_position: 6
---

# xTuring 2.0 milestone (as of February 28, 2026)

## 1) Documented findings: key gaps vs modern frameworks

1. Serving/runtime gap:
Current server path is a thin FastAPI wrapper without high-throughput runtime features (continuous batching, advanced KV-cache strategies, OpenAI-compatible interfaces, structured outputs).

2. Post-training/alignment gap:
Current focus is SFT + DPO-centric; modern stacks expose broader alignment workflows and clearer trainer abstractions.

3. Evaluation gap:
Perplexity-focused workflows are present, but benchmark orchestration, regression gates, and model-vs-model eval workflows are limited.

4. Multimodal gap:
The project has breadth in model support but lacks a unified multimodal train/serve/eval pathway.

5. Operability gap:
Production observability, API compatibility standards, and deployment guardrails are limited.

6. Quality/process gap:
CI and test coverage are not yet strong enough to reliably catch API/docs drift and runtime regressions.

## 2) Milestone definition

Milestone name: `xTuring 2.0: 2026 Train + Serve + Eval Baseline`

Target outcomes:

1. Serve:
Provide OpenAI-compatible inference endpoints and a stable server contract.

2. Train:
Strengthen post-training API surface and trainer extensibility.

3. Eval:
Add repeatable evaluation pipelines with CI regression checks.

4. Operate:
Add observability and reliability gates for docs/tests/API behavior.

## 3) Workstreams and task plans

### WS-A: Serving/API modernization

Task A1: Compatibility API baseline
Plan:
1. Add `/v1/models` and `/v1/chat/completions` endpoints.
2. Normalize prompt handling and generation parameter mapping.
3. Keep legacy `/api` route for backward compatibility.
4. Add endpoint contract tests with mock models.

Task A2: Generation contract hardening
Plan:
1. Define request/response schemas and validation behavior.
2. Add error handling for unloaded model and invalid message payloads.
3. Document compatibility limits (single completion first; streaming deferred).

Task A3: Structured output/tool-calling prep
Plan:
1. Add schema placeholders in API models.
2. Keep implementation behind feature flags until runtime support is ready.

### WS-B: Training/alignment modernization

Task B1: Trainer interface expansion
Plan:
1. Define standardized trainer capability matrix (SFT, DPO, next alignment methods).
2. Introduce config-level switches without breaking current APIs.
3. Add trainer registry docs and examples.

Task B2: Alignment data pipeline hardening
Plan:
1. Validate preference dataset contracts and error messages.
2. Add smoke tests for trainer initialization paths.
3. Add migration notes for existing fine-tuning scripts.

### WS-C: Evaluation modernization

Task C1: Eval harness integration layer
Plan:
1. Create adapter interfaces for benchmark runners.
2. Add result schema for run metadata and aggregate metrics.
3. Persist evaluation outputs in a consistent artifact format.

Task C2: Regression gates
Plan:
1. Define baseline eval set for CI smoke checks.
2. Add pass/fail thresholds and report summaries.
3. Wire results to pull request checks.

### WS-D: Reliability/CI/docs

Task D1: CI expansion
Plan:
1. Add lightweight unit/API tests in CI.
2. Add docs build check.
3. Keep runtime-heavy tests opt-in via explicit flags.

Task D2: Docs contract sync
Plan:
1. Ensure docs only advertise existing commands/routes.
2. Add command verification checks where possible.
3. Add a release checklist for docs/API parity.

## 4) Parallelization plan

Parallel lane 1 (can run now): `WS-A A1/A2` and `WS-D D1/D2`

Parallel lane 2 (after lane 1 contracts stabilize): `WS-C C1`

Parallel lane 3 (partially independent): `WS-B B1/B2`

Dependencies:
1. A1 is prerequisite for C1 API-eval adapter stability.
2. D1 should start immediately to protect all lanes.
3. B1 can run in parallel with A1 as long as shared config contracts are versioned.

## 5) Current execution status

Completed in current tranche:
1. WS-A A1: Added OpenAI-compatible `/v1/models`, `/v1/chat/completions`, and `/v1/completions`.
2. WS-A A2: Added stream skeleton support (`stream=true`) and basic usage accounting payloads.
3. WS-D D1/D2: Added docs-contract checks to CI and synchronized API docs.
4. WS-C C1: Added `xturing.evaluation` scaffold (adapter interface, lm-eval adapter scaffold, result schema, artifact persistence).

Next:
1. Implement real backend execution for the `LMEvalAdapter`.
2. Expand OpenAI compatibility to richer tool/response schema handling.
3. Add CI threshold gates on persisted evaluation artifacts.
