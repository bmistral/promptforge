# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -e .
pip install -e ".[notebook]"  # plotly, pandas, nest_asyncio (for notebooks)
pip install -e ".[metrics]"   # rouge-score, sentence-transformers
pip install -e ".[dev]"       # pytest, ruff, mypy, pre-commit + metrics deps

# Configure env (copy then edit)
cp .env.example .env      # renseigner ANTHROPIC_API_KEY

# Run on a built-in dataset
python main.py --dataset summarization
python main.py --dataset sentiment --iterations 3 --variants 4 --execute

# Run with a custom prompt
python main.py --prompt "Analyse ce texte." --task "Analyse de sentiment"

# Resume an interrupted run
python main.py --resume results/optimization_runs/checkpoint_<id>.json

# Tests
pytest
pytest tests/unit/test_evaluator.py -v
pytest tests/unit/test_metrics.py -v

# Lint / format
ruff check .
ruff format .
```

CLI flags: `--dataset` (summarization|sentiment|extraction|code|address|language|ticket|regex|feedback|security), `--prompt`, `--task`, `--iterations` (default 4), `--variants` (default 5), `--top-k` (default 2), `--model` (default `claude-haiku-4-5-20251001`), `--execute` (real execution mode), `--resume <checkpoint.json>`.

Results saved as JSON under `results/optimization_runs/`. Checkpoints are deleted on successful completion.

## Architecture

PromptForge runs an evolutionary optimization loop over prompts. The three `core/` components each wrap async Claude API calls with tenacity retry and shared cost tracking.

- **`core/generator.py` — `PromptGenerator`**: Given a parent prompt, generates N variant prompts (few-shot, CoT, role, format constraints, etc.). Returns `list[str]`.
- **`core/evaluator.py` — `PromptEvaluator`**: Hybrid scorer. LLM-as-judge scores a prompt 0–10 on clarity (0–3), output quality (0–4), and robustness (0–3). When `execute=True` and `dataset_name` is provided, also computes deterministic metrics and combines them: `score = alpha * llm_score + (1 - alpha) * det_score * 10`. `alpha` defaults to 0.5 and is set at construction time. Falls back to pure LLM score when no deterministic metric is available for the dataset. Returns `(float, str)`.
- **`core/optimizer.py` — `PromptOptimizer`**: Takes the top-K prompts and their judge feedback, produces mutated/crossed prompts for the next generation pool. Returns `list[str]`.

**`core/loop.py` — `PromptForge`** orchestrates the full cycle. Two public entry points:
- `run(...)` — synchronous wrapper, calls `asyncio.run()`. Use from CLI / plain Python scripts.
- `arun(...)` — async version, use with `await` in Jupyter notebooks (avoids the Python 3.12 contextvars conflict that occurs when `asyncio.run()` is called inside an already-running event loop).
1. Generate variants from each prompt in the current pool (parallel via `asyncio.gather`)
2. Evaluate all candidates including parents (all parallel)
3. Select top-K by score
4. Mutate top-K to seed the next pool
5. Save checkpoint; repeat for N iterations

All three components share a single `CostTracker` instance (passed at construction), accumulated across the run.

**`core/metrics.py`** — deterministic and semantic metrics for dataset-specific scoring. Each function takes `(generated, reference)` and returns a float in [0, 1]:
- `exact_match` — normalises case/whitespace before comparison (sentiment, language, ticket)
- `regex_validity(generated, test_cases)` — checks compilation, then pass rate on `{"input", "should_match"}` cases (regex)
- `code_execution(generated, test_cases)` — runs code in an isolated subprocess (timeout 5 s, tempfile), returns pass rate on `{"input", "expected_output"}` cases (code)
- `rouge_l` — ROUGE-L F-measure via `rouge_score.RougeScorer(use_stemmer=True)` (summarization)
- `f1_token` — token-level F1 after lowercasing and split() (extraction)
- `semantic_similarity` — cosine similarity with `paraphrase-multilingual-MiniLM-L12-v2`; model lazy-loaded on first call via global `_sentence_model = None` (feedback)
- `length_ratio_score` — 1.0 if `1.2 ≤ len(gen)/len(ref) ≤ 3.0`, else 0.5 (feedback complement)

`DATASET_METRICS` maps each dataset name to a list of `{"fn", "weight", "needs_test_cases"}` entries. `compute_deterministic_score(dataset_name, generated, reference, test_cases=None)` returns the weighted average, or `None` if no metric is active (triggers alpha=1.0 fallback in the evaluator).

**`core/cost_tracker.py` — `CostTracker`**: Per-session API cost tracking. Reads model prices, accumulates token counts, prints a rich table summary, and alerts if 80% of the daily budget (`MONTHLY_BUDGET_USD / 30`) is consumed. `track_from_usage(model, response.usage)` is the main entry point.

**`core/models.py`** defines `IterationResult` and `OptimizationRun` dataclasses. `OptimizationRun.cost_summary` stores the `CostTracker.summary()` dict.

**`core/utils.py`**: `parse_json_response` (strips markdown fences, calls `json.loads`), `deduplicate` (order-preserving), `CostStats` (legacy stats accumulator).

**`datasets/examples.py`** provides 10 built-in datasets, each a dict with `initial_prompt`, `task`, and `examples` (`list[{"input": ..., "expected_output": ...}]`).

**`datasets/fixtures/demo_code_run.json`** stores a pre-computed `OptimizationRun` result (3 iterations on the `code` dataset, best score 8.50/10) used by the notebook in mock mode. Structure mirrors `OptimizationRun.to_dict()`.

**`config.yml`** — hybrid scoring configuration. Defines the global `alpha` (default 0.5), the sentence-transformers model name, the code execution timeout, and per-dataset alpha overrides (`sentiment/language/ticket → 0.0` deterministic-only; `security/address → 1.0` LLM-judge-only). Not loaded automatically at runtime; serves as the canonical reference for alpha values to pass at `PromptEvaluator` construction.

**`tests/unit/`** — 111 unit tests (pytest + `unittest.mock`). No real API calls. Key files:
- `test_evaluator.py` — `_parse_score` (JSON parsing, missing keys, clamping) + async `evaluate` with `AsyncMock`
- `test_cost_tracker.py` — price resolution, accumulation, `track_from_usage`, budget alerts, `summary`
- `test_utils.py` — `parse_json_response` (fences, embedded JSON, malformed) + `deduplicate`
- `test_metrics.py` — 50 tests covering all 7 metric functions + `compute_deterministic_score`; `SentenceTransformer` always mocked; lazy-loading verified via `importlib.reload`; subprocess timeout tested with an infinite loop (≤ 8 s wall time)

**`notebooks/demo.ipynb`** — `USE_MOCK = True` by default: loads the fixture without any API key. Set to `False` for real API calls. Seed fixed at `RANDOM_SEED = 42`.

## Key implementation notes

- All API calls use `anthropic.AsyncAnthropic(max_retries=3)` with tenacity `@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))`.
- Prompt caching: system prompt and reference examples carry `cache_control: {"type": "ephemeral"}`. Stable content always precedes variable content (candidate prompt is never cached).
- Variant distribution: `base = n_variants // pool_size`, first `n_variants % pool_size` parents get one extra — ensures exactly `n_variants` variants total.
- `ANTHROPIC_API_KEY` is loaded via `python-dotenv` in `main.py`; never hardcoded.
- Hybrid scoring: deterministic metrics return [0, 1]; LLM-judge returns [0, 10]. Before combining, `det_score` is multiplied by 10 so both are on the same scale. If `det_score` is `None` (dataset has no active metrics, or `test_cases` not provided for regex/code), alpha is effectively 1.0.
- `SentenceTransformer` is imported at module level inside a `try/except ImportError` (sets to `None` if unavailable) so `@patch("core.metrics.SentenceTransformer")` works in tests without the module being installed. The model itself is never loaded at import time (`_sentence_model = None`).
- `code_execution` writes the generated code to a `tempfile.NamedTemporaryFile` (avoids shell-quoting issues) and runs it via `subprocess.run([sys.executable, path], timeout=5)`. `SystemExit`, `MemoryError`, and all exceptions are caught; the subprocess is killed on timeout.
