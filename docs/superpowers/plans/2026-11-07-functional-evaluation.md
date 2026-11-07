# Functional grading and model evaluation implementation plan

> **For agentic workers:** Use subagent-driven-development for the isolated evaluation-data task; execute infrastructure and integration sequentially in the controller. All paid requests use the existing Bailian credentials, with recorded usage and bounded experiment budgets.

**Goal:** Run the real asynchronous grading backend, deliver a reproducible A/B/C comparison on the supplied subjective questions, and save an evidence-based default configuration.

**Architecture:** Keep the existing FastAPI/Celery/PostgreSQL/Redis pipeline. Add local experiment tooling which freezes one exam rubric and one set of prepared answer assets, runs Max and Plus on identical inputs, and computes C using only the frozen runtime routing rule. CSV scores remain in a separate offline label file. A real C execution confirms production routing before final configuration selection.

**Tech stack:** Existing Python package, SQLAlchemy, httpx, Celery, Redis, PostgreSQL, Docker/Colima; standard-library JSONL/CSV statistics.

## Accepted scope and constraints

- User approved steps 1–3 of the preceding workflow: small real end-to-end run, A/B/C evaluation tooling and comparison, model/configuration selection.
- Subjective questions only; no teacher approval, silent model substitution, JSON repair, fabricated zero scores, or runtime mock grading.
- Existing dataset: 50 students × questions 15–19. Deterministic student-group split: 5 debug, 10 initial evaluation, 35 holdout. All variants stay within the student's split.
- First use 1–2 debug candidates; A/B use the 10-student initial set. Preserve all failed/ungradable records in reports. Selection is provisional for this exam, not a cross-exam quality claim.
- Reuse identical immutable rubric, prompt version, image assets, resolution, thinking and token limits between A/B and C branches. C simulation records hypothetical production cost separately from actual experiment spend.
- Record per-stage calls, tokens, costs, P50/P95 and coverage. Budget cap for the initial experiment is 50 CNY including shared preparation; stop issuing new paid work at the cap. Provider timeouts with unknown usage remain unknown/reserved.
- No Git repository exists here; work directly in the authorized workspace and do not create a repository.

## Task 1: Runtime prerequisites and real service integration

- [ ] Start the existing Colima VM without changing the active Docker context or SSH configuration.
- [ ] Obtain the official Compose binary in the ignored data/tools directory; verify its release checksum.
- [ ] Start PostgreSQL and Redis, initialize the real database, then start API/worker/scheduler processes or containers.
- [ ] Verify database connectivity, Redis ping, HTTP health, worker task registration and queue dispatch.
- [ ] Submit 1–2 debug students through the real HTTP API and collect real terminal results and model usage.

Commands: `colima start --activate=false --ssh-config=false`; `docker info`; `docker compose up -d postgres redis`; `.venv/bin/ai4edu init-db`; real API submit and poll. Record exact runtime commands and any environment-specific alternative in the run report.

## Task 2: Anonymous evaluation data and offline metrics

Files: create `src/ai4edu/evaluation_data.py`, `src/ai4edu/evaluation.py`, `tests/test_evaluation_data.py`, `tests/test_evaluation.py`.

- [ ] Write tests for deterministic 5/10/35 student splits, anonymous model manifests, exact score joins, and exclusion of all labels/names/student numbers from model inputs.
- [ ] Run the new tests to observe missing behavior, then implement `prepare_evaluation_data(archive: Path, output: Path, seed: int = 20261107) -> dict`.
- [ ] Store `inputs/manifest.json`, `labels.jsonl`, and `splits.json` separately. Label records contain `candidate_id`, `question_id`, `score`, `max_score`, `split`; input manifests never contain existing scores.
- [ ] Test and implement `evaluate_predictions(predictions: list[dict], labels: list[dict]) -> dict`. Prediction records contain `candidate_id`, `question_id`, `status`, `score`, and optional `cost_cny`, `latency_seconds`, `escalated`.
- [ ] Report denominator-aware coverage, resolved MAE, exact/within1/within2, normalized error, signed-difference distribution, per-question values, complete-five-question subjective totals, large differences, and unknown cost counts. Do not silently omit missing predictions.
- [ ] Test and implement paired A/B/C comparison and C upgrade capture/benefit metrics. No label is used in routing.

Verification: `.venv/bin/python -m pytest tests/test_evaluation.py tests/test_evaluation_data.py --tb=short`; `.venv/bin/ruff check src/ai4edu/evaluation*.py tests/test_evaluation*.py`.

## Task 3: Reproducible experiment runner

Files: create `src/ai4edu/benchmark.py`, `tests/test_benchmark.py`; extend CLI and documentation.

- [ ] Write failing tests proving shared frozen rubric/assets, independent model passes, faithful runtime C routing, and simulated-versus-real cost separation.
- [ ] Freeze prepared answer assets from the production pipeline; record image hashes, rubric hashes, prompt version and effective settings.
- [ ] Execute A and B with the same production grading functions and actual database records. Reuse successful identical calls only with a matching complete fingerprint; report reused costs separately.
- [ ] Derive C using B runtime evidence and the same stable audit key used by production. Reuse A only when input/model/prompt/schema/settings fingerprints match.
- [ ] Produce JSONL predictions, JSON and Markdown reports, per-call cost/latency records, model selection decision and reproducible run manifest. Labels are loaded only after model execution.
- [ ] Expose CLI commands for data preparation, running comparisons and rebuilding a report from existing artifacts.

Verification: targeted benchmark tests; CLI `--help`; real experiment artifacts contain every selected student/question and preserve failure states.

## Task 4: Real data debugging and initial comparison

- [ ] Visually inspect anonymous sample pages, teacher question images and reference images before sending real inputs.
- [ ] Confirm that layout crops contain complete student work, including continued answers; fix implementation defects with regression tests.
- [ ] Freeze settings after the debug run. Execute A/B on 10 initial students (50 questions), derive C, and record every actual and estimated charge.
- [ ] Independently run the chosen candidate on fresh holdout students to verify the actual production path and both input modes, without selecting routing using labels.

## Task 5: Select and save the default configuration

- [ ] Compare paired score error, automatic coverage, major-error count, time and estimated production cost.
- [ ] Prefer quality/coverage first; if quality differences are too small for this sample, document uncertainty and choose the more economical supported configuration as provisional.
- [ ] Save nonsecret effective configuration and decision evidence; update `.env` only with the selected model roles and tested parameters.
- [ ] Preserve original dataset and original model evaluation proposal unchanged.

## Task 6: Final review and verification

- [ ] Independent spec review, then code-quality review of new evaluation functionality and integration fixes.
- [ ] Run full pytest, Ruff, compile, build and live HTTP query/export checks.
- [ ] Update `docs/verification.md`, README and a dated experiment report with measured results, exact costs, limitations and launch instructions.
- [ ] Keep only intentional runtime services and document how to stop/restart them. Do not claim 1000-person production throughput or cross-paper accuracy from this experiment.
