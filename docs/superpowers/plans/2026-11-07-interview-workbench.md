# Interview Workbench Implementation Plan

> Execute backend and UI as bounded parallel tasks, integrate them, then run a focused independent review. The user chose AI application/Agent engineering and needs a deliverable today.

**Goal:** Deliver a local, real-data grading workbench and a short interview demonstration, reusing the existing normal-input grading backend.

**Architecture:** Serve buildless HTML/CSS/JS at `/demo/` from FastAPI. Keep model requests in the existing queues and preserve exam-level budgets. Read-only workbench/evaluation endpoints expose real stored results; offline reference labels never enter grading calls. Add an explicit question subset to existing job submission so a live single-question demo reuses a ready rubric.

**Tech stack:** Existing FastAPI, SQLAlchemy, Celery, PostgreSQL/Redis; native browser JavaScript and CSS; existing pytest and CUA browser verification. No new cloud hosting or frontend dependencies.

## Task 1: Single-question submission

Files: `schemas.py`, `api.py`, `pipeline.py`, `tests/test_question_subset.py`.

- [x] Add failing tests for `question_ids=["16"]` creating exactly one answer, invalid/duplicate/empty selections, default full-paper compatibility, and missing selected answers.
- [x] Add `JobInput.question_ids: list[str] | None`, validate nonempty unique IDs, and validate membership against the ready exam.
- [x] Create answer rows only for selected questions. Include selection in the existing request fingerprint. Preparation derives requested IDs from actual candidate answer rows. Shared exam/rubric/budget remains unchanged.
- [x] Run targeted tests and existing API/pipeline tests.

## Task 2: Workbench read APIs

Files: new `workbench.py`, `config.py`, `api.py`, `tests/test_workbench.py`.

- [x] Test recent batches, available exams, authentication and offline evaluation before implementation.
- [x] Implement `GET /v1/workbench`: real recent jobs with state counts, candidate/answer counts, model/thinking metadata and available exams.
- [x] Implement `GET /v1/jobs/{id}/evaluation` using optional `AI4EDU_EVALUATION_LABELS_FILE`; require complete matching labels for a batch, keep failed answers in denominator, return unavailable explicitly when labels are absent/unmatched.
- [x] Mount static assets at `/demo/`; frontend shell is public, `/v1` remains Bearer protected. Add `/` redirect for convenience.
- [x] Check that evaluation makes no model calls and that labels are excluded from execution snapshots.

## Task 3: Buildless UI

Files owned by UI task: `web/index.html`, `web/app.css`, `web/app.js`.

- [x] Real recent batches and student/question navigation; paginated results.
- [x] Student image, rubric items, scores, reasons and evidence-linked transcript. Unknown scores never become zero; no invented ink bounding boxes.
- [x] Known cost, unknown usage, calls and offline score agreement clearly labelled.
- [x] Explicit budgeted live submission: upload answer, choose ready exam/question, submit one selected question, poll actual progress. No automatic resubmission or model retries.
- [x] Optional service-token input; authorized image fetch and export; escape all model/student text.
- [x] Responsive interface without external assets or dependencies.

## Task 4: Verification and delivery

- [x] Full pytest, Ruff, compile, wheel resource inspection and Compose config.
- [x] Launch the real local backend, open `/demo/` in browser and verify image, question selection, evidence navigation and offline metrics; verify JSONL export via the API.
- [x] Run one budgeted real single-question demo within the original 50 CNY exam budget and preserve the result; no rerunning the whole benchmark.
- [x] Independent read-only review and resolve material findings.
- [x] Add README/runbook entry and `docs/interview-demo.md`: five-minute demonstration, architecture/tradeoffs, measured limitations, AI assistance disclosure without inventing personal contributions.
- Before delivery, sync code only to the existing repository; datasets, labels, outputs and credentials remain local.

Scope review: no claim that current scores meet production accuracy; no new teacher approval step. This is a focused demonstration of orchestration, traceability, evaluation and cost control.

Latest scope: the user asked to focus on the complete technical workflow and stop frontend optimization. Existing functional UI is retained, with no further visual work. The final handoff centers on `docs/technical-workflow.md`, API/queue validation and code synchronization.
