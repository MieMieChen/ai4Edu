# Fixed Answer Template Implementation Plan

**Goal:** Collect every fixed answer box, preserve its question/subquestion ownership, and refuse incomplete inputs before scoring.

**Architecture:** Optional exam-level `answer_template` is copied into every new JobRecord. A template lists ordered pages with reference width/height and boxes (`id`, `question_id`, optional `subquestion_id`, `bbox`, global per-question `order`). Fixed-template jobs crop exact normalized coordinates without layout inference or padding. Legacy jobs retain their explicitly untemplated path. No frontend changes or paid experiments are required.

**Tech Stack:** Existing Pydantic, SQLAlchemy JSON, FastAPI, Pillow/OpenCV, Celery, pytest.

Approved scope: user confirmed the fixed-template/multiple-box/completeness design and requested implementation and submission. Execute continuously; use one bounded schema/helper implementer and focused independent reviews. Reuse the current clean checkout and existing local service rather than duplicate runtime state.

## Tasks

- [x] Add failing tests for template validation, complete multi-box paper and per-question inputs, missing/unknown/duplicate boxes, missing pages, empty frames, selected questions, template snapshots, and legacy fingerprints.
- [x] Implement schema/helper contract in `schemas.py` and new `answer_templates.py`: validate all question IDs are covered, unique frame IDs and per-question orders, ordered nonempty bboxes, no overlap on a page; compute template hash and filter/sort selected boxes. `AnswerInput` retains `assets` and adds optional aligned `box_ids`; length/uniqueness are mandatory when specified.
- [x] Add nullable `answer_template` JSON to ExamRecord and JobRecord, including explicit additive init-db migrations. Create/get exam and job summary expose template/version; `PUT /v1/exams/{id}/answer-template` updates future-job defaults only. Templates are validated against the exam; shared exam budget and rubric remain unchanged.
- [x] Validate fixed paper page count and named per-question box coverage at POST /jobs before outbox/model work. Copy the complete template into the job. Remove absent `box_ids` from legacy fingerprint payload so old idempotency keys remain valid.
- [x] Route fixed input preparation through exact template boxes. Normalize resolution using relative coordinates; reject pages with relative aspect-ratio mismatch over2% instead of guessing positions. Skip unselected pages, retain blank boxes, store box ID/subquestion/source/version. Distinct boxes need distinct logical asset IDs even for identical pixels; allow `AssetStore.put_image(..., deduplicate=False)` only for this path.
- [x] Validate the selected question's complete box set again before grading. Pass saved box ownership to `grading_prompt` only for template jobs; save the template version in final results. Keep the existing per-image transcription, evidence validation and Decimal scoring. Do not claim additional mathematical correctness from this mapping.
- [x] Verify targeted tests, full suite, Ruff, compilation, wheel build, Compose config and real local PostgreSQL migration/restart. Use local sample scans to exercise exact crops without making paid requests. Check old stored results remain queryable.
- [x] Independent spec review followed by code-quality review; resolve material findings. Update README/technical workflow with exact JSON examples, normal aligned-page assumptions, and no automatic template generation claim.
- Before delivery, commit and push code/docs/tests only to the existing authorized GitHub repository; confirm remote hash and clean tree. Keep scans, labels, local crop audit and credentials ignored.

## Concrete contracts and checks

Example template: `{"pages":[{"width":1000,"height":1400,"boxes":[{"id":"q17-1","question_id":"17","subquestion_id":"1","bbox":[0.1,0.1,0.9,0.45],"order":0},{"id":"q17-2","question_id":"17","subquestion_id":"2","bbox":[0.1,0.5,0.9,0.9],"order":1}]}]}`.

Named question input: `{"question_id":"17","assets":["image-a","image-b"],"box_ids":["q17-1","q17-2"]}`. The pairing is explicit and order-independent; preparation reorders by template order. A whole-question shared frame has `subquestion_id=null`. Cross-page continuation uses another box with the same question/subquestion and another order.

Run `.venv/bin/python -m pytest tests/test_answer_templates.py tests/test_fixed_template_pipeline.py tests/test_question_subset.py tests/test_api.py --tb=short`, then the full suite. Expected red initially: unknown `answer_template`/`box_ids` rejected or absent behavior. Expected green: exact complete coverage, no layout calls for fixed jobs, no model request for invalid inputs, and unchanged legacy job IDs for repeated old payloads.
