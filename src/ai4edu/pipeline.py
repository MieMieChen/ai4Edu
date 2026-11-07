import hashlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal
from functools import partial

from sqlalchemy import select, update

from ai4edu import prompts
from ai4edu.assets import AssetStore
from ai4edu.bailian import BailianClient, ModelImage
from ai4edu.calls import Calls
from ai4edu.config import Settings
from ai4edu.db import (
    AnswerRecord,
    CandidateRecord,
    Database,
    ExamRecord,
    JobRecord,
    Outbox,
    StaleExecution,
    lock_unit,
)
from ai4edu.images import crop_and_rotate, image_info, pad_region_boxes
from ai4edu.limits import RateLimitWait, RedisLimiter
from ai4edu.schemas import CriterionScore, Judgment, PageLayout, Question, Rubric, Transcript
from ai4edu.scoring import finalize_grade, needs_verification, validate_rubric, validate_transcript


class Pipeline:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        assets: AssetStore,
        client: BailianClient,
        limiter: RedisLimiter,
    ):
        self.db, self.settings, self.assets = db, settings, assets
        self.client, self.limiter = client, limiter

    def calls(
        self, task: str, unit_id: str, exam_id: str, token: str, job_id: str | None = None
    ) -> Calls:
        if job_id is not None:
            with self.db.session() as session:
                job = session.get(JobRecord, job_id)
                if job.execution_config.get("prompt_version") != prompts.VERSION:
                    raise ValueError("批次提示词版本与当前代码不一致，请创建新批次；不复用旧检查点")
        return Calls(
            self.db,
            self.settings,
            self.client,
            self.limiter,
            task=task,
            unit_id=unit_id,
            exam_id=exam_id,
            job_id=job_id,
            execution_token=token,
        )

    def images(self, ids: list[str]) -> list[ModelImage]:
        result = []
        for ident in ids:
            image = self.assets.image(ident)
            data = crop_and_rotate(image.data, None, 0, self.settings.max_pixels)
            mime, width, height = image_info(data)
            result.append(ModelImage(ident, data, mime, width, height))
        return result

    def prepare_exam(self, exam_id: str, token: str):
        with self.db.session() as session:
            exam = session.get(ExamRecord, exam_id)
            questions, rubrics, step = exam.questions, dict(exam.rubrics), exam.score_step
        calls = self.calls("prepare_exam", exam_id, exam_id, token)
        for item in questions:
            question = Question.model_validate(item)
            if question.id in rubrics:
                continue
            images = self.images(question.question_assets + question.reference_assets)
            validator = partial(
                validate_rubric, question_id=question.id, max_score=question.max_score, step=step
            )
            draft = calls.complete(
                "rubric",
                Rubric,
                prompts.rubric_prompt(question, str(step)),
                images,
                validate=partial(validator, require_total=False),
            )
            final = calls.complete(
                "rubric",
                Rubric,
                prompts.check_rubric_prompt(question, draft, str(step)),
                images,
                validate=validator,
            )
            rubrics[question.id] = final.model_dump(mode="json")
            with self.db.session() as session:
                exam = lock_unit(session, ExamRecord, exam_id, token)
                exam.rubrics = dict(rubrics)
                session.commit()
        with self.db.session() as session:
            exam = lock_unit(session, ExamRecord, exam_id, token)
            exam.state, exam.error, exam.lease_until = "ready", None, None
            exam.execution_token = None
            session.commit()

    def prepare_candidate(self, candidate_id: str, token: str):
        with self.db.session() as session:
            candidate = session.get(CandidateRecord, candidate_id)
            job = session.get(JobRecord, candidate.job_id)
            exam = session.get(ExamRecord, job.exam_id)
            payload, prepared = candidate.payload, dict(candidate.prepared_pages)
            job_id, exam_id = job.id, exam.id
            question_ids = list(
                session.scalars(
                    select(AnswerRecord.question_id)
                    .where(AnswerRecord.candidate_record_id == candidate_id)
                    .order_by(AnswerRecord.question_id)
                )
            )
        cropped = payload["kind"] == "question"
        pages = (
            [(a, answer["question_id"]) for answer in payload["answers"] for a in answer["assets"]]
            if cropped
            else [(a, None) for a in payload["pages"]]
        )
        calls = self.calls("prepare_candidate", candidate_id, exam_id, token, job_id)

        def valid_layout(value):
            if any(r.question_id not in question_ids for r in value.regions):
                raise ValueError("版面解析包含未知题号")

        for page_index, (asset_id, question_id) in enumerate(pages):
            cache_key = f"{page_index}:{asset_id}:{question_id}"
            if cache_key in prepared:
                continue
            image = self.assets.image(asset_id)
            if cropped and not self.settings.orient_question_images:
                rotation = 0
            else:
                layout = calls.complete(
                    "layout",
                    PageLayout,
                    prompts.layout_prompt(question_ids, cropped=cropped),
                    self.images([asset_id]),
                    validate=valid_layout,
                )
                rotation = layout.rotation
            if cropped:
                regions = [(question_id, None, 0, None)]
            else:
                padded = pad_region_boxes(
                    [(region.question_id, region.bbox) for region in layout.regions],
                    (image.width, image.height),
                    self.settings.region_padding_ratio,
                )
                regions = [
                    (region.question_id, box, region.order, region.bbox)
                    for region, box in zip(layout.regions, padded, strict=True)
                ]
            mapped = []
            for qid, box, order, detected_box in regions:
                data = crop_and_rotate(image.data, box, rotation, self.settings.max_pixels)
                derived = self.assets.put_image(data)
                mapped.append(
                    {
                        "question_id": qid,
                        "asset_id": derived.id,
                        "source_asset_id": asset_id,
                        "bbox": box,
                        "detected_bbox": detected_box,
                        "rotation": rotation,
                        "page_index": page_index,
                        "order": order,
                    }
                )
            prepared[cache_key] = mapped
            with self.db.session() as session:
                candidate = lock_unit(session, CandidateRecord, candidate_id, token)
                candidate.prepared_pages = dict(prepared)
                session.commit()
        with self.db.session() as session:
            candidate = lock_unit(session, CandidateRecord, candidate_id, token)
            answers = session.scalars(
                select(AnswerRecord).where(AnswerRecord.candidate_record_id == candidate_id)
            ).all()
            missing = []
            for answer in answers:
                if answer.state not in {"waiting", "failed"}:
                    continue
                regions = [
                    r
                    for page in prepared.values()
                    for r in page
                    if r["question_id"] == answer.question_id
                ]
                if not regions:
                    changes = {"state": "failed", "error": "未定位到该题作答区域；不按零分处理"}
                else:
                    ordered = sorted(regions, key=lambda r: (r["page_index"], r["order"]))
                    changes = {"assets": ordered, "state": "queued", "error": None}
                    if answer.assets != ordered:
                        changes.update(intermediate={}, result=None)
                changed = session.execute(
                    update(AnswerRecord)
                    .where(
                        AnswerRecord.id == answer.id,
                        AnswerRecord.state.in_(["waiting", "failed"]),
                        AnswerRecord.execution_token.is_(None),
                    )
                    .values(**changes),
                    execution_options={"synchronize_session": False},
                )
                if changed.rowcount:
                    if regions:
                        session.add(Outbox(task="grade_answer", unit_id=answer.id))
                    else:
                        missing.append(answer.question_id)
            candidate.state = "failed" if missing else "ready"
            candidate.error_kind = "missing_regions" if missing else None
            candidate.error = f"未定位到题目：{', '.join(missing)}" if missing else None
            candidate.lease_until, candidate.execution_token = None, None
            session.commit()

    def grade_answer(self, answer_id: str, token: str):
        with self.db.session() as session:
            answer = session.get(AnswerRecord, answer_id)
            job = session.get(JobRecord, answer.job_id)
            exam = session.get(ExamRecord, job.exam_id)
            question = Question.model_validate(
                next(q for q in exam.questions if q["id"] == answer.question_id)
            )
            rubric = Rubric.model_validate(exam.rubrics[question.id])
            intermediate, source_regions = dict(answer.intermediate), answer.assets
            exam_id, job_id, step = exam.id, job.id, exam.score_step
        if not source_regions:
            raise ValueError("该题没有作答图片，必须先完成题目定位")
        images = self.images([r["asset_id"] for r in source_regions])
        # Repeated fragments of the same image share their asset/evidence ID intentionally.
        calls = self.calls("grade_answer", answer_id, exam_id, token, job_id)

        def checkpoint(key: str, value):
            intermediate[key] = value.model_dump(mode="json")
            with self.db.session() as session:
                record = lock_unit(session, AnswerRecord, answer_id, token)
                record.intermediate = dict(intermediate)
                session.commit()

        def transcribe_parts(prefix: str, role: str) -> Transcript:
            size = self.settings.transcription_images_per_call
            chunks = [images[index : index + size] for index in range(0, len(images), size)]
            parts = {}
            missing = []
            for index, chunk in enumerate(chunks):
                key = f"{prefix}_transcript_part_{index + 1}"
                if key in intermediate:
                    part = Transcript.model_validate(intermediate[key])
                    validate_transcript(part, {image.id for image in chunk})
                    parts[index] = part
                else:
                    missing.append(index)

            def transcribe_part(index: int):
                chunk = chunks[index]
                return calls.complete(
                    role,
                    Transcript,
                    prompts.transcription_prompt(),
                    chunk,
                    validate=lambda value: validate_transcript(
                        value, {image.id for image in chunk}
                    ),
                )

            if missing:
                concurrency = min(self.settings.transcription_concurrency, len(missing))
                remaining = iter(missing[concurrency:])
                failures = []
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    pending = {
                        executor.submit(transcribe_part, index): index
                        for index in missing[:concurrency]
                    }
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            index = pending.pop(future)
                            try:
                                part = future.result()
                                # Only this coordinator writes the shared intermediate dictionary.
                                checkpoint(f"{prefix}_transcript_part_{index + 1}", part)
                                parts[index] = part
                            except Exception as exc:
                                failures.append(exc)
                        # A failed part must not return the unit to the queue while paid
                        # requests are still running. Drain them and checkpoint successes.
                        if not failures:
                            while len(pending) < concurrency:
                                index = next(remaining, None)
                                if index is None:
                                    break
                                pending[executor.submit(transcribe_part, index)] = index
                if failures:
                    # A capacity retry must not hide a fatal request failure or a
                    # transferred execution lease observed while draining other parts.
                    failure = next((e for e in failures if isinstance(e, StaleExecution)), None)
                    if failure is None:
                        failure = next(
                            (e for e in failures if not isinstance(e, RateLimitWait)), None
                        )
                    if failure is None:
                        failure = max(failures, key=lambda e: e.seconds)
                    raise failure

            lines, issues = [], []
            for index, chunk in enumerate(chunks):
                part = parts[index]
                positions = {}
                for position, image in enumerate(chunk):
                    positions.setdefault(image.id, position)
                lines.extend(
                    line.model_copy(update={"id": f"P{index + 1}:{line.id}"})
                    for line in sorted(part.lines, key=lambda line: positions[line.image_id])
                )
                issues.extend(part.issues)
            transcript = Transcript(
                lines=lines,
                is_blank=all(part.is_blank for part in parts.values()),
                issues=issues,
            )
            validate_transcript(transcript, {image.id for image in images})
            return transcript

        def score_pass(prefix: str, transcription_role: str, grading_role: str):
            transcription_key = prefix + "_transcript"
            if transcription_key in intermediate:
                transcript = Transcript.model_validate(intermediate[transcription_key])
            elif self.settings.transcription_images_per_call is not None:
                transcript = transcribe_parts(prefix, transcription_role)
                checkpoint(transcription_key, transcript)
            else:
                transcript = calls.complete(
                    transcription_role,
                    Transcript,
                    prompts.transcription_prompt(),
                    images,
                    validate=lambda value: validate_transcript(
                        value, {image.id for image in images}
                    ),
                )
                checkpoint(transcription_key, transcript)
            judgment_key = prefix + "_judgment"
            if judgment_key in intermediate:
                judgment = Judgment.model_validate(intermediate[judgment_key])
            elif transcript.is_blank:
                judgment = Judgment(
                    scores=[
                        CriterionScore(
                            criterion_id=c.id,
                            score=0,
                            evidence_ids=[],
                            reason="作答区域确认为空白",
                        )
                        for c in rubric.criteria
                    ]
                )
            else:
                judgment = calls.complete(
                    grading_role,
                    Judgment,
                    prompts.grading_prompt(question, rubric, transcript, str(step)),
                    images + self.images(question.question_assets),
                    validate=lambda value: finalize_grade(rubric, transcript, value, Decimal(step)),
                )
            grade = finalize_grade(rubric, transcript, judgment, Decimal(step))
            checkpoint(judgment_key, judgment)
            return transcript, grade

        primary_transcript, primary_grade = score_pass("primary", "transcribe", "grade")
        verify = self.settings.verification_policy == "all" or (
            self.settings.verification_policy == "risk"
            and needs_verification(
                primary_transcript,
                primary_grade,
                answer_id,
                self.settings.audit_rate,
            )
        )
        intermediate["verification_requested"] = verify
        with self.db.session() as session:
            answer = lock_unit(session, AnswerRecord, answer_id, token)
            answer.intermediate = dict(intermediate)
            session.commit()
        transcript, grade = (
            score_pass("verified", "verify", "verify")
            if verify
            else (primary_transcript, primary_grade)
        )
        result = {
            "grade": grade.model_dump(mode="json"),
            "transcript": transcript.model_dump(mode="json"),
            "verified": verify,
            "primary_grade": primary_grade.model_dump(mode="json"),
            "rubric_version": hashlib.sha256(rubric.model_dump_json().encode()).hexdigest(),
            "prompt_version": prompts.VERSION,
        }
        with self.db.session() as session:
            answer = lock_unit(session, AnswerRecord, answer_id, token)
            answer.result, answer.state = result, grade.status
            answer.error, answer.lease_until = None, None
            answer.execution_token = None
            session.commit()
