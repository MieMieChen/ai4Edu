import hashlib
import json
import secrets
from decimal import Decimal
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from PIL import UnidentifiedImageError
from pymupdf import FileDataError
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from ai4edu.assets import AssetStore
from ai4edu.config import Settings, get_settings
from ai4edu.db import (
    AnswerRecord,
    Asset,
    CandidateRecord,
    Database,
    ExamRecord,
    JobRecord,
    ModelCall,
    Outbox,
    uid,
)
from ai4edu.execution import freeze_settings, strategy_settings
from ai4edu.schemas import ExamInput, JobInput
from ai4edu.storage import make_storage

TERMINAL = {"scored", "ungradable", "failed"}


def job_summary(db: Database, job: JobRecord) -> dict:
    with db.session() as session:
        counts = dict(
            session.execute(
                select(AnswerRecord.state, func.count())
                .where(
                    AnswerRecord.job_id == job.id,
                )
                .group_by(AnswerRecord.state)
            ).all()
        )
        calls = session.scalars(select(ModelCall).where(ModelCall.job_id == job.id)).all()
        total = sum(counts.values())
        completed = sum(counts.get(state, 0) for state in TERMINAL)
        if completed == total:
            state = "completed_with_errors" if counts.get("failed") else "completed"
        else:
            state = (
                "running"
                if counts.get("running") or completed or counts.get("queued")
                else "queued"
            )
        known_cost = sum((c.cost for c in calls if c.cost is not None), Decimal(0))
        total_score = None
        if counts.get("scored", 0) == total:
            results = session.scalars(
                select(AnswerRecord).where(AnswerRecord.job_id == job.id)
            ).all()
            total_score = sum(Decimal(a.result["grade"]["total_score"]) for a in results)
        return {
            "id": job.id,
            "exam_id": job.exam_id,
            "execution_config": job.execution_config,
            "state": state,
            "total": total,
            "completed": completed,
            "counts": counts,
            "total_score": total_score,
            "known_cost_cny": known_cost,
            "budget_cny": job.budget_cny,
            "committed_cost_cny": job.committed_cost,
            "model_calls": len(calls),
            "unpriced_or_unknown_calls": sum(c.cost is None for c in calls),
            "input_tokens": sum((c.usage or {}).get("prompt_tokens", 0) for c in calls),
            "output_tokens": sum((c.usage or {}).get("completion_tokens", 0) for c in calls),
        }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    def authenticate(authorization: Annotated[str | None, Header()] = None):
        if settings.service_api_key is not None:
            expected = f"Bearer {settings.service_api_key.get_secret_value()}"
            if authorization is None or not secrets.compare_digest(authorization, expected):
                raise HTTPException(401, "需要有效的服务访问凭据")

    app = FastAPI(title="AI4Edu 主观题评分", version="0.1.0", dependencies=[Depends(authenticate)])
    db = Database(settings.database_url)
    assets = AssetStore(db, make_storage(settings))
    app.state.db, app.state.settings, app.state.assets = db, settings, assets

    def require_assets(session, identifiers: list[str]):
        found = set(session.scalars(select(Asset.id).where(Asset.id.in_(identifiers))))
        if set(identifiers) != found:
            raise HTTPException(422, "引用了不存在的图片")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/assets", status_code=201)
    def upload_asset(file: Annotated[UploadFile, File()]):
        data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "文件超过配置的大小限制")
        if file.content_type not in {"image/jpeg", "image/png", "image/webp", "application/pdf"}:
            raise HTTPException(415, "支持 JPEG、PNG、WebP 和 PDF")
        try:
            saved = assets.upload(data, file.content_type)
        except (ValueError, UnidentifiedImageError, FileDataError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return [{"id": a.id, "width": a.width, "height": a.height} for a in saved]

    @app.get("/v1/assets/{asset_id}")
    def get_asset(asset_id: str):
        with db.session() as session:
            if session.get(Asset, asset_id) is None:
                raise HTTPException(404, "图片不存在")
        return Response(assets.image(asset_id).data, media_type="image/png")

    @app.post("/v1/exams", status_code=201)
    def create_exam(payload: ExamInput):
        if payload.budget_cny is not None and any(
            settings.model_for(role) not in settings.model_prices
            for role in ("rubric", "layout", "transcribe", "grade", "verify")
        ):
            raise HTTPException(422, "启用试卷总预算时必须配置所有角色模型单价")
        with db.session() as session:
            require_assets(
                session,
                [a for q in payload.questions for a in q.question_assets + q.reference_assets],
            )
            exam = ExamRecord(
                id=uid(),
                title=payload.title,
                questions=[q.model_dump(mode="json") for q in payload.questions],
                score_step=payload.score_step,
                budget_cny=payload.budget_cny,
            )
            session.add(exam)
            session.add(Outbox(task="prepare_exam", unit_id=exam.id))
            session.commit()
            return {"id": exam.id, "state": exam.state}

    @app.get("/v1/exams/{exam_id}")
    def get_exam(exam_id: str):
        with db.session() as session:
            exam = session.get(ExamRecord, exam_id)
            if exam is None:
                raise HTTPException(404, "试卷不存在")
            return {
                "id": exam.id,
                "state": exam.state,
                "title": exam.title,
                "questions": exam.questions,
                "rubrics": exam.rubrics,
                "error": exam.error,
                "budget_cny": exam.budget_cny,
                "committed_cost_cny": exam.committed_cost,
            }

    @app.post("/v1/exams/{exam_id}/retry", status_code=202)
    def retry_exam(exam_id: str):
        with db.session() as session:
            changed = session.execute(
                update(ExamRecord)
                .where(
                    ExamRecord.id == exam_id,
                    ExamRecord.state == "failed",
                )
                .values(state="queued", error=None, lease_until=None, execution_token=None)
            )
            if changed.rowcount != 1:
                if session.get(ExamRecord, exam_id) is None:
                    raise HTTPException(404, "试卷不存在")
                raise HTTPException(409, "只能重试失败的准备任务")
            session.add(Outbox(task="prepare_exam", unit_id=exam_id))
            session.commit()
            return {"id": exam_id, "state": "queued"}

    @app.post("/v1/jobs", status_code=202)
    def create_job(
        payload: JobInput,
        idempotency_key: Annotated[str | None, Header(min_length=1, max_length=128)] = None,
    ):
        config = strategy_settings(settings, payload.strategy) if payload.strategy else settings
        fingerprint = hashlib.sha256(
            json.dumps(payload.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()
        with db.session() as session:
            if idempotency_key:
                existing = session.scalar(
                    select(JobRecord).where(JobRecord.idempotency_key == idempotency_key)
                )
                if existing:
                    if existing.fingerprint != fingerprint:
                        raise HTTPException(409, "同一个幂等键对应的提交内容不同")
                    return job_summary(db, existing)
            exam = session.get(ExamRecord, payload.exam_id)
            if exam is None:
                raise HTTPException(404, "试卷不存在")
            if exam.state != "ready":
                raise HTTPException(409, "试卷评分细则尚未准备完成")
            expected = {q["id"] for q in exam.questions}
            identifiers = []
            for candidate in payload.candidates:
                if candidate.kind == "question":
                    if {a.question_id for a in candidate.answers} != expected:
                        raise HTTPException(
                            422, "按题输入必须覆盖试卷中的所有主观题，且不能包含未知题号"
                        )
                    identifiers.extend(a for answer in candidate.answers for a in answer.assets)
                else:
                    identifiers.extend(candidate.pages)
            require_assets(session, identifiers)
            if payload.budget_cny is not None:
                models = [
                    config.model_for(role) for role in ("layout", "transcribe", "grade", "verify")
                ]
                if any(model not in config.model_prices for model in models):
                    raise HTTPException(422, "启用费用预算时必须配置所用模型的输入/输出单价")
            job = JobRecord(
                id=uid(),
                exam_id=exam.id,
                idempotency_key=idempotency_key,
                fingerprint=fingerprint,
                execution_config=freeze_settings(config),
                budget_cny=payload.budget_cny,
            )
            candidate_rows, answer_rows, messages = [], [], []
            for candidate in payload.candidates:
                record = CandidateRecord(
                    id=uid(),
                    job_id=job.id,
                    candidate_id=candidate.candidate_id,
                    payload=candidate.model_dump(mode="json"),
                )
                candidate_rows.append(record)
                answer_rows.extend(
                    AnswerRecord(
                        id=uid(), job_id=job.id, candidate_record_id=record.id, question_id=q["id"]
                    )
                    for q in exam.questions
                )
                messages.append(Outbox(task="prepare_candidate", unit_id=record.id))
            try:
                session.add(job)
                session.flush()
                session.add_all(candidate_rows)
                session.flush()
                session.add_all(answer_rows + messages)
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                existing = (
                    session.scalar(
                        select(JobRecord).where(JobRecord.idempotency_key == idempotency_key)
                    )
                    if idempotency_key
                    else None
                )
                if existing is None:
                    raise
                if existing.fingerprint != fingerprint:
                    raise HTTPException(409, "同一个幂等键对应的提交内容不同") from exc
                return job_summary(db, existing)
            return job_summary(db, job)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        with db.session() as session:
            job = session.get(JobRecord, job_id)
            if job is None:
                raise HTTPException(404, "批次不存在")
            return job_summary(db, job)

    @app.get("/v1/jobs/{job_id}/results")
    def job_results(
        job_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=1000)
    ):
        with db.session() as session:
            if session.get(JobRecord, job_id) is None:
                raise HTTPException(404, "批次不存在")
            candidates = session.scalars(
                select(CandidateRecord)
                .where(CandidateRecord.job_id == job_id)
                .order_by(CandidateRecord.candidate_id)
                .offset(offset)
                .limit(limit)
            ).all()
            output = []
            for candidate in candidates:
                answers = session.scalars(
                    select(AnswerRecord)
                    .where(AnswerRecord.candidate_record_id == candidate.id)
                    .order_by(AnswerRecord.question_id)
                ).all()
                total = (
                    sum((Decimal(a.result["grade"]["total_score"]) for a in answers), Decimal(0))
                    if all(a.state == "scored" for a in answers)
                    else None
                )
                output.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "total_score": total,
                        "answers": [
                            {
                                "id": a.id,
                                "question_id": a.question_id,
                                "state": a.state,
                                "result": a.result,
                                "error": a.error,
                                "source_regions": a.assets,
                            }
                            for a in answers
                        ],
                    }
                )
            return output

    @app.get("/v1/jobs/{job_id}/export")
    def export_job(job_id: str):
        with db.session() as session:
            if session.get(JobRecord, job_id) is None:
                raise HTTPException(404, "批次不存在")

        def rows():
            for offset in range(0, 1000, 100):
                results = job_results(job_id, offset, 100)
                for result in results:
                    yield json.dumps(result, ensure_ascii=False, default=str) + "\n"
                if len(results) < 100:
                    break

        return StreamingResponse(
            rows(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.jsonl"'},
        )

    @app.post("/v1/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str):
        with db.session() as session:
            job = session.get(JobRecord, job_id)
            if job is None:
                raise HTTPException(404, "批次不存在")
            candidates = session.scalars(
                select(CandidateRecord).where(CandidateRecord.job_id == job_id)
            ).all()
            for candidate in candidates:
                changed = session.execute(
                    update(CandidateRecord)
                    .where(
                        CandidateRecord.id == candidate.id,
                        CandidateRecord.state == "failed",
                    )
                    .values(state="queued", error=None, lease_until=None, execution_token=None)
                    .returning(CandidateRecord.id, CandidateRecord.error_kind),
                    execution_options={"synchronize_session": False},
                ).first()
                if changed is not None:
                    preparation = {"error_kind": None}
                    if changed.error_kind == "missing_regions":
                        preparation["prepared_pages"] = {}
                        session.execute(
                            update(ModelCall)
                            .where(
                                ModelCall.unit_id == candidate.id,
                                ModelCall.role == "layout",
                                ModelCall.state == "succeeded",
                            )
                            .values(state="invalidated")
                        )
                    session.execute(
                        update(CandidateRecord)
                        .where(CandidateRecord.id == candidate.id)
                        .values(**preparation),
                        execution_options={"synchronize_session": False},
                    )
                    session.execute(
                        update(AnswerRecord)
                        .where(
                            AnswerRecord.candidate_record_id == candidate.id,
                            AnswerRecord.state == "failed",
                        )
                        .values(state="waiting", error=None, execution_token=None, lease_until=None)
                    )
                    session.add(Outbox(task="prepare_candidate", unit_id=candidate.id))
                else:
                    changed_answers = session.scalars(
                        update(AnswerRecord)
                        .where(
                            AnswerRecord.candidate_record_id == candidate.id,
                            AnswerRecord.state == "failed",
                        )
                        .values(state="queued", error=None, lease_until=None, execution_token=None)
                        .returning(AnswerRecord.id)
                    ).all()
                    session.add_all(
                        Outbox(task="grade_answer", unit_id=ident) for ident in changed_answers
                    )
            session.commit()
            return job_summary(db, job)

    return app
