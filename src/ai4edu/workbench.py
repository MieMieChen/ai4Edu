"""Local demonstration shell and read-only grading/evaluation views."""

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from ai4edu.config import Settings
from ai4edu.db import AnswerRecord, CandidateRecord, Database, ExamRecord, JobRecord
from ai4edu.evaluation import evaluate_predictions

TERMINAL = {"scored", "ungradable", "failed"}


def install_workbench(app: FastAPI, db: Database, settings: Settings):
    # The static shell contains no private data. /v1 routes retain app authentication.
    app.mount(
        "/demo", StaticFiles(directory=Path(__file__).with_name("web"), html=True), name="demo"
    )

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/demo/")

    @app.get("/v1/workbench")
    def workbench():
        with db.session() as session:
            jobs = session.scalars(
                select(JobRecord).order_by(JobRecord.created_at.desc()).limit(30)
            ).all()
            exams = session.scalars(
                select(ExamRecord).order_by(ExamRecord.created_at.desc()).limit(30)
            ).all()
            exam_index = {exam.id: exam for exam in exams}
            counts = defaultdict(dict)
            candidates = {}
            if jobs:
                ids = [job.id for job in jobs]
                for job_id, state, count in session.execute(
                    select(AnswerRecord.job_id, AnswerRecord.state, func.count())
                    .where(AnswerRecord.job_id.in_(ids))
                    .group_by(AnswerRecord.job_id, AnswerRecord.state)
                ):
                    counts[job_id][state] = count
                candidates = dict(
                    session.execute(
                        select(CandidateRecord.job_id, func.count())
                        .where(CandidateRecord.job_id.in_(ids))
                        .group_by(CandidateRecord.job_id)
                    ).all()
                )
            rows = []
            for job in jobs:
                exam = exam_index.get(job.exam_id) or session.get(ExamRecord, job.exam_id)
                states = counts[job.id]
                total = sum(states.values())
                completed = sum(count for state, count in states.items() if state in TERMINAL)
                state = (
                    ("completed_with_errors" if states.get("failed") else "completed")
                    if total == completed
                    else (
                        "running"
                        if completed or states.get("running") or states.get("queued")
                        else "queued"
                    )
                )
                rows.append(
                    {
                        "id": job.id,
                        "exam_id": job.exam_id,
                        "exam_title": exam.title,
                        "created_at": job.created_at.isoformat() + "Z",
                        "candidate_count": candidates.get(job.id, 0),
                        "answer_count": total,
                        "state": state,
                        "counts": states,
                        "grade_model": job.execution_config.get("grade_model"),
                        "thinking": "grade" in job.execution_config.get("thinking_roles", []),
                    }
                )
            return {
                "jobs": rows,
                "exams": [
                    {
                        "id": exam.id,
                        "title": exam.title,
                        "state": exam.state,
                        "questions": [
                            {key: question[key] for key in ("id", "stem", "max_score")}
                            for question in exam.questions
                        ],
                        "budget_cny": exam.budget_cny,
                        "committed_cost_cny": exam.committed_cost,
                    }
                    for exam in exams
                ],
                "evaluation_available": settings.evaluation_labels_file is not None,
            }

    @app.get("/v1/jobs/{job_id}/evaluation")
    def evaluation(job_id: str):
        with db.session() as session:
            if session.get(JobRecord, job_id) is None:
                raise HTTPException(404, "批次不存在")
            answers = session.execute(
                select(CandidateRecord.candidate_id, AnswerRecord)
                .join(AnswerRecord, AnswerRecord.candidate_record_id == CandidateRecord.id)
                .where(CandidateRecord.job_id == job_id)
                .order_by(CandidateRecord.candidate_id, AnswerRecord.question_id)
            ).all()
            if settings.evaluation_labels_file is None:
                return {"available": False, "reason": "未配置独立参考分数"}
            if any(answer.state not in TERMINAL for _, answer in answers):
                return {"available": False, "reason": "批次完成后计算分数一致性"}
            if not settings.evaluation_labels_file.is_file():
                raise HTTPException(503, "已配置的评测标签文件不存在")
            expected = {(candidate, answer.question_id) for candidate, answer in answers}
            labels = [
                json.loads(line)
                for line in settings.evaluation_labels_file.read_text().splitlines()
                if line.strip()
            ]
            labels = [
                row for row in labels if (row["candidate_id"], row["question_id"]) in expected
            ]
            matched = {(row["candidate_id"], row["question_id"]) for row in labels}
            if matched != expected:
                return {
                    "available": False,
                    "reason": "参考分数未完整覆盖本批次",
                    "matched_count": len(matched),
                    "expected_count": len(expected),
                }
            predictions = [
                {
                    "candidate_id": candidate,
                    "question_id": answer.question_id,
                    "status": answer.state,
                    "score": (answer.result or {}).get("grade", {}).get("total_score"),
                }
                for candidate, answer in answers
            ]
            metrics = evaluate_predictions(predictions, labels)
            good = sum(
                row["score"] is not None and Decimal(row["absolute_error"]) <= 1
                for row in metrics["records"]
            )
            return {
                "available": True,
                "metrics": metrics,
                "all_expected_within_1_rate": good / metrics["expected_count"],
            }
