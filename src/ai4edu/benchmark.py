"""Reproducible comparisons using production task records and isolated offline labels."""

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select

from ai4edu import prompts
from ai4edu.config import Settings
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
from ai4edu.execution import execution_settings, freeze_settings, strategy_settings
from ai4edu.schemas import Grade, Judgment, Rubric, Transcript
from ai4edu.scoring import finalize_grade, needs_verification


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def create_comparison_job(
    db: Database, settings: Settings, source_job_id: str, strategy: str, *, budget_cny: str | None
) -> str:
    """Queue a production grading pass, reusing an exam and its exact prepared assets."""
    if strategy not in {"A", "B", "C"}:
        raise ValueError("策略必须为 A、B 或 C")
    with db.session() as session:
        source = session.get(JobRecord, source_job_id)
        if source is None:
            raise ValueError("源批次不存在")
        if source.execution_config.get("prompt_version") != prompts.VERSION:
            raise ValueError("源批次提示词版本与当前代码不一致")
        config = strategy_settings(execution_settings(settings, source.execution_config), strategy)
        snapshot = freeze_settings(config)
        candidates = session.scalars(
            select(CandidateRecord)
            .where(CandidateRecord.job_id == source.id)
            .order_by(CandidateRecord.candidate_id)
            .with_for_update()
        ).all()
        if not candidates or any(
            candidate.state not in {"ready", "failed"} for candidate in candidates
        ):
            raise ValueError("必须先完成源批次的输入准备")
        exam = session.get(ExamRecord, source.exam_id)
        prepared_answers = {
            candidate.id: list(
                session.scalars(
                    select(AnswerRecord)
                    .where(AnswerRecord.candidate_record_id == candidate.id)
                    .order_by(AnswerRecord.question_id)
                    .with_for_update()
                )
            )
            for candidate in candidates
        }
        prepared_snapshot = [
            {
                "candidate_id": candidate.candidate_id,
                "answers": [
                    {
                        "question_id": answer.question_id,
                        "source_regions": answer.assets,
                        "image_hashes": [
                            {
                                "id": region["asset_id"],
                                "sha256": session.get(Asset, region["asset_id"]).sha256,
                            }
                            for region in answer.assets
                        ],
                    }
                    for answer in prepared_answers[candidate.id]
                ],
            }
            for candidate in candidates
        ]
        fingerprint = digest(
            {
                "source": source_job_id,
                "strategy": strategy,
                "config": snapshot,
                "budget": budget_cny,
                "questions": exam.questions,
                "rubrics": exam.rubrics,
                "prepared_answers": prepared_snapshot,
            }
        )
        key = f"benchmark:{fingerprint}"
        existing = session.scalar(select(JobRecord).where(JobRecord.idempotency_key == key))
        if existing is not None:
            return existing.id
        job = JobRecord(
            id=uid(),
            exam_id=source.exam_id,
            execution_config=snapshot,
            fingerprint=fingerprint,
            idempotency_key=key,
            budget_cny=Decimal(budget_cny) if budget_cny else None,
        )
        session.add(job)
        session.flush()
        for original in candidates:
            candidate = CandidateRecord(
                id=uid(),
                job_id=job.id,
                candidate_id=original.candidate_id,
                payload=deepcopy(original.payload),
                prepared_pages=deepcopy(original.prepared_pages),
                state=original.state,
                error=original.error,
                error_kind=original.error_kind,
            )
            session.add(candidate)
            session.flush()
            for original_answer in prepared_answers[original.id]:
                prepared = bool(original_answer.assets)
                answer = AnswerRecord(
                    id=uid(),
                    job_id=job.id,
                    candidate_record_id=candidate.id,
                    question_id=original_answer.question_id,
                    assets=deepcopy(original_answer.assets),
                    state="queued" if prepared else "failed",
                    error=None if prepared else original_answer.error,
                )
                session.add(answer)
                if prepared:
                    session.add(Outbox(task="grade_answer", unit_id=answer.id))
        session.commit()
        return job.id


def collect_predictions(db: Database, job_id: str, *, primary: bool = False) -> list[dict]:
    with db.session() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            raise ValueError("批次不存在")
        exam = session.get(ExamRecord, job.exam_id)
        config = job.execution_config
        output = []
        for candidate in session.scalars(
            select(CandidateRecord)
            .where(CandidateRecord.job_id == job_id)
            .order_by(CandidateRecord.candidate_id)
        ):
            for answer in session.scalars(
                select(AnswerRecord)
                .where(AnswerRecord.candidate_record_id == candidate.id)
                .order_by(AnswerRecord.question_id)
            ):
                result = answer.result or {}
                question = next(item for item in exam.questions if item["id"] == answer.question_id)
                rubric = Rubric.model_validate(exam.rubrics[answer.question_id])
                grade, transcript = result.get("grade"), result.get("transcript")
                status = answer.state
                if primary:
                    transcript = answer.intermediate.get("primary_transcript")
                    judgment = answer.intermediate.get("primary_judgment")
                    if transcript is not None and judgment is not None:
                        grade = finalize_grade(
                            rubric,
                            Transcript.model_validate(transcript),
                            Judgment.model_validate(judgment),
                            exam.score_step,
                        ).model_dump(mode="json")
                        status = grade["status"]
                    else:
                        grade, status = None, "failed"
                roles = {"transcribe", "grade"} if primary else {"transcribe", "grade", "verify"}
                calls = session.scalars(
                    select(ModelCall)
                    .where(ModelCall.unit_id == answer.id, ModelCall.role.in_(roles))
                    .order_by(ModelCall.created_at)
                ).all()
                unknown = any(call.cost is None for call in calls)
                cost = None if unknown else str(sum((call.cost for call in calls), Decimal(0)))
                duration = sum(
                    (call.finished_at - call.created_at).total_seconds()
                    for call in calls
                    if call.finished_at
                )
                if any(call.finished_at is None for call in calls):
                    duration = None
                asset_ids = [region["asset_id"] for region in answer.assets] + question.get(
                    "question_assets", []
                )
                assets = [
                    {"id": ident, "sha256": session.get(Asset, ident).sha256} for ident in asset_ids
                ]
                role_pair = (
                    ["verify", "verify"]
                    if primary or result.get("verified")
                    else ["transcribe", "grade"]
                )
                thinking = [role in config.get("thinking_roles", []) for role in role_pair]
                signature = digest(
                    {
                        "question": question,
                        "rubric": exam.rubrics[answer.question_id],
                        "images": assets,
                        "max_pixels": config.get("max_pixels"),
                        "output_limits": {
                            "transcription": config.get("max_transcription_tokens")
                            or config.get("max_output_tokens"),
                            "grading": config.get("max_grading_tokens")
                            or config.get("max_output_tokens"),
                        },
                        "max_input_tokens": config.get("max_input_tokens"),
                        "transcription_images_per_call": config.get(
                            "transcription_images_per_call"
                        ),
                        "transcription_concurrency": config.get("transcription_concurrency", 2)
                        if config.get("transcription_images_per_call")
                        else None,
                        "thinking": thinking,
                        "thinking_budget": config.get("thinking_budget") if any(thinking) else None,
                        "prompt_version": result.get(
                            "prompt_version", config.get("prompt_version")
                        ),
                    }
                )
                output.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "question_id": answer.question_id,
                        "status": status,
                        "score": grade.get("total_score") if grade else None,
                        "answer_id": answer.id,
                        "grade": grade,
                        "transcript": transcript,
                        "branch_model": config.get("grade_model"),
                        "replay_signature": signature,
                        "rubric_version": digest(exam.rubrics[answer.question_id]),
                        "escalated": bool(
                            result.get("verified")
                            or answer.intermediate.get("verification_requested")
                            or any(call.role == "verify" for call in calls)
                        )
                        and not primary,
                        "cost_cny": cost,
                        "latency_seconds": duration,
                        "flow_elapsed_seconds": (
                            max(call.finished_at for call in calls)
                            - min(call.created_at for call in calls)
                        ).total_seconds()
                        if calls and all(call.finished_at for call in calls)
                        else None,
                        "unknown_calls": sum(call.cost is None for call in calls),
                        "error": answer.error,
                        "calls": [
                            {
                                "id": call.id,
                                "role": call.role,
                                "model": call.model,
                                "response_model": (call.response or {}).get("model"),
                                "state": call.state,
                                "fingerprint": call.fingerprint,
                                "cost_cny": str(call.cost) if call.cost is not None else None,
                                "usage": call.usage,
                                "latency_seconds": (
                                    call.finished_at - call.created_at
                                ).total_seconds()
                                if call.finished_at
                                else None,
                            }
                            for call in calls
                        ],
                        "source_regions": answer.assets,
                        "image_hashes": assets,
                        "prompt_version": result.get(
                            "prompt_version", config.get("prompt_version")
                        ),
                    }
                )
    return output


def simulate_c(a: list[dict], b: list[dict], *, audit_rate: float, max_model: str) -> list[dict]:
    maximum = {(item["candidate_id"], item["question_id"]): item for item in a}
    if len(maximum) != len(a):
        raise ValueError("A 策略存在重复作答")
    result = []
    for primary in b:
        item = deepcopy(primary)
        item.update(simulated=True, escalated=False)
        if primary["status"] in {"scored", "resolved", "ungradable"}:
            upgrade = needs_verification(
                Transcript.model_validate(primary["transcript"]),
                Grade.model_validate(primary["grade"]),
                primary["answer_id"],
                audit_rate,
            )
            if upgrade:
                verified = maximum[(primary["candidate_id"], primary["question_id"])]
                if (
                    verified["replay_signature"] != primary["replay_signature"]
                    or verified["branch_model"] != max_model
                ):
                    raise ValueError("C 复用要求 Max 模型、图像、细则、提示词与参数一致")
                item = deepcopy(verified)
                item.update(simulated=True, escalated=True, audit_key=primary["answer_id"])
                item["cost_cny"] = (
                    str(Decimal(primary["cost_cny"]) + Decimal(verified["cost_cny"]))
                    if primary.get("cost_cny") is not None and verified.get("cost_cny") is not None
                    else None
                )
                item["latency_seconds"] = (
                    primary["latency_seconds"] + verified["latency_seconds"]
                    if primary.get("latency_seconds") is not None
                    and verified.get("latency_seconds") is not None
                    else None
                )
                item["flow_elapsed_seconds"] = (
                    primary["flow_elapsed_seconds"] + verified["flow_elapsed_seconds"]
                    if primary.get("flow_elapsed_seconds") is not None
                    and verified.get("flow_elapsed_seconds") is not None
                    else None
                )
        result.append(item)
    return result


def select_strategy(metrics: dict[str, dict]) -> dict:
    if not metrics:
        raise ValueError("没有候选策略")
    expected = {item["expected_count"] for item in metrics.values()}
    if len(expected) != 1:
        raise ValueError("候选必须使用相同评测样本")

    def rank(name):
        item = metrics[name]
        cost = item["cost"]
        return (
            -item["resolved_count"],
            Decimal(item["mae"]),
            item["major_error_count"],
            cost["unknown_count"],
            Decimal(cost["known_cny"]),
            name,
        )

    eligible = [name for name, item in metrics.items() if item["resolved_count"] > 0]
    selected = min(eligible, key=rank) if eligible else None
    return {
        "strategy": selected,
        "scope": "provisional_same_exam" if selected else "insufficient_resolved_results",
        "rule": "resolved_count desc; MAE asc; major errors asc; unknown costs asc; cost asc",
        "excluded": {name: "zero resolved results" for name in metrics if name not in eligible},
        "metrics": metrics[selected] if selected else None,
    }


def write_evaluation_report(
    db: Database,
    source_job_id: str,
    max_job_id: str,
    labels_path: Path,
    output: Path,
    *,
    split: str | None = None,
) -> dict:
    """Load isolated labels only after the real runs, and save reproducible artifacts."""
    from ai4edu.evaluation import compare_strategies, evaluate_predictions

    a = collect_predictions(db, max_job_id)
    b = collect_predictions(db, source_job_id, primary=True)
    c = collect_predictions(db, source_job_id)
    a_inputs = {(item["candidate_id"], item["question_id"]): item["replay_signature"] for item in a}
    b_inputs = {(item["candidate_id"], item["question_id"]): item["replay_signature"] for item in b}
    if a_inputs != b_inputs:
        raise ValueError("A/B/C 的图像、细则、提示词与参数必须一致")
    terminal = {"scored", "resolved", "ungradable", "failed", "budget_exceeded"}
    if any(item["status"] not in terminal for item in a + b + c):
        raise ValueError("评测批次仍有未完成作答")
    selected = {item["candidate_id"] for item in b}
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    if (
        split is not None
        and {item["candidate_id"] for item in labels if item["split"] == split} != selected
    ):
        raise ValueError("源批次与声明的数据分组不一致")
    labels = [item for item in labels if item["candidate_id"] in selected]
    report = compare_strategies({"A": a, "B": b, "C": c}, labels)
    with db.session() as session:
        source = session.get(JobRecord, source_job_id)
        maximum = session.get(JobRecord, max_job_id)
        if source.exam_id != maximum.exam_id or {item["candidate_id"] for item in a} != selected:
            raise ValueError("A/B/C 必须使用相同试卷和考生")
        config = source.execution_config
        if (
            config["verification_policy"] != "risk"
            or config["transcribe_model"] != config["model"]
            or config["grade_model"] != config["model"]
        ):
            raise ValueError("实际 C 结果必须来自 risk 策略")
        if (
            maximum.execution_config["verification_policy"] != "off"
            or maximum.execution_config["transcribe_model"] != config["verify_model"]
            or maximum.execution_config["grade_model"] != config["verify_model"]
        ):
            raise ValueError("A 必须使用 C 的 Max 模型")
        simulated = simulate_c(
            a, b, audit_rate=config["audit_rate"], max_model=config["verify_model"]
        )
        calls = session.scalars(
            select(ModelCall)
            .where(ModelCall.exam_id == source.exam_id)
            .order_by(ModelCall.created_at)
        ).all()
        shared = [call for call in calls if call.job_id is None]
        layout = [call for call in calls if call.job_id == source_job_id and call.role == "layout"]

        def expense(items):
            return {
                "calls": len(items),
                "known_cny": str(
                    sum((call.cost for call in items if call.cost is not None), Decimal(0))
                ),
                "unknown_calls": sum(call.cost is None for call in items),
                "reserved_unknown_cny": str(
                    sum((call.reserved_cost for call in items if call.cost is None), Decimal(0))
                ),
            }

        report["run"] = {
            "created_at": datetime.now(UTC).isoformat(),
            "source_job_id": source_job_id,
            "max_job_id": max_job_id,
            "exam_id": source.exam_id,
            "split": split or "explicit_subset",
            "candidate_ids": sorted(selected),
            "source_settings": config,
            "max_settings": maximum.execution_config,
            "labels_used_only_offline": True,
        }
        report["cost_accounting"] = {
            "basis": "configured reference prices; provider bill may differ",
            "all_experiment_calls_including_debug": expense(calls),
            "shared_exam_preparation": expense(shared),
            "source_layout": expense(layout),
            "actual_C_batch": expense([call for call in calls if call.job_id == source_job_id]),
            "actual_A_batch": expense([call for call in calls if call.job_id == max_job_id]),
            "strategy_metrics_exclude_shared_preparation_and_layout": True,
        }

        def job_wall(record):
            finished = [
                call.finished_at for call in calls if call.job_id == record.id and call.finished_at
            ]
            return (max(finished) - record.created_at).total_seconds() if finished else None

        report["timing"] = {
            "actual_C_job_wall_seconds": job_wall(source),
            "actual_A_job_wall_seconds": job_wall(maximum),
            "wall_time_basis": "job creation through last model-call completion; includes shared queue contention, excludes client upload",
            "per_answer_flow_wall_in_prediction_records": True,
            "strategy_latency_metric_basis": "sum of model-call durations; parallel chunk overlap is reflected separately by flow_elapsed_seconds",
        }
    report["C_simulated"] = evaluate_predictions(simulated, labels)
    report["C_simulated"]["latency_basis"] = (
        "sum of sequential B and selected A model-call durations; excludes queue wait"
    )
    report["selection"] = select_strategy(report["strategies"])
    report["limitations"] = [
        "已有 CSV 是得分对照，不提供逐步骤真值。",
        "同一套试卷的初评结论，不能证明跨试卷泛化或生产吞吐。",
        "B 主评结果来自实际 C 的独立初评阶段；A 为相同裁图与细则下的独立 Max 两阶段流程。",
        "C_simulated 与实际 C 分列；模型重复请求可能产生不同结果。",
        "时延指标为每题模型调用耗时之和，队列等待与整批墙钟单独记录。",
    ]
    output.mkdir(parents=True, exist_ok=True)
    for name, predictions in {"A": a, "B": b, "C": c, "C_simulated": simulated}.items():
        (output / f"predictions_{name}.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in predictions
            )
        )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    rows = [
        "# 主观题模型初评报告",
        "",
        f"数据分组：{report['run']['split']}；考生 {len(selected)} 名。已有成绩只用于离线评测。",
        "",
        "| 策略 | 已判定 / 全部 | MAE | 完全一致 | ±1分 | ≥3分差 | 判题流程总费用/元 | 累计调用 P50/P95 秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, values in report["strategies"].items():
        latency = values["latency_seconds"]

        def number(value, precision=3):
            return f"{Decimal(str(value)):.{precision}f}" if value is not None else "—"

        def percentage(value):
            return f"{value * 100:.1f}%" if value is not None else "—"

        rows.append(
            f"| {name} | {values['resolved_count']}/{values['expected_count']} | {number(values['mae'])} | {percentage(values['exact_rate'])} | {percentage(values['within_1_rate'])} | {values['major_error_count']} | {number(values['cost']['known_cny'], 6)} | {number(latency['p50'])} / {number(latency['p95'])} |"
        )
    rows += [
        "",
        f"当前试卷暂选：**{report['selection']['strategy']}**。排序规则：覆盖数量、MAE、大分差数量、未知费用、费用。",
        "",
        "费用表只包含各题转写、判分及该策略复判；定位与共享细则费用单列于 report.json。实际实验支出与 C 模拟生产费用分别记录。",
        "",
        "## 适用范围",
        "",
        *[f"- {item}" for item in report["limitations"]],
    ]
    (output / "report.md").write_text("\n".join(rows) + "\n")
    return report
