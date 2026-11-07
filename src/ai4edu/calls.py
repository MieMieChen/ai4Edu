import hashlib
import json
import math
from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import select, update

from ai4edu import prompts
from ai4edu.bailian import BailianClient, ModelImage
from ai4edu.config import Settings
from ai4edu.db import (
    AnswerRecord,
    CandidateRecord,
    Database,
    ExamRecord,
    JobRecord,
    ModelCall,
    lock_unit,
    now,
    uid,
)
from ai4edu.limits import RedisLimiter

UNIT_TYPES = {
    "prepare_exam": ExamRecord,
    "prepare_candidate": CandidateRecord,
    "grade_answer": AnswerRecord,
}


class BudgetExceeded(Exception):
    pass


class Calls:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        client: BailianClient,
        limiter: RedisLimiter,
        *,
        exam_id: str,
        unit_id: str,
        task: str,
        execution_token: str,
        job_id: str | None = None,
    ):
        self.db, self.settings, self.client, self.limiter = db, settings, client, limiter
        self.exam_id, self.unit_id, self.task, self.job_id = exam_id, unit_id, task, job_id
        self.execution_token = execution_token

    def complete(
        self,
        role: str,
        schema: type[BaseModel],
        prompt: str,
        images: list[ModelImage],
        *,
        validate: Callable[[BaseModel], object] | None = None,
    ):
        model = self.settings.model_for(role)
        effective_output_limit = self.settings.output_limit_for(schema.__name__)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "role": role,
                    "model": model,
                    "prompt": prompt,
                    "schema": schema.model_json_schema(),
                    "images": [
                        {"id": image.id, "sha256": hashlib.sha256(image.data).hexdigest()}
                        for image in images
                    ],
                    "output_limit": effective_output_limit,
                    "prompt_version": prompts.VERSION,
                    "enable_thinking": role in self.settings.thinking_roles,
                    "thinking_budget": self.settings.thinking_budget
                    if role in self.settings.thinking_roles
                    else None,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with self.db.session() as session:
            cached = session.scalar(
                select(ModelCall)
                .where(
                    ModelCall.unit_id == self.unit_id,
                    ModelCall.fingerprint == fingerprint,
                    ModelCall.state == "succeeded",
                )
                .order_by(ModelCall.created_at.desc())
            )
            if cached is not None:
                value = schema.model_validate_json(
                    cached.response["choices"][0]["message"]["content"]
                )
                if validate:
                    validate(value)
                return value

        # UTF-8 byte count is conservative for text; visual reservation includes a resize margin.
        input_estimate = len(prompt.encode()) + len(json.dumps(schema.model_json_schema()).encode())
        input_estimate += sum(
            2 * math.ceil(max(image.width * image.height, 65536) / 1024) + 2 for image in images
        )
        if (
            self.settings.max_input_tokens is not None
            and input_estimate > self.settings.max_input_tokens
        ):
            raise ValueError(
                f"输入 Token 保守估算为 {input_estimate}，超过配置上限 {self.settings.max_input_tokens}；"
                "请求未发送，请减少单次输入或调整输入上限"
            )
        output_limit = effective_output_limit + (
            self.settings.thinking_budget if role in self.settings.thinking_roles else 0
        )
        token_reservation = input_estimate + output_limit
        ticket = self.limiter.acquire(model, token_reservation)
        price = self.settings.model_prices.get(model)
        reserved = (
            (input_estimate * price.input + output_limit * price.output) / 1000000
            if price
            else Decimal(0)
        )
        ident, actual_tokens = uid(), None
        raw, usage, cost, error = None, None, None, None
        started = False
        try:
            with self.db.session() as session:
                unit = lock_unit(session, UNIT_TYPES[self.task], self.unit_id, self.execution_token)
                if price is None and session.get(ExamRecord, self.exam_id).budget_cny is not None:
                    raise ValueError("试卷启用了总预算，必须配置当前模型单价")
                experiment = session.execute(
                    update(ExamRecord)
                    .where(
                        ExamRecord.id == self.exam_id,
                        (ExamRecord.budget_cny.is_(None))
                        | (ExamRecord.committed_cost + reserved <= ExamRecord.budget_cny),
                    )
                    .values(committed_cost=ExamRecord.committed_cost + reserved)
                )
                if experiment.rowcount != 1:
                    raise BudgetExceeded("试卷累计费用预留超过总预算")
                if self.job_id is not None:
                    reservation = update(JobRecord).where(JobRecord.id == self.job_id)
                    reservation = reservation.where(
                        (JobRecord.budget_cny.is_(None))
                        | (JobRecord.committed_cost + reserved <= JobRecord.budget_cny)
                    )
                    result = session.execute(
                        reservation.values(committed_cost=JobRecord.committed_cost + reserved)
                    )
                    if result.rowcount != 1:
                        raise BudgetExceeded("批次费用预留超过预算")
                session.add(
                    ModelCall(
                        id=ident,
                        exam_id=self.exam_id,
                        job_id=self.job_id,
                        unit_id=self.unit_id,
                        role=role,
                        model=model,
                        fingerprint=fingerprint,
                        reserved_cost=reserved,
                    )
                )
                unit.lease_until = now() + timedelta(seconds=self.settings.task_lease_seconds)
                session.commit()
                started = True
            raw = self.client.request(role, schema, prompt, images)
            usage = raw["usage"]
            actual_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
            if price:
                cost = (
                    usage["prompt_tokens"] * price.input + usage["completion_tokens"] * price.output
                ) / 1000000
            if raw["choices"][0].get("finish_reason") == "length":
                raise ValueError("模型输出达到长度上限，不能接受不完整的转写或评分")
            value = schema.model_validate_json(raw["choices"][0]["message"]["content"])
            if validate:
                validate(value)
            return value
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if started:
                with self.db.session() as session:
                    call = session.get(ModelCall, ident)
                    call.response, call.usage, call.cost = raw, usage, cost
                    call.error, call.finished_at = error, now()
                    call.state = "failed" if error else "succeeded"
                    if cost is not None:
                        session.execute(
                            update(ExamRecord)
                            .where(ExamRecord.id == self.exam_id)
                            .values(committed_cost=ExamRecord.committed_cost + cost - reserved)
                        )
                    if self.job_id is not None and cost is not None:
                        session.execute(
                            update(JobRecord)
                            .where(JobRecord.id == self.job_id)
                            .values(committed_cost=JobRecord.committed_cost + cost - reserved)
                        )
                    session.commit()
            else:
                actual_tokens = 0
            self.limiter.release(model, ticket, actual_tokens)
