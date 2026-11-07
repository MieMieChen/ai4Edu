import logging
from datetime import timedelta
from functools import lru_cache

from celery import Celery
from redis import Redis
from sqlalchemy import select, update

from ai4edu.assets import AssetStore
from ai4edu.bailian import BailianClient
from ai4edu.calls import UNIT_TYPES
from ai4edu.config import Settings, get_settings
from ai4edu.db import AnswerRecord, Database, JobRecord, Outbox, StaleExecution, now, uid
from ai4edu.execution import execution_settings
from ai4edu.limits import RateLimitWait, RedisLimiter
from ai4edu.pipeline import Pipeline
from ai4edu.storage import make_storage

logger = logging.getLogger(__name__)


def dispatch_outbox(db: Database, publish, limit: int = 100) -> int:
    with db.session() as session:
        entries = session.scalars(
            select(Outbox)
            .where(Outbox.available_at <= now())
            .order_by(Outbox.available_at, Outbox.created_at, Outbox.id)
            .with_for_update(skip_locked=True)
            .limit(limit)
        ).all()
        for entry in entries:
            publish(entry.task, entry.unit_id)
            session.delete(entry)
        session.commit()
        return len(entries)


def recover_expired(db: Database) -> int:
    recovered = 0
    with db.session() as session:
        for task, model in UNIT_TYPES.items():
            entries = session.scalars(
                select(model)
                .where(
                    model.state == "running",
                    model.lease_until < now(),
                )
                .with_for_update(skip_locked=True)
            ).all()
            for entry in entries:
                entry.state, entry.lease_until, entry.execution_token = "queued", None, None
                session.add(Outbox(task=task, unit_id=entry.id))
                recovered += 1
        session.commit()
    return recovered


def execute_unit(db: Database, settings: Settings, pipeline: Pipeline, task: str, unit_id: str):
    model = UNIT_TYPES[task]
    token = uid()
    with db.session() as session:
        claimed = session.execute(
            update(model)
            .where(model.id == unit_id, model.state == "queued")
            .values(
                state="running",
                error=None,
                execution_token=token,
                lease_until=now() + timedelta(seconds=settings.task_lease_seconds),
            )
        )
        session.commit()
        if claimed.rowcount != 1:
            return
    try:
        getattr(pipeline, task)(unit_id, token)
    except StaleExecution:
        logger.info("Discarded stale execution %s/%s", task, unit_id)
    except RateLimitWait as exc:
        with db.session() as session:
            changed = session.execute(
                update(model)
                .where(
                    model.id == unit_id,
                    model.execution_token == token,
                    model.state == "running",
                )
                .values(state="queued", lease_until=None, execution_token=None)
            )
            if changed.rowcount:
                session.add(
                    Outbox(
                        task=task,
                        unit_id=unit_id,
                        available_at=now() + timedelta(seconds=exc.seconds),
                    )
                )
            session.commit()
    except Exception as exc:
        # One failure boundary per task. Do not manufacture a score or switch models.
        with db.session() as session:
            failure = {
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "lease_until": None,
                "execution_token": None,
            }
            if task == "prepare_candidate":
                failure["error_kind"] = "execution"
            changed = session.execute(
                update(model)
                .where(
                    model.id == unit_id,
                    model.execution_token == token,
                    model.state == "running",
                )
                .values(**failure)
            )
            if changed.rowcount and task == "prepare_candidate":
                session.execute(
                    update(AnswerRecord)
                    .where(
                        AnswerRecord.candidate_record_id == unit_id,
                        AnswerRecord.state.in_(["waiting", "queued"]),
                    )
                    .values(state="failed", error=f"输入处理失败：{type(exc).__name__}: {exc}")
                )
            session.commit()
        if changed.rowcount:
            raise
        logger.info("Discarded error from stale execution %s/%s", task, unit_id)


settings = get_settings()
celery = Celery("ai4edu", broker=settings.redis_url)
celery.conf.update(
    task_default_queue="grading",
    task_ignore_result=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": settings.task_lease_seconds + 60},
    broker_connection_retry_on_startup=True,
    beat_schedule={
        "dispatch-pending-tasks": {
            "task": "ai4edu.dispatch",
            "schedule": 1,
            "options": {"queue": "control"},
        }
    },
)


@lru_cache
def runtime() -> tuple[Database, Pipeline]:
    config = get_settings()
    db = Database(config.database_url)
    assets = AssetStore(db, make_storage(config))
    limiter = RedisLimiter(
        Redis.from_url(config.redis_url),
        concurrency=config.model_concurrency,
        rpm=config.model_rpm,
        tpm=config.model_tpm,
        lease_seconds=config.task_lease_seconds,
    )
    return db, Pipeline(db, config, assets, BailianClient(config), limiter)


def publish(task: str, unit_id: str):
    queue = "preparation" if task == "prepare_candidate" else "grading"
    celery.send_task(f"ai4edu.{task}", args=[unit_id], queue=queue, retry=False)


@celery.task(name="ai4edu.dispatch")
def dispatch():
    db, _ = runtime()
    recover_expired(db)
    return dispatch_outbox(db, publish)


def run_celery_unit(task, name: str, unit_id: str):
    db, pipeline = runtime()
    config = get_settings()
    if name == "prepare_exam":
        return execute_unit(db, config, pipeline, name, unit_id)
    with db.session() as session:
        unit = session.get(UNIT_TYPES[name], unit_id)
        if unit is None:
            return
        job = session.get(JobRecord, unit.job_id)
        config = execution_settings(config, job.execution_config)
    client = BailianClient(config)
    try:
        configured = Pipeline(db, config, pipeline.assets, client, pipeline.limiter)
        return execute_unit(db, config, configured, name, unit_id)
    finally:
        client.close()


@celery.task(bind=True, name="ai4edu.prepare_exam")
def prepare_exam(task, unit_id: str):
    return run_celery_unit(task, "prepare_exam", unit_id)


@celery.task(bind=True, name="ai4edu.prepare_candidate")
def prepare_candidate(task, unit_id: str):
    return run_celery_unit(task, "prepare_candidate", unit_id)


@celery.task(bind=True, name="ai4edu.grade_answer")
def grade_answer(task, unit_id: str):
    return run_celery_unit(task, "grade_answer", unit_id)
