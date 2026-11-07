from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def uid() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class LeasedUnit:
    execution_token: Mapped[str | None] = mapped_column(String(32))


class StaleExecution(Exception):
    pass


def lock_unit(session, model, ident: str, token: str):
    record = session.scalar(
        select(model)
        .where(
            model.id == ident,
            model.execution_token == token,
            model.state == "running",
        )
        .with_for_update()
    )
    if record is None:
        raise StaleExecution("任务执行权已移交；丢弃过期执行的写入")
    return record


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    key: Mapped[str] = mapped_column(String(256))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(default=now)


class ExamRecord(LeasedUnit, Base):
    __tablename__ = "exams"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    title: Mapped[str] = mapped_column(String(200))
    questions: Mapped[list] = mapped_column(JSON)
    score_step: Mapped[Decimal] = mapped_column(Numeric(8, 3))
    budget_cny: Mapped[Decimal | None] = mapped_column(Numeric(16, 8))
    committed_cost: Mapped[Decimal] = mapped_column(Numeric(16, 8), default=0)
    rubrics: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(24), default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(default=now)


class JobRecord(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    exam_id: Mapped[str] = mapped_column(ForeignKey("exams.id"), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    execution_config: Mapped[dict] = mapped_column(JSON, default=dict)
    budget_cny: Mapped[Decimal | None] = mapped_column(Numeric(16, 8))
    committed_cost: Mapped[Decimal] = mapped_column(Numeric(16, 8), default=0)
    created_at: Mapped[datetime] = mapped_column(default=now)


class CandidateRecord(LeasedUnit, Base):
    __tablename__ = "candidates"
    __table_args__ = (UniqueConstraint("job_id", "candidate_id"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    candidate_id: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict] = mapped_column(JSON)
    prepared_pages: Mapped[dict] = mapped_column(JSON, default=dict)
    error_kind: Mapped[str | None] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(24), default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime)


class AnswerRecord(LeasedUnit, Base):
    __tablename__ = "answers"
    __table_args__ = (
        UniqueConstraint("candidate_record_id", "question_id"),
        Index("ix_answers_job_state", "job_id", "state"),
    )
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    candidate_record_id: Mapped[str] = mapped_column(ForeignKey("candidates.id"))
    question_id: Mapped[str] = mapped_column(String(64))
    assets: Mapped[list] = mapped_column(JSON, default=list)
    intermediate: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON)
    state: Mapped[str] = mapped_column(String(24), default="waiting")
    error: Mapped[str | None] = mapped_column(Text)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime)


class ModelCall(Base):
    __tablename__ = "model_calls"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    exam_id: Mapped[str] = mapped_column(ForeignKey("exams.id"), index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), index=True)
    unit_id: Mapped[str] = mapped_column(String(32), index=True)
    role: Mapped[str] = mapped_column(String(24))
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(128))
    state: Mapped[str] = mapped_column(String(24), default="running")
    reserved_cost: Mapped[Decimal] = mapped_column(Numeric(16, 8), default=0)
    cost: Mapped[Decimal | None] = mapped_column(Numeric(16, 8))
    usage: Mapped[dict | None] = mapped_column(JSON)
    response: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class Outbox(Base):
    __tablename__ = "outbox"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    task: Mapped[str] = mapped_column(String(32))
    unit_id: Mapped[str] = mapped_column(String(32))
    available_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    created_at: Mapped[datetime] = mapped_column(default=now)


class Database:
    def __init__(self, url: str):
        options = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=options, pool_pre_ping=True)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def sqlite_foreign_keys(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")

        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self):
        Base.metadata.create_all(self.engine)
        # Explicit init-db also performs the additive v0.1 -> execution snapshot upgrade.
        columns = {column["name"] for column in inspect(self.engine).get_columns("jobs")}
        if "execution_config" not in columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE jobs ADD COLUMN execution_config JSON NOT NULL DEFAULT '{}'")
                )
        columns = {column["name"] for column in inspect(self.engine).get_columns("exams")}
        with self.engine.begin() as connection:
            if "budget_cny" not in columns:
                connection.execute(text("ALTER TABLE exams ADD COLUMN budget_cny NUMERIC(16,8)"))
            if "committed_cost" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE exams ADD COLUMN committed_cost NUMERIC(16,8) NOT NULL DEFAULT 0"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE exams SET committed_cost = COALESCE((SELECT SUM(COALESCE(model_calls.cost, model_calls.reserved_cost)) FROM model_calls WHERE model_calls.exam_id = exams.id), 0)"
                    )
                )
