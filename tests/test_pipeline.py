import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from io import BytesIO

import fakeredis
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import func, select

from ai4edu.api import create_app
from ai4edu.config import Settings
from ai4edu.db import AnswerRecord, ExamRecord, Outbox, now
from ai4edu.limits import RateLimitWait, RedisLimiter
from ai4edu.pipeline import Pipeline
from ai4edu.schemas import Judgment, PageLayout, Rubric, Transcript
from ai4edu.tasks import dispatch_outbox, execute_unit, recover_expired


class ModelService:
    """External model test double; never used by application runtime."""

    def __init__(self):
        self.calls = []
        self.invalid_score = False
        self.missing_regions = False
        self.empty_transcriptions = 0
        self.layout_failure_image = None
        self.truncate_transcript = False

    def request(self, role, schema, prompt, images):
        self.calls.append(role)
        if schema is Rubric:
            value = {
                "question_id": "15",
                "max_score": 4,
                "solution": "证明",
                "criteria": [
                    {
                        "id": "R1",
                        "description": "建立关系",
                        "max_score": 2,
                        "rules": "允许等价解法",
                    },
                    {
                        "id": "R2",
                        "description": "完成证明",
                        "max_score": 2,
                        "rules": "允许部分给分",
                    },
                ],
            }
        elif schema is PageLayout:
            if images[0].id == self.layout_failure_image:
                raise ConnectionError("page2 temporarily unavailable")
            value = {
                "rotation": 0,
                "regions": []
                if self.missing_regions
                else [{"question_id": "15", "bbox": [0, 0, 1, 1], "order": 0}],
            }
        elif schema is Transcript:
            if not images:
                self.empty_transcriptions += 1
                value = {"lines": [], "is_blank": True}
            else:
                value = {
                    "lines": [{"id": "L1", "image_id": images[0].id, "text": "BD垂直于平面PEC"}]
                }
        elif schema is Judgment:
            value = {
                "scores": [
                    {
                        "criterion_id": "R1",
                        "score": 99 if self.invalid_score else 2,
                        "reason": "成立",
                        "evidence_ids": ["L1"],
                    },
                    {
                        "criterion_id": "R2",
                        "score": 1,
                        "reason": "部分论证",
                        "evidence_ids": ["L1"],
                    },
                ]
            }
        else:
            raise AssertionError(schema)
        return {
            "id": f"call-{len(self.calls)}",
            "choices": [
                {
                    "message": {"content": json.dumps(value)},
                    "finish_reason": "length"
                    if schema is Transcript and self.truncate_transcript
                    else "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }


@pytest.fixture
def runtime(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'pipeline.db'}",
        storage_dir=tmp_path / "assets",
        model="test",
        model_rpm=1000,
        model_tpm=1000000,
        audit_rate=0,
        model_prices={"test": {"input": 2, "output": 8}},
    )
    app = create_app(settings)
    app.state.db.create_all()
    provider = ModelService()
    limiter = RedisLimiter(
        fakeredis.FakeRedis(), concurrency=8, rpm=1000, tpm=1000000, lease_seconds=900
    )
    pipeline = Pipeline(app.state.db, settings, app.state.assets, provider, limiter)
    with TestClient(app) as client:
        image = Image.new("RGB", (240, 160), "white")
        ImageDraw.Draw(image).text((20, 20), "x + 1 = 2", fill="black")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        asset_id = client.post(
            "/v1/assets", files={"file": ("input.png", buffer.getvalue(), "image/png")}
        ).json()[0]["id"]
        exam_id = client.post(
            "/v1/exams",
            json={
                "title": "test",
                "questions": [
                    {"id": "15", "stem": "证明", "reference": "参考证明", "max_score": 4},
                ],
            },
        ).json()["id"]
        yield client, settings, pipeline, provider, asset_id, exam_id


def drain(db, settings, pipeline):
    pending = []
    while True:
        dispatch_outbox(db, lambda task, unit_id: pending.append((task, unit_id)))
        if not pending:
            return
        task, unit_id = pending.pop(0)
        execute_unit(db, settings, pipeline, task, unit_id)


def submit(runtime, **extra):
    client, _, _, _, asset_id, exam_id = runtime
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "candidates": [
                {"candidate_id": "anonymous-1", "kind": "paper", "pages": [asset_id]},
            ],
            **extra,
        },
    )
    assert response.status_code == 202
    return response.json()["id"]


def test_complete_automatic_pipeline_records_score_and_cost(runtime):
    client, settings, pipeline, provider, _, exam_id = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    assert client.get(f"/v1/exams/{exam_id}").json()["state"] == "ready"
    job_id = submit(runtime)
    drain(db, settings, pipeline)
    progress = client.get(f"/v1/jobs/{job_id}").json()
    assert progress["state"] == "completed"
    assert progress["counts"] == {"scored": 1}
    assert progress["known_cost_cny"] > 0
    result = client.get(f"/v1/jobs/{job_id}/results").json()[0]
    assert result["total_score"] == 3
    assert result["answers"][0]["source_regions"][0]["source_asset_id"]
    call_count = len(provider.calls)
    with db.session() as session:
        answer_id = session.scalar(select(AnswerRecord.id).where(AnswerRecord.job_id == job_id))
    execute_unit(db, settings, pipeline, "grade_answer", answer_id)
    assert len(provider.calls) == call_count


def test_rubric_draft_with_wrong_total_reaches_independent_check(runtime):
    client, settings, pipeline, provider, _, exam_id = runtime
    original = provider.request
    seen = []

    def request(role, schema, prompt, images):
        raw = original(role, schema, prompt, images)
        if schema is Rubric:
            seen.append(prompt)
            if len(seen) == 1:
                value = json.loads(raw["choices"][0]["message"]["content"])
                value["criteria"][0]["max_score"] = 3
                raw["choices"][0]["message"]["content"] = json.dumps(value)
        return raw

    provider.request = request
    drain(client.app.state.db, settings, pipeline)
    exam = client.get(f"/v1/exams/{exam_id}").json()
    assert exam["state"] == "ready"
    assert len(seen) == 2
    assert "5" in seen[1] and "4" in seen[1]


def test_old_prompt_version_does_not_reuse_prepared_or_scoring_checkpoints(runtime):
    from ai4edu.db import JobRecord

    client, settings, pipeline, provider, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    job_id = submit(runtime)
    drain(db, settings, pipeline)
    previous = len(provider.calls)
    with db.session() as session:
        job = session.get(JobRecord, job_id)
        job.execution_config = job.execution_config | {"prompt_version": "obsolete"}
        answer = session.scalar(select(AnswerRecord).where(AnswerRecord.job_id == job_id))
        answer.state, answer.result = "queued", None
        answer_id = answer.id
        session.commit()
    with pytest.raises(ValueError, match="提示词版本"):
        execute_unit(db, settings, pipeline, "grade_answer", answer_id)
    assert len(provider.calls) == previous


def test_invalid_model_points_fail_without_zero_score(runtime):
    client, settings, pipeline, provider, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    provider.invalid_score = True
    job_id = submit(runtime)
    with pytest.raises(ValueError, match="分值"):
        drain(db, settings, pipeline)
    result = client.get(f"/v1/jobs/{job_id}/results").json()[0]
    assert result["total_score"] is None
    assert result["answers"][0]["state"] == "failed"
    assert result["answers"][0]["result"] is None
    provider.invalid_score = False
    previous = provider.calls.count("transcribe")
    client.post(f"/v1/jobs/{job_id}/retry")
    drain(db, settings, pipeline)
    assert provider.calls.count("transcribe") == previous
    assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "completed"


def test_budget_denial_stops_before_paid_request(runtime):
    client, settings, pipeline, provider, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    calls = len(provider.calls)
    job_id = submit(runtime, budget_cny="0.000001")
    with pytest.raises(Exception, match="预算"):
        drain(db, settings, pipeline)
    assert len(provider.calls) == calls
    assert client.get(f"/v1/jobs/{job_id}").json()["counts"] == {"failed": 1}


def test_verified_scoring_is_independent_of_primary_score(runtime):
    client, settings, pipeline, provider, _, _ = runtime
    pipeline.settings.verification_policy = "all"
    drain(client.app.state.db, settings, pipeline)
    job_id = submit(runtime)
    drain(client.app.state.db, settings, pipeline)
    assert provider.calls.count("verify") == 2
    assert (
        client.get(f"/v1/jobs/{job_id}/results").json()[0]["answers"][0]["result"]["verified"]
        is True
    )


def test_baseline_policy_off_does_not_add_verification_calls(runtime):
    client, settings, pipeline, provider, _, _ = runtime
    pipeline.settings.verification_policy = "off"
    pipeline.settings.audit_rate = 1
    drain(client.app.state.db, settings, pipeline)
    job_id = submit(runtime)
    drain(client.app.state.db, settings, pipeline)
    assert provider.calls.count("verify") == 0
    assert (
        client.get(f"/v1/jobs/{job_id}/results").json()[0]["answers"][0]["result"]["verified"]
        is False
    )


def test_outbox_does_not_drop_task_when_publication_fails(runtime):
    client, _, _, _, _, _ = runtime
    db = client.app.state.db

    def fail(*_):
        raise ConnectionError("broker unavailable")

    with pytest.raises(ConnectionError):
        dispatch_outbox(db, fail)
    with db.session() as session:
        assert session.scalar(select(func.count()).select_from(Outbox)) == 1


def test_expired_lease_can_be_recovered(runtime):
    client, _, _, _, _, exam_id = runtime
    db = client.app.state.db
    with db.session() as session:
        for entry in session.scalars(select(Outbox)):
            session.delete(entry)
        exam = session.get(ExamRecord, exam_id)
        exam.state, exam.lease_until = "running", now() - timedelta(seconds=1)
        session.commit()
    assert recover_expired(db) == 1
    pending = []
    dispatch_outbox(db, lambda task, unit_id: pending.append((task, unit_id)))
    assert pending == [("prepare_exam", exam_id)]


def test_missing_region_retry_repeats_layout_not_empty_image_scoring(runtime):
    client, settings, pipeline, provider, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    provider.missing_regions = True
    job_id = submit(runtime)
    drain(db, settings, pipeline)
    assert client.get(f"/v1/jobs/{job_id}").json()["counts"] == {"failed": 1}
    provider.missing_regions = False
    client.post(f"/v1/jobs/{job_id}/retry")
    drain(db, settings, pipeline)
    assert provider.empty_transcriptions == 0
    assert provider.calls.count("layout") == 2
    assert client.get(f"/v1/jobs/{job_id}/results").json()[0]["total_score"] == 3


def test_capacity_wait_is_durable_even_after_many_reschedules(runtime):
    client, settings, pipeline, provider, _, exam_id = runtime
    db = client.app.state.db
    original = pipeline.limiter

    class Busy:
        def acquire(self, *_):
            raise RateLimitWait(5)

    pipeline.limiter = Busy()
    for _ in range(6):
        pending = []
        dispatch_outbox(db, lambda task, unit, pending=pending: pending.append((task, unit)))
        assert pending == [("prepare_exam", exam_id)]
        execute_unit(db, settings, pipeline, *pending[0])
        with db.session() as session:
            exam = session.get(ExamRecord, exam_id)
            assert exam.state == "queued" and exam.lease_until is None
            entry = session.scalar(select(Outbox))
            assert entry is not None and entry.available_at > now()
            entry.available_at = now() - timedelta(seconds=1)
            session.commit()
    assert provider.calls == []
    pipeline.limiter = original
    drain(db, settings, pipeline)
    assert client.get(f"/v1/exams/{exam_id}").json()["state"] == "ready"


def test_preparation_retry_reuses_successful_page_checkpoint(runtime):
    client, settings, pipeline, provider, asset_id, exam_id = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    second = BytesIO()
    Image.new("RGB", (200, 120), "black").save(second, format="PNG")
    second_id = client.post(
        "/v1/assets", files={"file": ("page2.png", second.getvalue(), "image/png")}
    ).json()[0]["id"]
    provider.layout_failure_image = second_id
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "candidates": [
                {"candidate_id": "c1", "kind": "paper", "pages": [asset_id, second_id]},
            ],
        },
    )
    job_id = response.json()["id"]
    with pytest.raises(ConnectionError):
        drain(db, settings, pipeline)
    assert provider.calls.count("layout") == 2
    provider.layout_failure_image = None
    client.post(f"/v1/jobs/{job_id}/retry")
    drain(db, settings, pipeline)
    assert provider.calls.count("layout") == 3
    assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "completed"


def test_truncated_but_valid_json_is_not_accepted_as_a_complete_transcript(runtime):
    client, settings, pipeline, provider, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    provider.truncate_transcript = True
    job_id = submit(runtime)
    with pytest.raises(ValueError, match="长度"):
        drain(db, settings, pipeline)
    assert client.get(f"/v1/jobs/{job_id}").json()["counts"] == {"failed": 1}


def test_late_worker_failure_cannot_overwrite_replacement_result(runtime):
    client, settings, _, _, _, exam_id = runtime
    db = client.app.state.db
    started, release = threading.Event(), threading.Event()

    class RacingPipeline:
        def prepare_exam(self, unit_id, execution_token=None):
            if not started.is_set():
                started.set()
                assert release.wait(10)
                raise TimeoutError("old worker timed out")
            with db.session() as session:
                session.get(ExamRecord, unit_id).state = "ready"
                session.commit()

    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(execute_unit, db, settings, RacingPipeline(), "prepare_exam", exam_id)
        assert started.wait(10)
        try:
            with db.session() as session:
                session.get(ExamRecord, exam_id).lease_until = now() - timedelta(seconds=1)
                session.commit()
            recover_expired(db)
            execute_unit(db, settings, RacingPipeline(), "prepare_exam", exam_id)
        finally:
            release.set()
        try:
            old.result()
        except TimeoutError:
            pass
    assert client.get(f"/v1/exams/{exam_id}").json()["state"] == "ready"
