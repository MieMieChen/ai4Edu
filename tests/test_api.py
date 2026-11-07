import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError

from ai4edu.api import create_app
from ai4edu.config import Settings
from ai4edu.db import AnswerRecord, CandidateRecord, ExamRecord, Outbox


@pytest.fixture
def client(tmp_path):
    app = create_app(
        Settings(
            _env_file=None,
            database_url=f"sqlite:///{tmp_path / 'test.db'}",
            storage_dir=tmp_path / "assets",
            model="test-model",
        )
    )
    app.state.db.create_all()
    with TestClient(app) as value:
        yield value


def upload(client):
    buffer = BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="PNG")
    response = client.post(
        "/v1/assets", files={"file": ("answer.png", buffer.getvalue(), "image/png")}
    )
    assert response.status_code == 201
    return response.json()[0]["id"]


def ready_exam(client, count=5):
    response = client.post(
        "/v1/exams",
        json={
            "title": "数学主观题",
            "questions": [
                {
                    "id": str(15 + i),
                    "stem": "证明命题成立",
                    "reference": "参考证明",
                    "max_score": 10,
                }
                for i in range(count)
            ],
        },
    )
    assert response.status_code == 201
    exam_id = response.json()["id"]
    with client.app.state.db.session() as session:
        exam = session.get(ExamRecord, exam_id)
        exam.state = "ready"
        session.commit()
    return exam_id


def test_upload_persists_real_image(client):
    asset_id = upload(client)
    response = client.get(f"/v1/assets/{asset_id}")
    assert response.status_code == 200
    assert Image.open(BytesIO(response.content)).size == (200, 100)


def test_thousand_candidates_create_five_thousand_answer_units(client):
    asset_id = upload(client)
    exam_id = ready_exam(client)
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "candidates": [
                {"candidate_id": f"c{i:04}", "kind": "paper", "pages": [asset_id]}
                for i in range(1000)
            ],
        },
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["total"] == 5000
    with client.app.state.db.session() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(AnswerRecord).where(AnswerRecord.job_id == job_id)
            )
            == 5000
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(CandidateRecord)
                .where(CandidateRecord.job_id == job_id)
            )
            == 1000
        )
        assert (
            session.scalar(
                select(func.count()).select_from(Outbox).where(Outbox.task == "prepare_candidate")
            )
            == 1000
        )
    progress = client.get(f"/v1/jobs/{job_id}").json()
    assert progress["completed"] == 0
    assert progress["total_score"] is None


def test_job_submission_is_idempotent(client):
    asset_id = upload(client)
    exam_id = ready_exam(client, 1)
    payload = {
        "exam_id": exam_id,
        "candidates": [
            {
                "candidate_id": "c1",
                "kind": "question",
                "answers": [{"question_id": "15", "assets": [asset_id]}],
            }
        ],
    }
    a = client.post("/v1/jobs", json=payload, headers={"Idempotency-Key": "batch-1"})
    b = client.post("/v1/jobs", json=payload, headers={"Idempotency-Key": "batch-1"})
    assert a.status_code == b.status_code == 202
    assert a.json()["id"] == b.json()["id"]
    payload["candidates"][0]["candidate_id"] = "different"
    assert (
        client.post("/v1/jobs", json=payload, headers={"Idempotency-Key": "batch-1"}).status_code
        == 409
    )


def test_unknown_question_is_rejected(client):
    asset_id = upload(client)
    exam_id = ready_exam(client, 1)
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "candidates": [
                {
                    "candidate_id": "c1",
                    "kind": "question",
                    "answers": [{"question_id": "999", "assets": [asset_id]}],
                }
            ],
        },
    )
    assert response.status_code == 422


def test_duplicate_candidate_is_rejected(client):
    asset_id = upload(client)
    exam_id = ready_exam(client, 1)
    entry = {"candidate_id": "c1", "kind": "paper", "pages": [asset_id]}
    assert (
        client.post("/v1/jobs", json={"exam_id": exam_id, "candidates": [entry, entry]}).status_code
        == 422
    )


def test_unprepared_exam_cannot_start_grading(client):
    asset_id = upload(client)
    response = client.post(
        "/v1/exams",
        json={
            "title": "test",
            "questions": [
                {"id": "15", "stem": "题干", "reference": "解答", "max_score": 10},
            ],
        },
    )
    assert (
        client.post(
            "/v1/jobs",
            json={
                "exam_id": response.json()["id"],
                "candidates": [
                    {"candidate_id": "c1", "kind": "paper", "pages": [asset_id]},
                ],
            },
        ).status_code
        == 409
    )


def test_database_enforces_foreign_keys(client):
    with client.app.state.db.session() as session:
        session.add(AnswerRecord(job_id="missing", candidate_record_id="missing", question_id="15"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_invalid_image_is_an_input_error(client):
    response = client.post(
        "/v1/assets", files={"file": ("broken.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 422


def test_simultaneous_idempotent_posts_return_the_same_job(client):
    asset_id = upload(client)
    exam_id = ready_exam(client, 1)
    payload = {
        "exam_id": exam_id,
        "candidates": [{"candidate_id": "c1", "kind": "paper", "pages": [asset_id]}],
    }
    barrier, lock = threading.Barrier(2), threading.Lock()
    remaining = [2]

    def synchronize(_, cursor, statement, parameters, context, executemany):
        if "jobs.idempotency_key =" not in statement or parameters != ("race",):
            return
        with lock:
            should_wait = remaining[0] > 0
            remaining[0] -= 1
        if should_wait:
            barrier.wait(timeout=10)

    event.listen(client.app.state.db.engine, "after_cursor_execute", synchronize)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    lambda _: client.post(
                        "/v1/jobs", json=payload, headers={"Idempotency-Key": "race"}
                    ),
                    range(2),
                )
            )
    finally:
        event.remove(client.app.state.db.engine, "after_cursor_execute", synchronize)
    assert [r.status_code for r in responses] == [202, 202]
    assert responses[0].json()["id"] == responses[1].json()["id"]


def test_unsupported_score_precision_fails_before_persistence(client):
    response = client.post(
        "/v1/exams",
        json={
            "title": "test",
            "score_step": "0.0001",
            "questions": [
                {"id": "15", "stem": "题目", "reference": "答案", "max_score": "1"},
            ],
        },
    )
    assert response.status_code == 422
