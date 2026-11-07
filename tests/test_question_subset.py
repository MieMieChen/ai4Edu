import hashlib
import json

import pytest
from sqlalchemy import select
from test_api import client as original_client
from test_api import ready_exam, upload
from test_pipeline import drain
from test_pipeline import runtime as original_runtime

from ai4edu.db import AnswerRecord, ExamRecord, JobRecord
from ai4edu.schemas import PageLayout

client = original_client
runtime = original_runtime


def test_one_selected_question_reuses_exam_and_creates_one_answer(client):
    asset = upload(client)
    exam = ready_exam(client)
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "question_ids": ["16"],
            "candidates": [
                {
                    "candidate_id": "live",
                    "kind": "question",
                    "answers": [{"question_id": "16", "assets": [asset]}],
                }
            ],
        },
    )
    assert response.status_code == 202
    assert response.json()["total"] == 1
    assert response.json()["exam_id"] == exam
    with client.app.state.db.session() as session:
        answers = session.scalars(
            select(AnswerRecord).where(AnswerRecord.job_id == response.json()["id"])
        ).all()
        assert [a.question_id for a in answers] == ["16"]
        assert len(session.get(ExamRecord, exam).questions) == 5


@pytest.mark.parametrize("selected", [[], ["999"], ["15", "15"]])
def test_invalid_question_subset_is_rejected(client, selected):
    asset = upload(client)
    exam = ready_exam(client)
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "question_ids": selected,
            "candidates": [{"candidate_id": "live", "kind": "paper", "pages": [asset]}],
        },
    )
    assert response.status_code == 422


def test_selected_questions_must_be_fully_supplied(client):
    asset = upload(client)
    exam = ready_exam(client)
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "question_ids": ["15", "16"],
            "candidates": [
                {
                    "candidate_id": "live",
                    "kind": "question",
                    "answers": [{"question_id": "15", "assets": [asset]}],
                }
            ],
        },
    )
    assert response.status_code == 422


def test_legacy_full_exam_submission_keeps_its_idempotency_hash(client):
    asset = upload(client)
    exam = ready_exam(client)
    payload = {
        "exam_id": exam,
        "candidates": [{"candidate_id": "legacy", "kind": "paper", "pages": [asset]}],
        "budget_cny": None,
        "strategy": None,
    }
    headers = {"Idempotency-Key": "legacy-full-exam"}
    first = client.post("/v1/jobs", json=payload, headers=headers).json()
    legacy_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with client.app.state.db.session() as session:
        assert session.get(JobRecord, first["id"]).fingerprint == legacy_hash
    assert first["total"] == 5
    assert client.post("/v1/jobs", json=payload, headers=headers).json()["id"] == first["id"]
    assert (
        client.post(
            "/v1/jobs", json={**payload, "question_ids": ["15"]}, headers=headers
        ).status_code
        == 409
    )


def test_selected_paper_question_flows_through_layout_and_scoring(runtime, monkeypatch):
    client, settings, pipeline, provider, asset, exam = runtime
    settings.verification_policy = "off"
    drain(client.app.state.db, settings, pipeline)
    with client.app.state.db.session() as session:
        row = session.get(ExamRecord, exam)
        row.questions = [*row.questions, {**row.questions[0], "id": "16"}]
        row.rubrics = {**row.rubrics, "16": {**row.rubrics["15"], "question_id": "16"}}
        session.commit()
    requested_ids = []
    original_request = provider.request

    def capture(role, schema, prompt, images):
        if schema is PageLayout:
            requested_ids.append(json.loads(prompt.split("指定题号：", 1)[1]))
        return original_request(role, schema, prompt, images)

    monkeypatch.setattr(provider, "request", capture)
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "question_ids": ["15"],
            "candidates": [{"candidate_id": "paper", "kind": "paper", "pages": [asset]}],
        },
    )
    assert response.status_code == 202
    job = response.json()["id"]
    drain(client.app.state.db, settings, pipeline)
    result = client.get(f"/v1/jobs/{job}/results").json()[0]
    assert requested_ids == [["15"]]
    assert len(result["answers"]) == 1
    assert result["answers"][0]["question_id"] == "15"
    assert result["answers"][0]["state"] == "scored"
