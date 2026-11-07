import hashlib
import json
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image
from sqlalchemy import select, text
from test_api import client as original_client
from test_api import ready_exam, upload
from test_pipeline import drain
from test_pipeline import runtime as original_runtime

from ai4edu.db import AnswerRecord, ExamRecord, JobRecord
from ai4edu.schemas import Judgment, Transcript
from ai4edu.tasks import execute_unit

client = original_client
runtime = original_runtime


def template():
    return {
        "pages": [
            {
                "width": 240,
                "height": 160,
                "boxes": [
                    {
                        "id": "left",
                        "question_id": "15",
                        "subquestion_id": "1",
                        "bbox": [0, 0, 0.5, 1],
                        "order": 0,
                    },
                    {
                        "id": "right",
                        "question_id": "15",
                        "subquestion_id": "2",
                        "bbox": [0.5, 0, 1, 1],
                        "order": 1,
                    },
                ],
            }
        ]
    }


def install(client, exam, value=None):
    response = client.put(f"/v1/exams/{exam}/answer-template", json=value or template())
    assert response.status_code == 200, response.text
    return response.json()


def paper_job(client, exam, pages, **extra):
    return client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "candidates": [{"candidate_id": "c1", "kind": "paper", "pages": pages}],
            **extra,
        },
    )


def test_exam_creation_persists_template(client):
    response = client.post(
        "/v1/exams",
        json={
            "title": "fixed",
            "answer_template": template(),
            "questions": [{"id": "15", "stem": "q", "reference": "a", "max_score": 4}],
        },
    )
    assert response.status_code == 201
    assert response.json()["answer_template_version"]
    assert response.json()["answer_template"]["pages"][0]["boxes"][0]["id"] == "left"
    exam = client.get(f"/v1/exams/{response.json()['id']}").json()
    assert exam["answer_template"]["pages"][0]["boxes"][1]["id"] == "right"


def test_template_updates_do_not_mutate_existing_job_snapshots(client):
    asset, exam = upload(client), ready_exam(client, 1)
    legacy = paper_job(client, exam, [asset]).json()
    first_version = install(client, exam)["answer_template_version"]
    first = paper_job(client, exam, [asset]).json()
    changed = deepcopy(template())
    changed["pages"][0]["boxes"][0]["bbox"] = [0, 0, 0.4, 1]
    second_version = install(client, exam, changed)["answer_template_version"]
    second = paper_job(client, exam, [asset]).json()
    assert first_version != second_version
    assert client.get(f"/v1/jobs/{legacy['id']}").json()["answer_template"] is None
    assert client.get(f"/v1/jobs/{first['id']}").json()["answer_template_version"] == first_version
    assert (
        client.get(f"/v1/jobs/{second['id']}").json()["answer_template_version"] == second_version
    )


@pytest.mark.parametrize("box_ids", [None, ["left"], ["left", "unknown"], ["left", "left"]])
def test_incomplete_or_unknown_question_boxes_are_rejected(client, box_ids):
    asset, exam = upload(client), ready_exam(client, 1)
    install(client, exam)
    answer = {"question_id": "15", "assets": [asset, asset]}
    if box_ids is not None:
        answer["box_ids"] = box_ids
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "candidates": [{"candidate_id": "c1", "kind": "question", "answers": [answer]}],
        },
    )
    assert response.status_code == 422


def test_paper_with_missing_page_is_rejected(client):
    asset, exam = upload(client), ready_exam(client, 1)
    value = template()
    value["pages"].append({"width": 240, "height": 160, "boxes": []})
    install(client, exam, value)
    assert paper_job(client, exam, [asset]).status_code == 422


@pytest.mark.parametrize("kind", ["paper", "question"])
def test_every_frame_reaches_scoring_and_keeps_subquestion_ownership(runtime, monkeypatch, kind):
    client, settings, pipeline, provider, asset, exam = runtime
    settings.verification_policy = "off"
    drain(client.app.state.db, settings, pipeline)
    install(client, exam)
    original = provider.request
    grading_images, grading_prompts = [], []

    def capture(role, schema, prompt, images):
        if schema is Transcript:
            # The useful answer is only in the SECOND frame.
            value = {
                "lines": [{"id": "L1", "image_id": images[-1].id, "text": "second frame answer"}]
            }
            return {
                "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        if schema is Judgment:
            grading_images.extend(images)
            grading_prompts.append(prompt)
        return original(role, schema, prompt, images)

    monkeypatch.setattr(provider, "request", capture)
    candidate = (
        {"candidate_id": "c1", "kind": "paper", "pages": [asset]}
        if kind == "paper"
        else {
            "candidate_id": "c1",
            "kind": "question",
            "answers": [
                {"question_id": "15", "assets": [asset, asset], "box_ids": ["right", "left"]}
            ],
        }
    )
    response = client.post("/v1/jobs", json={"exam_id": exam, "candidates": [candidate]})
    assert response.status_code == 202, response.text
    job = response.json()["id"]
    drain(client.app.state.db, settings, pipeline)
    answer = client.get(f"/v1/jobs/{job}/results").json()[0]["answers"][0]
    assert answer["state"] == "scored", answer["error"]
    regions = answer["source_regions"]
    assert [r["box_id"] for r in regions] == ["left", "right"]
    assert [r["subquestion_id"] for r in regions] == ["1", "2"]
    assert len({r["asset_id"] for r in regions}) == 2
    assert len(grading_images) == 2
    assert '"box_id": "right"' in grading_prompts[0]
    assert answer["result"]["answer_template_version"]
    assert "layout" not in provider.calls
    if kind == "paper":
        assert [
            Image.open(BytesIO(client.get(f"/v1/assets/{r['asset_id']}").content)).size
            for r in regions
        ] == [(120, 160), (120, 160)]


def test_identical_blank_frames_are_kept_and_score_as_blank(runtime, monkeypatch):
    client, settings, pipeline, provider, _, exam = runtime
    settings.verification_policy = "off"
    drain(client.app.state.db, settings, pipeline)
    install(client, exam)
    image = BytesIO()
    Image.new("RGB", (240, 160), "white").save(image, "PNG")
    asset = client.post(
        "/v1/assets", files={"file": ("blank.png", image.getvalue(), "image/png")}
    ).json()[0]["id"]
    original = provider.request

    def blank(role, schema, prompt, images):
        if schema is Transcript:
            return {
                "choices": [
                    {
                        "message": {"content": '{"lines":[],"is_blank":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        return original(role, schema, prompt, images)

    monkeypatch.setattr(provider, "request", blank)
    job = paper_job(client, exam, [asset]).json()["id"]
    drain(client.app.state.db, settings, pipeline)
    answer = client.get(f"/v1/jobs/{job}/results").json()[0]["answers"][0]
    assert answer["state"] == "scored" and answer["result"]["grade"]["total_score"] == "0"
    assert len(answer["source_regions"]) == 2
    assert len({r["asset_id"] for r in answer["source_regions"]}) == 2


def test_missing_frame_cannot_be_graded_from_a_partial_checkpoint(runtime):
    client, settings, pipeline, provider, asset, exam = runtime
    drain(client.app.state.db, settings, pipeline)
    install(client, exam)
    job = paper_job(client, exam, [asset]).json()["id"]
    db = client.app.state.db
    # Prepare only, then reproduce an incomplete persisted source mapping.
    from ai4edu.db import CandidateRecord

    with db.session() as session:
        candidate = session.scalar(select(CandidateRecord).where(CandidateRecord.job_id == job))
        candidate_id = candidate.id
    execute_unit(db, settings, pipeline, "prepare_candidate", candidate_id)
    with db.session() as session:
        answer = session.scalar(select(AnswerRecord).where(AnswerRecord.job_id == job))
        answer.assets = answer.assets[:1]
        answer_id = answer.id
        session.commit()
    count = len(provider.calls)
    with pytest.raises(ValueError, match="答题框"):
        execute_unit(db, settings, pipeline, "grade_answer", answer_id)
    assert len(provider.calls) == count
    with db.session() as session:
        answer = session.get(AnswerRecord, answer_id)
        assert answer.state == "failed" and answer.result is None
        assert session.get(JobRecord, job).answer_template is not None


def test_cross_page_continuation_keeps_both_frames(runtime):
    client, settings, pipeline, provider, asset, exam = runtime
    settings.verification_policy = "off"
    drain(client.app.state.db, settings, pipeline)
    value = template()
    second = value["pages"][0]["boxes"].pop()
    second["subquestion_id"] = "1"
    value["pages"].append({"width": 240, "height": 160, "boxes": [second]})
    install(client, exam, value)
    job = paper_job(client, exam, [asset, asset]).json()["id"]
    drain(client.app.state.db, settings, pipeline)
    answer = client.get(f"/v1/jobs/{job}/results").json()[0]["answers"][0]
    assert answer["state"] == "scored"
    assert [r["page_index"] for r in answer["source_regions"]] == [0, 1]
    assert [r["subquestion_id"] for r in answer["source_regions"]] == ["1", "1"]
    assert "layout" not in provider.calls


def test_question_selection_collects_only_its_fixed_boxes(runtime):
    client, settings, pipeline, _, asset, exam = runtime
    settings.verification_policy = "off"
    drain(client.app.state.db, settings, pipeline)
    with client.app.state.db.session() as session:
        row = session.get(ExamRecord, exam)
        row.questions = [*row.questions, {**row.questions[0], "id": "16"}]
        row.rubrics = {**row.rubrics, "16": {**row.rubrics["15"], "question_id": "16"}}
        session.commit()
    value = template()
    value["pages"].append(
        {
            "width": 240,
            "height": 160,
            "boxes": [{"id": "q16", "question_id": "16", "bbox": [0, 0, 1, 1], "order": 0}],
        }
    )
    install(client, exam, value)
    job = paper_job(client, exam, [asset, asset], question_ids=["15"]).json()["id"]
    drain(client.app.state.db, settings, pipeline)
    answers = client.get(f"/v1/jobs/{job}/results").json()[0]["answers"]
    assert len(answers) == 1 and answers[0]["state"] == "scored"
    assert [r["box_id"] for r in answers[0]["source_regions"]] == ["left", "right"]


def test_legacy_named_question_submission_retains_old_hash(client):
    asset, exam = upload(client), ready_exam(client, 1)
    payload = {
        "exam_id": exam,
        "budget_cny": None,
        "strategy": None,
        "candidates": [
            {
                "candidate_id": "legacy",
                "kind": "question",
                "answers": [{"question_id": "15", "assets": [asset]}],
            }
        ],
    }
    response = client.post("/v1/jobs", json=payload)
    assert response.status_code == 202
    with client.app.state.db.session() as session:
        job = session.get(JobRecord, response.json()["id"])
        assert (
            job.fingerprint
            == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        )


def test_additive_template_migration_preserves_old_results(client):
    asset, exam = upload(client), ready_exam(client, 1)
    job = paper_job(client, exam, [asset]).json()["id"]
    db = client.app.state.db
    with db.engine.begin() as connection:
        connection.execute(text("ALTER TABLE exams DROP COLUMN answer_template"))
        connection.execute(text("ALTER TABLE jobs DROP COLUMN answer_template"))
    db.create_all()
    assert client.get(f"/v1/exams/{exam}").json()["state"] == "ready"
    summary = client.get(f"/v1/jobs/{job}").json()
    assert summary["total"] == 1 and summary["answer_template"] is None
