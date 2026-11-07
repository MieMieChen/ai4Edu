import json

from fastapi.testclient import TestClient
from pydantic import SecretStr
from test_api import client as original_client
from test_api import ready_exam, upload
from test_pipeline import drain
from test_pipeline import runtime as original_runtime

from ai4edu.api import create_app
from ai4edu.config import Settings
from ai4edu.db import ModelCall
from ai4edu.execution import freeze_settings

client = original_client
runtime = original_runtime


def test_workbench_lists_real_batches_and_exams(client):
    asset = upload(client)
    exam = ready_exam(client)
    job = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "candidates": [{"candidate_id": "c1", "kind": "paper", "pages": [asset]}],
        },
    ).json()
    response = client.get("/v1/workbench")
    assert response.status_code == 200
    data = response.json()
    row = data["jobs"][0]
    assert row["id"] == job["id"]
    assert row["candidate_count"] == 1 and row["answer_count"] == 5
    assert row["counts"] == {"waiting": 5}
    assert row["exam_title"] == "数学主观题"
    assert data["exams"][0]["id"] == exam
    assert data["evaluation_available"] is False


def test_workbench_shell_is_public_but_data_remains_authenticated(tmp_path):
    app = create_app(
        Settings(
            _env_file=None,
            database_url=f"sqlite:///{tmp_path}/auth.db",
            storage_dir=tmp_path / "assets",
            service_api_key=SecretStr("service-token"),
        )
    )
    app.state.db.create_all()
    with TestClient(app) as client:
        redirect = client.get("/", follow_redirects=False)
        assert redirect.status_code == 307
        assert redirect.headers["location"] == "/demo/"
        assert client.get("/demo/").status_code == 200
        assert client.get("/v1/workbench").status_code == 401
        assert client.get("/v1/assets/unknown").status_code == 401
        assert (
            client.get(
                "/v1/workbench", headers={"Authorization": "Bearer service-token"}
            ).status_code
            == 200
        )


def test_unknown_usage_is_counted_separately_from_unpriced_usage(client):
    asset = upload(client)
    exam = ready_exam(client, 1)
    job = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "candidates": [{"candidate_id": "c1", "kind": "paper", "pages": [asset]}],
        },
    ).json()["id"]
    with client.app.state.db.session() as session:
        session.add_all(
            [
                ModelCall(
                    exam_id=exam,
                    job_id=job,
                    unit_id=job,
                    role="grade",
                    fingerprint=str(index) * 64,
                    model="test-model",
                    state="failed" if usage is None else "succeeded",
                    usage=usage,
                    cost=None,
                )
                for index, usage in enumerate(
                    [None, {"prompt_tokens": 120, "completion_tokens": 45}]
                )
            ]
        )
        session.commit()
    summary = client.get(f"/v1/jobs/{job}").json()
    assert summary["unknown_usage_calls"] == 1
    assert summary["unpriced_or_unknown_calls"] == 2
    assert summary["input_tokens"] == 120 and summary["output_tokens"] == 45


def test_offline_evaluation_uses_labels_without_new_model_calls(runtime, tmp_path):
    client, settings, pipeline, provider, asset, exam = runtime
    settings.verification_policy = "off"
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"candidate_id": "c1", "question_id": "15", "score": "2", "max_score": "4"})
        + "\n"
    )
    settings.evaluation_labels_file = labels
    drain(client.app.state.db, settings, pipeline)
    job = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "candidates": [
                {
                    "candidate_id": "c1",
                    "kind": "question",
                    "answers": [{"question_id": "15", "assets": [asset]}],
                }
            ],
        },
    ).json()["id"]
    drain(client.app.state.db, settings, pipeline)
    count = len(provider.calls)
    response = client.get(f"/v1/jobs/{job}/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["metrics"]["mae"] == "1"
    assert data["all_expected_within_1_rate"] == 1
    assert data["metrics"]["records"][0]["label_score"] == "2"
    assert len(provider.calls) == count
    assert "evaluation_labels_file" not in freeze_settings(settings)
    labels.write_text(
        json.dumps({"candidate_id": "other", "question_id": "15", "score": "2", "max_score": "4"})
        + "\n"
    )
    unmatched = client.get(f"/v1/jobs/{job}/evaluation").json()
    assert unmatched["available"] is False
    assert unmatched["matched_count"] == 0
    assert unmatched["expected_count"] == 1


def test_unconfigured_evaluation_is_explicitly_unavailable(client):
    asset = upload(client)
    exam = ready_exam(client, 1)
    job = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam,
            "candidates": [{"candidate_id": "c1", "kind": "paper", "pages": [asset]}],
        },
    ).json()["id"]
    data = client.get(f"/v1/jobs/{job}/evaluation").json()
    assert data["available"] is False
