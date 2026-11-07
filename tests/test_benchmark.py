from copy import deepcopy

import pytest
from test_pipeline import drain, submit
from test_pipeline import runtime as original_runtime

runtime = original_runtime


def prediction(model, score="2", *, issue=False, key="sample-1"):
    return {
        "candidate_id": "c1",
        "question_id": "15",
        "status": "scored",
        "score": score,
        "answer_id": key,
        "branch_model": model,
        "replay_signature": "frozen-input",
        "cost_cny": "0.1",
        "latency_seconds": 2.0,
        "transcript": {
            "lines": [{"id": "L1", "image_id": "img", "text": "x=1"}],
            "issues": ["ambiguous"] if issue else [],
        },
        "grade": {
            "status": "scored",
            "total_score": score,
            "scores": [
                {"criterion_id": "R1", "score": score, "evidence_ids": ["L1"], "reason": "evidence"}
            ],
            "issues": [],
        },
    }


def test_simulated_c_uses_runtime_evidence_and_adds_sequential_cost():
    from ai4edu.benchmark import simulate_c

    a, b = prediction("max", "3"), prediction("plus", issue=True)
    result = simulate_c([a], [b], audit_rate=0, max_model="max")[0]
    assert result["score"] == "3"
    assert result["escalated"] is True
    assert result["cost_cny"] == "0.2"
    assert result["latency_seconds"] == 4.0
    assert result["simulated"] is True


def test_simulated_c_adds_sequential_flow_wall_or_keeps_it_unknown():
    from ai4edu.benchmark import simulate_c

    a, b = prediction("max", "3"), prediction("plus", issue=True)
    a["flow_elapsed_seconds"], b["flow_elapsed_seconds"] = 3, 5
    assert simulate_c([a], [b], audit_rate=0, max_model="max")[0]["flow_elapsed_seconds"] == 8
    b["flow_elapsed_seconds"] = None
    assert simulate_c([a], [b], audit_rate=0, max_model="max")[0]["flow_elapsed_seconds"] is None


def test_simulated_c_keeps_plus_when_runtime_rule_does_not_trigger():
    from ai4edu.benchmark import simulate_c

    a, b = prediction("max", "3"), prediction("plus")
    assert simulate_c([a], [b], audit_rate=0, max_model="max")[0]["score"] == "2"


def test_simulated_c_refuses_incompatible_max_replay():
    from ai4edu.benchmark import simulate_c

    a, b = prediction("max"), prediction("plus", issue=True)
    a["replay_signature"] = "different-images-or-settings"
    with pytest.raises(ValueError, match="一致"):
        simulate_c([a], [b], audit_rate=0, max_model="max")


def test_simulated_c_preserves_failure_without_zero_or_upgrade():
    from ai4edu.benchmark import simulate_c

    a, b = prediction("max"), prediction("plus")
    b.update(status="failed", score=None, grade=None, transcript=None)
    result = simulate_c([a], [b], audit_rate=0, max_model="max")[0]
    assert result["status"] == "failed"
    assert result["score"] is None
    assert result["escalated"] is False


def test_model_selection_prioritizes_coverage_and_error_before_price():
    from ai4edu.benchmark import select_strategy

    metrics = {
        "A": {
            "expected_count": 50,
            "resolved_count": 50,
            "mae": "0.5",
            "major_error_count": 0,
            "cost": {"known_cny": "5", "unknown_count": 0},
        },
        "B": {
            "expected_count": 50,
            "resolved_count": 49,
            "mae": "0.2",
            "major_error_count": 0,
            "cost": {"known_cny": "1", "unknown_count": 0},
        },
    }
    assert select_strategy(metrics)["strategy"] == "A"
    same_quality = deepcopy(metrics["A"])
    same_quality["cost"]["known_cny"] = "1"
    metrics["B"] = same_quality
    assert select_strategy(metrics)["strategy"] == "B"


def test_comparison_job_reuses_identical_assets_and_queues_only_grading(runtime):
    from sqlalchemy import select

    from ai4edu.benchmark import collect_predictions, create_comparison_job
    from ai4edu.db import AnswerRecord, JobRecord, Outbox

    client, settings, pipeline, _, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    source = submit(runtime)
    drain(db, settings, pipeline)
    comparison = create_comparison_job(db, settings, source, "A", budget_cny="2")
    with db.session() as session:
        original = session.scalar(select(AnswerRecord).where(AnswerRecord.job_id == source))
        cloned = session.scalar(select(AnswerRecord).where(AnswerRecord.job_id == comparison))
        assert cloned.assets == original.assets
        assert cloned.intermediate == {}
        assert session.get(JobRecord, comparison).exam_id == session.get(JobRecord, source).exam_id
        assert session.get(JobRecord, comparison).execution_config["verification_policy"] == "off"
        assert [item.task for item in session.scalars(select(Outbox))] == ["grade_answer"]
    drain(db, settings, pipeline)
    a = collect_predictions(db, comparison)
    b = collect_predictions(db, source, primary=True)
    assert a[0]["replay_signature"] == b[0]["replay_signature"]
    assert a[0]["score"] == b[0]["score"] == "3"
    assert create_comparison_job(db, settings, source, "A", budget_cny="2") == comparison


def test_celery_worker_reads_models_from_immutable_job_snapshot(runtime, monkeypatch):
    from sqlalchemy import select

    from ai4edu import tasks
    from ai4edu.benchmark import create_comparison_job
    from ai4edu.db import AnswerRecord

    client, settings, pipeline, provider, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    settings.verify_model = "max"
    source = submit(runtime)
    drain(db, settings, pipeline)
    comparison = create_comparison_job(db, settings, source, "A", budget_cny=None)
    models = []

    class Client:
        def __init__(self, config):
            self.config = config

        def request(self, role, *args):
            models.append(self.config.model_for(role))
            return provider.request(role, *args)

        def close(self):
            pass

    monkeypatch.setattr(tasks, "runtime", lambda: (db, pipeline))
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    monkeypatch.setattr(tasks, "BailianClient", Client)
    with db.session() as session:
        answer_id = session.scalar(select(AnswerRecord.id).where(AnswerRecord.job_id == comparison))
    tasks.run_celery_unit(None, "grade_answer", answer_id)
    assert models == ["max", "max"]


def test_report_rejects_changed_assets_even_without_c_escalation(runtime, tmp_path):
    import json

    from sqlalchemy import select

    from ai4edu.benchmark import create_comparison_job, write_evaluation_report
    from ai4edu.db import AnswerRecord, Asset

    client, settings, pipeline, _, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    source = submit(runtime)
    drain(db, settings, pipeline)
    maximum = create_comparison_job(db, settings, source, "A", budget_cny=None)
    drain(db, settings, pipeline)
    with db.session() as session:
        answer = session.scalar(select(AnswerRecord).where(AnswerRecord.job_id == maximum))
        copied = deepcopy(answer.assets)
        asset = session.get(Asset, copied[0]["asset_id"])
        other = Asset(
            id="different-id",
            key=asset.key,
            sha256="different-hash",
            width=asset.width,
            height=asset.height,
        )
        session.add(other)
        copied[0]["asset_id"] = other.id
        answer.assets = copied
        session.commit()
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "candidate_id": "anonymous-1",
                "question_id": "15",
                "score": "3",
                "max_score": "4",
                "split": "initial",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="一致"):
        write_evaluation_report(db, source, maximum, labels, tmp_path / "report", split="initial")


def test_failed_max_verification_is_counted_as_escalated(runtime):
    from ai4edu.benchmark import collect_predictions
    from ai4edu.schemas import Judgment

    client, settings, pipeline, provider, _, _ = runtime
    settings.audit_rate = 1
    drain(client.app.state.db, settings, pipeline)
    original = provider.request

    def request(role, schema, prompt, images):
        if role == "verify" and schema is Judgment:
            raise TimeoutError("verify timed out")
        return original(role, schema, prompt, images)

    provider.request = request
    source = submit(runtime)
    with pytest.raises(TimeoutError):
        drain(client.app.state.db, settings, pipeline)
    assert collect_predictions(client.app.state.db, source)[0]["escalated"] is True
    assert collect_predictions(client.app.state.db, source, primary=True)[0]["status"] == "scored"


def test_selection_retains_candidates_with_zero_resolved_results():
    from ai4edu.benchmark import select_strategy

    metrics = {
        "A": {
            "expected_count": 50,
            "resolved_count": 0,
            "mae": None,
            "major_error_count": 0,
            "cost": {"known_cny": "1", "unknown_count": 0},
        },
        "B": {
            "expected_count": 50,
            "resolved_count": 50,
            "mae": "1",
            "major_error_count": 2,
            "cost": {"known_cny": "2", "unknown_count": 0},
        },
    }
    assert select_strategy(metrics)["strategy"] == "B"
    metrics.pop("B")
    assert select_strategy(metrics)["strategy"] is None


def test_comparison_identity_changes_when_prepared_inputs_change(runtime):
    from sqlalchemy import select

    from ai4edu.benchmark import create_comparison_job
    from ai4edu.db import AnswerRecord, Asset

    client, settings, pipeline, _, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    source = submit(runtime)
    drain(db, settings, pipeline)
    first = create_comparison_job(db, settings, source, "A", budget_cny=None)
    with db.session() as session:
        answer = session.scalar(select(AnswerRecord).where(AnswerRecord.job_id == source))
        regions = deepcopy(answer.assets)
        original = session.get(Asset, regions[0]["asset_id"])
        other = Asset(
            id="corrected-image",
            key=original.key,
            sha256="corrected-hash",
            width=original.width,
            height=original.height,
        )
        session.add(other)
        regions[0]["asset_id"] = other.id
        answer.assets = regions
        session.commit()
    assert create_comparison_job(db, settings, source, "A", budget_cny=None) != first


def test_report_saves_zero_coverage_strategy_predictions(runtime, tmp_path):
    import json

    from sqlalchemy import select

    from ai4edu.benchmark import create_comparison_job, write_evaluation_report
    from ai4edu.db import AnswerRecord

    client, settings, pipeline, _, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    source = submit(runtime)
    drain(db, settings, pipeline)
    maximum = create_comparison_job(db, settings, source, "A", budget_cny=None)
    with db.session() as session:
        answer = session.scalar(select(AnswerRecord).where(AnswerRecord.job_id == maximum))
        answer.state, answer.error = "failed", "provider failure"
        session.commit()
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "candidate_id": "anonymous-1",
                "question_id": "15",
                "score": "3",
                "max_score": "4",
                "split": "initial",
            }
        )
        + "\n"
    )
    output = tmp_path / "report"
    report = write_evaluation_report(db, source, maximum, labels, output, split="initial")
    assert report["strategies"]["A"]["resolved_count"] == 0
    assert report["selection"]["strategy"] in {"B", "C"}
    assert json.loads((output / "predictions_A.jsonl").read_text())["status"] == "failed"


def test_api_can_request_c_profile_without_changing_default_models(runtime):
    client, settings, pipeline, _, asset, exam_id = runtime
    drain(client.app.state.db, settings, pipeline)
    settings.grade_model = "default-max"
    settings.verification_policy = "off"
    response = client.post(
        "/v1/jobs",
        json={
            "exam_id": exam_id,
            "strategy": "C",
            "candidates": [{"candidate_id": "anonymous", "kind": "paper", "pages": [asset]}],
        },
    )
    assert response.status_code == 202
    config = response.json()["execution_config"]
    assert config["grade_model"] == config["transcribe_model"] == settings.model
    assert config["verification_policy"] == "risk"
    assert settings.grade_model == "default-max"


def test_report_rejects_different_stage_output_limits(runtime, tmp_path):
    import json

    from ai4edu.benchmark import create_comparison_job, write_evaluation_report
    from ai4edu.db import JobRecord

    client, settings, pipeline, _, _, _ = runtime
    db = client.app.state.db
    drain(db, settings, pipeline)
    source = submit(runtime)
    drain(db, settings, pipeline)
    maximum = create_comparison_job(db, settings, source, "A", budget_cny=None)
    drain(db, settings, pipeline)
    with db.session() as session:
        job = session.get(JobRecord, maximum)
        job.execution_config = job.execution_config | {"max_transcription_tokens": 8192}
        session.commit()
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "candidate_id": "anonymous-1",
                "question_id": "15",
                "score": "3",
                "max_score": "4",
                "split": "initial",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="一致"):
        write_evaluation_report(db, source, maximum, labels, tmp_path / "report", split="initial")
