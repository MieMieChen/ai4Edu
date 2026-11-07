"""Contract tests for the independent, local compatible HTTP load-test service."""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ai4edu import prompts
from ai4edu.loadtest import (
    FaultPlan,
    IsolatedStack,
    assess_full_run,
    create_mock_app,
    isolated_environment,
    synthetic_paper,
)
from ai4edu.schemas import Judgment, PageLayout, Question, Rubric, Transcript
from ai4edu.scoring import finalize_grade, validate_rubric, validate_transcript


def request_body(schema, prompt, image_ids=("source-image",)):
    content = [{"type": "text", "text": prompt}]
    for ident in image_ids:
        content.extend(
            [
                {"type": "text", "text": f"image_id={ident}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
            ]
        )
    return {
        "model": "loadtest-only",
        "messages": [
            {"role": "system", "content": "JSON Schema\n" + json.dumps(schema.model_json_schema())},
            {"role": "user", "content": content},
        ],
    }


def question():
    return Question(id="15", stem="计算 2+2", reference="4", max_score="10")


def content(response):
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def test_compatible_service_round_trips_the_four_real_pipeline_schemas():
    app = create_mock_app(delay_seconds=0)
    with TestClient(app) as client:
        q = question()
        rubric = Rubric.model_validate_json(
            content(
                client.post(
                    "/v1/chat/completions",
                    json=request_body(Rubric, prompts.rubric_prompt(q, "0.5")),
                )
            )
        )
        validate_rubric(rubric, question_id=q.id, max_score=q.max_score, step=q.max_score / 20)
        assert sum(c.max_score for c in rubric.criteria) == q.max_score
        layout = PageLayout.model_validate_json(
            content(
                client.post(
                    "/v1/chat/completions",
                    json=request_body(
                        PageLayout, prompts.layout_prompt(["15", "16"], cropped=False)
                    ),
                )
            )
        )
        assert {r.question_id for r in layout.regions} == {"15", "16"}
        transcript = Transcript.model_validate_json(
            content(
                client.post(
                    "/v1/chat/completions",
                    json=request_body(Transcript, prompts.transcription_prompt()),
                )
            )
        )
        validate_transcript(transcript, {"source-image"})
        judgment = Judgment.model_validate_json(
            content(
                client.post(
                    "/v1/chat/completions",
                    json=request_body(Judgment, prompts.grading_prompt(q, rubric, transcript)),
                )
            )
        )
        grade = finalize_grade(rubric, transcript, judgment, q.max_score / 20)
        assert grade.total_score == q.max_score
        stats = client.get("/stats").json()
        assert stats["calls_by_schema"] == {
            "Rubric": 1,
            "PageLayout": 1,
            "Transcript": 1,
            "Judgment": 1,
        }


def test_seeded_faults_are_independent_of_request_arrival_order():
    identities = [f"answer-{n}" for n in range(100)]
    first = FaultPlan(seed="replay", probability=1)
    second = FaultPlan(seed="replay", probability=1)
    expected = {ident: first.select("Judgment", ident, "15") for ident in identities}
    actual = {ident: second.select("Judgment", ident, "15") for ident in reversed(identities)}
    assert actual == expected
    assert set(expected.values()) == {"429", "503", "timeout", "bad_json", "invalid_evidence"}


@pytest.mark.parametrize("fault", ["429", "503", "timeout", "bad_json", "invalid_evidence"])
def test_fault_protocol_is_observable_and_keeps_a_call_record(fault):
    app = create_mock_app(delay_seconds=0)
    with TestClient(app) as client:
        client.post(
            "/control",
            json={
                "faults": {"by_question": {"15": fault}},
                "timeout_seconds": 0.04,
            },
        ).raise_for_status()
        q = question()
        rubric = Rubric(
            question_id="15",
            max_score="10",
            solution="4",
            criteria=[
                {
                    "id": "15.c1",
                    "description": "计算",
                    "max_score": "10",
                    "rules": "正确得满分",
                }
            ],
        )
        transcript = Transcript(lines=[{"id": "e1", "image_id": "source-image", "text": "4"}])
        start = time.monotonic()
        response = client.post(
            "/v1/chat/completions",
            json=request_body(Judgment, prompts.grading_prompt(q, rubric, transcript)),
        )
        if fault in {"429", "503"}:
            assert response.status_code == int(fault)
        elif fault == "timeout":
            assert time.monotonic() - start >= 0.03
        elif fault == "bad_json":
            with pytest.raises(ValidationError):
                Judgment.model_validate_json(content(response))
        else:
            judgment = Judgment.model_validate_json(content(response))
            with pytest.raises(ValueError, match="证据"):
                finalize_grade(rubric, transcript, judgment, q.max_score / 20)
        stats = client.get("/stats").json()
        assert stats["total_calls"] == 1
        assert stats["faults"] == {fault: 1}


def test_service_observes_overlapping_requests_instead_of_serializing_them():
    async def run():
        app = create_mock_app(delay_seconds=0.04)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local"
        ) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/v1/chat/completions",
                        json=request_body(Transcript, prompts.transcription_prompt(), (str(n),)),
                    )
                    for n in range(4)
                ]
            )
            assert all(r.status_code == 200 for r in responses)
            stats = (await client.get("/stats")).json()
            assert stats["peak_inflight"] == 4
            assert stats["active"] == 0

    asyncio.run(run())


def test_environment_never_inherits_provider_credentials_or_real_stack_targets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "must-never-inherit")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://must-never-contact.invalid")
    monkeypatch.setenv("AI4EDU_TRANSCRIPTION_IMAGES_PER_CALL", "999")
    env = isolated_environment("safe123", tmp_path, concurrency=32)
    assert env["AI4EDU_DATABASE_URL"].endswith("/ai4edu_loadtest_safe123")
    assert env["AI4EDU_REDIS_URL"] == "redis://127.0.0.1:16379/3"
    assert env["DASHSCOPE_API_KEY"] == env["OPENAI_API_KEY"] == "local-loadtest-fake-key"
    assert env["AI4EDU_CREDENTIALS_FILE"] == ""
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:18091/v1"
    assert env["AI4EDU_MODEL_CONCURRENCY"] == "32"
    assert env["AI4EDU_STORAGE_DIR"] == str(tmp_path / "assets")
    inherited_chunk_option = "AI4EDU_TRANSCRIPTION_IMAGES_PER_CALL" in env
    assert not inherited_chunk_option


@pytest.mark.parametrize("run_id", ["../../ai4edu", "ai4edu;DROP DATABASE ai4edu", ""])
def test_isolation_rejects_unsafe_run_identifiers(tmp_path, run_id):
    with pytest.raises(ValueError):
        isolated_environment(run_id, tmp_path, concurrency=32)


def full_run_measurements():
    return {
        "candidates": 1000,
        "unique_candidate_ids": 1000,
        "answers": 5000,
        "unique_answer_ids": 5000,
        "answer_states": {"scored": 5000},
        "correct_scores": 5000,
        "api_queryable_candidates": 1000,
        "api_queryable_answers": 5000,
        "http_calls": {"PageLayout": 1000, "Transcript": 5000, "Judgment": 5000},
        "peak_inflight": 32,
        "model_cap": 32,
        "pending_outbox": 0,
        "duplicate_submit_same_job": True,
        "duplicate_delivery_extra_calls": 0,
    }


def test_full_run_acceptance_requires_real_call_volume_and_unique_queryable_results():
    assert all(assess_full_run(full_run_measurements()).values())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("candidates", 999),
        ("unique_candidate_ids", 999),
        ("answers", 4999),
        ("unique_answer_ids", 4999),
        ("correct_scores", 4999),
        ("api_queryable_candidates", 999),
        ("api_queryable_answers", 4999),
        ("answer_states", {"scored": 4999, "failed": 1}),
        ("peak_inflight", 1),
        ("peak_inflight", 33),
        ("pending_outbox", 1),
        ("duplicate_submit_same_job", False),
        ("duplicate_delivery_extra_calls", 1),
        ("http_calls", {"PageLayout": 1000, "Transcript": 0, "Judgment": 0}),
    ],
)
def test_acceptance_catches_drops_cached_work_serial_execution_and_duplicate_effects(field, bad):
    measurements = full_run_measurements() | {field: bad}
    assert not all(assess_full_run(measurements).values())


def test_each_candidate_and_question_has_a_distinct_raster_even_after_normalization():
    import hashlib

    from ai4edu.images import crop_and_rotate

    digests = set()
    for candidate in range(3):
        data = synthetic_paper(candidate)
        for index in range(5):
            crop = crop_and_rotate(data, (0, index / 5, 1, (index + 1) / 5), 0, 65536)
            digests.add(hashlib.sha256(crop).hexdigest())
    assert len(digests) == 15


def test_resource_sampling_is_optional_when_the_os_blocks_process_inspection(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("OS sandbox blocks ps")

    stack = IsolatedStack(tmp_path, "metrics")
    monkeypatch.setattr("ai4edu.loadtest.subprocess.run", denied)
    try:
        stack.sample_resources()
        assert "PermissionError" in stack.report["resource_sampling_unavailable"]
        assert stack.report["resource_samples"] == []
    finally:
        stack.api.close()
        stack.mock.close()


def test_stack_rejects_unsafe_id_before_creating_any_artifacts(tmp_path):
    with pytest.raises(ValueError):
        IsolatedStack(tmp_path, "../unsafe")
    assert not (tmp_path / "data").exists()


def test_cleanup_cannot_touch_redis_before_this_stack_has_claimed_ownership(tmp_path, monkeypatch):
    def unexpected_connection(*args, **kwargs):
        raise AssertionError("An unstarted stack must not inspect or delete Redis keys")

    stack = IsolatedStack(tmp_path, "unstarted")
    monkeypatch.setattr("redis.Redis.from_url", unexpected_connection)
    stack.cleanup()
    assert stack.report["cleanup"]["removed_own_new_redis_keys"] == 0


def test_actual_child_process_loads_no_project_dotenv_or_credentials_file(tmp_path):
    (tmp_path / ".env").write_text("AI4EDU_TRANSCRIPTION_IMAGES_PER_CALL=999\n")
    stack = IsolatedStack(tmp_path, "child_settings")
    child = stack.spawn(
        "settings_contract",
        [
            "-c",
            (
                "from ai4edu.config import get_settings; "
                "s=get_settings(); "
                "assert s.credentials_file is None; "
                "assert s.api_key.get_secret_value() == 'local-loadtest-fake-key'; "
                "assert s.base_url == 'http://127.0.0.1:18091/v1'; "
                "assert getattr(s, 'transcription_images_per_call', None) is None"
            ),
        ],
    )
    try:
        assert child.wait(timeout=10) == 0
    finally:
        stack.cleanup()
