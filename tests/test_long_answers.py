"""Long answers keep bounded requests, faithful evidence, and resumable paid work."""

import json
import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from io import BytesIO

import fakeredis
import httpx
import pytest
from PIL import Image, ImageDraw
from pydantic import SecretStr, ValidationError
from sqlalchemy import select

from ai4edu import pipeline as pipeline_module
from ai4edu import prompts
from ai4edu.assets import AssetStore
from ai4edu.bailian import BailianClient, ModelImage
from ai4edu.calls import Calls
from ai4edu.config import Settings
from ai4edu.db import (
    AnswerRecord,
    CandidateRecord,
    Database,
    ExamRecord,
    JobRecord,
    ModelCall,
    StaleExecution,
)
from ai4edu.execution import execution_settings, freeze_settings
from ai4edu.limits import RateLimitWait, RedisLimiter
from ai4edu.pipeline import Pipeline
from ai4edu.schemas import Judgment, PageLayout, Rubric, Transcript
from ai4edu.storage import LocalStorage


def response(value):
    return {
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def configured(**overrides):
    return Settings(
        _env_file=None,
        **(
            {
                "model": "test",
                "api_key": SecretStr("test-key"),
                "model_prices": {"test": {"input": 2, "output": 8}},
                "verification_policy": "off",
            }
            | overrides
        ),
    )


@pytest.fixture
def answer_db(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'long-answers.db'}")
    db.create_all()
    with db.session() as session:
        session.add(
            ExamRecord(
                id="exam",
                title="long answer",
                questions=[],
                score_step=Decimal("0.5"),
                state="ready",
            )
        )
        session.commit()
        session.add(JobRecord(id="job", exam_id="exam", fingerprint="test", execution_config={}))
        session.commit()
        session.add(CandidateRecord(id="candidate", job_id="job", candidate_id="1", payload={}))
        session.commit()
        session.add(
            AnswerRecord(
                id="answer",
                job_id="job",
                candidate_record_id="candidate",
                question_id="19",
                state="running",
                execution_token="owner",
            )
        )
        session.commit()
    yield db
    db.engine.dispose()


class RecordingLimiter:
    def __init__(self):
        self.reservations = []
        self.releases = []

    def acquire(self, model, tokens):
        self.reservations.append((model, tokens))
        return str(len(self.reservations))

    def release(self, model, ticket, actual_tokens):
        self.releases.append((model, ticket, actual_tokens))


def calls_for(db, settings, client, limiter):
    return Calls(
        db,
        settings,
        client,
        limiter,
        exam_id="exam",
        unit_id="answer",
        task="grade_answer",
        execution_token="owner",
        job_id="job",
    )


@pytest.mark.parametrize(
    ("role", "schema", "expected"),
    [
        ("transcribe", Transcript, 8192),
        ("verify", Transcript, 8192),
        ("grade", Judgment, 6144),
        ("verify", Judgment, 6144),
        ("layout", PageLayout, 4096),
        ("rubric", Rubric, 4096),
    ],
)
def test_schema_specific_output_caps_reach_provider(role, schema, expected):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response({}))

    settings = configured(max_transcription_tokens=8192, max_grading_tokens=6144)
    client = BailianClient(settings, transport=httpx.MockTransport(handle))
    try:
        client.request(role, schema, "faithful transcription", [])
    finally:
        client.close()
    assert requests[0]["max_tokens"] == expected


def test_effective_cap_is_used_for_reservation_cost_and_cache(answer_db):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response({"lines": []}))

    prompt = "student writing"
    images = [ModelImage("original", b"png", "image/png", 320, 640)]
    input_estimate = len(prompt.encode()) + len(json.dumps(Transcript.model_json_schema()).encode())
    input_estimate += 2 * math.ceil(320 * 640 / 1024) + 2
    settings = configured(max_transcription_tokens=8192)
    limiter = RecordingLimiter()
    client = BailianClient(settings, transport=httpx.MockTransport(handle))
    try:
        calls = calls_for(answer_db, settings, client, limiter)
        calls.complete("transcribe", Transcript, prompt, images)
        calls.complete("transcribe", Transcript, prompt, images)
        settings.max_transcription_tokens = 8193
        calls.complete("transcribe", Transcript, prompt, images)
    finally:
        client.close()
    assert [r["max_tokens"] for r in requests] == [8192, 8193]
    assert limiter.reservations == [
        ("test", input_estimate + 8192),
        ("test", input_estimate + 8193),
    ]
    with answer_db.session() as session:
        records = session.scalars(select(ModelCall).order_by(ModelCall.created_at)).all()
        assert len(records) == 2
        assert records[0].fingerprint != records[1].fingerprint
        assert records[0].reserved_cost == Decimal(input_estimate * 2 + 8192 * 8) / 1000000
        assert records[1].reserved_cost == Decimal(input_estimate * 2 + 8193 * 8) / 1000000


def test_input_limit_preflight_stops_before_limiter_or_provider(answer_db):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response({"lines": []}))

    settings = configured(max_input_tokens=10)
    limiter = RecordingLimiter()
    client = BailianClient(settings, transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(ValueError, match="输入.*Token.*10"):
            calls_for(answer_db, settings, client, limiter).complete(
                "transcribe",
                Transcript,
                "read this",
                [ModelImage("page", b"png", "image/png", 100, 100)],
            )
    finally:
        client.close()
    assert requests == []
    assert limiter.reservations == []
    with answer_db.session() as session:
        assert session.scalar(select(ModelCall)) is None
        assert session.get(JobRecord, "job").committed_cost == 0


def test_new_options_are_frozen_and_legacy_jobs_keep_original_caps():
    optional = {
        "max_transcription_tokens": 8192,
        "max_grading_tokens": 6144,
        "max_input_tokens": 250000,
        "transcription_images_per_call": 1,
        "transcription_concurrency": 3,
    }
    modern = configured(max_output_tokens=8192, **optional)
    frozen = freeze_settings(modern)
    assert {key: frozen.get(key) for key in optional} == optional
    legacy = freeze_settings(configured())
    for key in optional:
        legacy.pop(key, None)
    effective = execution_settings(modern, legacy)
    assert effective.output_limit_for("Transcript") == 4096
    assert effective.output_limit_for("Judgment") == 4096
    assert effective.max_input_tokens is None
    assert effective.transcription_images_per_call is None
    assert effective.transcription_concurrency == 2


@pytest.mark.parametrize(
    "name",
    [
        "max_transcription_tokens",
        "max_grading_tokens",
        "max_input_tokens",
        "transcription_images_per_call",
        "transcription_concurrency",
    ],
)
def test_long_answer_limits_must_be_positive(name):
    with pytest.raises(ValidationError):
        configured(**{name: 0})


def make_answer(db, tmp_path, settings, *, pages=4):
    assets = AssetStore(db, LocalStorage(tmp_path / "assets"))
    ids = []
    for index in range(pages + 1):
        image = Image.new("RGB", (240, 160), "white")
        ImageDraw.Draw(image).text((20, 20), f"page {index}", fill="black")
        stream = BytesIO()
        image.save(stream, format="PNG")
        ids.append(assets.put_image(stream.getvalue()).id)
    question = {
        "id": "19",
        "stem": "QUESTION_SENTINEL",
        "reference": "REFERENCE_SENTINEL",
        "question_assets": [ids[-1]],
        "max_score": 4,
    }
    rubric = {
        "question_id": "19",
        "max_score": 4,
        "solution": "RUBRIC_SENTINEL",
        "criteria": [{"id": "R1", "description": "proof", "max_score": 4, "rules": "proof"}],
    }
    with db.session() as session:
        exam = session.get(ExamRecord, "exam")
        exam.questions, exam.rubrics = [question], {"19": rubric}
        session.get(JobRecord, "job").execution_config = freeze_settings(settings)
        session.get(AnswerRecord, "answer").assets = [{"asset_id": ident} for ident in ids[:-1]]
        session.commit()
    return assets, ids[:-1], ids[-1]


class ChunkModel:
    def __init__(self, *, first_wave=0, blank_ids=(), unclear_ids=()):
        self.requests = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(first_wave) if first_wave else None
        self.blank_ids, self.unclear_ids = set(blank_ids), set(unclear_ids)

    def request(self, role, schema, prompt, images):
        with self.lock:
            number = sum(request[1] is Transcript for request in self.requests)
            self.requests.append((role, schema, prompt, list(images)))
            if schema is Transcript:
                self.active += 1
                self.peak = max(self.peak, self.active)
        if schema is Transcript:
            try:
                if self.barrier and number < self.barrier.parties:
                    self.barrier.wait(timeout=5)
                lines = []
                for index, image in reversed(list(enumerate(images))):
                    if image.id in self.blank_ids:
                        continue
                    unclear = image.id in self.unclear_ids
                    lines.append(
                        {
                            "id": f"L{index + 1}",
                            "image_id": image.id,
                            "text": f"student {image.id}",
                            "status": "unclear" if unclear else "clear",
                            "alternatives": ["x", "y"] if unclear else [],
                            "issues": ["ambiguous symbol"] if unclear else [],
                        }
                    )
                    lines.append(
                        {
                            "id": f"X{index + 1}",
                            "image_id": image.id,
                            "text": "abandoned attempt",
                            "status": "crossed_out",
                        }
                    )
                return response(
                    {
                        "lines": lines,
                        "is_blank": all(image.id in self.blank_ids for image in images),
                        "issues": ["page ambiguity"]
                        if any(image.id in self.unclear_ids for image in images)
                        else [],
                    }
                )
            finally:
                with self.lock:
                    self.active -= 1
        assert schema is Judgment
        transcript = Transcript.model_validate_json(prompt.split("\n忠实转写：", 1)[1])
        evidence = [line.id for line in transcript.lines if line.status == "clear"]
        return response(
            {
                "scores": [
                    {
                        "criterion_id": "R1",
                        "score": 4 if evidence else None,
                        "evidence_ids": evidence,
                        "reason": "supported" if evidence else "cannot read",
                    }
                ]
            }
        )


def chunk_pipeline(db, tmp_path, settings, provider, *, pages=4):
    assets, ids, question_id = make_answer(db, tmp_path, settings, pages=pages)
    limiter = RedisLimiter(
        fakeredis.FakeRedis(), concurrency=8, rpm=1000, tpm=1000000, lease_seconds=900
    )
    return Pipeline(db, settings, assets, provider, limiter), ids, question_id


def test_transcription_chunks_run_in_parallel_and_grade_complete_original_answer(
    answer_db, tmp_path
):
    settings = configured(transcription_images_per_call=2, transcription_concurrency=2)
    provider = ChunkModel(first_wave=2)
    pipeline, ids, question_id = chunk_pipeline(answer_db, tmp_path, settings, provider, pages=6)
    provider.unclear_ids = {ids[1]}
    pipeline.grade_answer("answer", "owner")
    transcriptions = [request for request in provider.requests if request[1] is Transcript]
    assert len(transcriptions) == 3
    assert provider.peak == 2
    assert {tuple(image.id for image in request[3]) for request in transcriptions} == {
        tuple(ids[index : index + 2]) for index in range(0, len(ids), 2)
    }
    for role, _, prompt, images in transcriptions:
        assert role == "transcribe"
        assert prompt == prompts.transcription_prompt()
        assert not any(
            sentinel in prompt
            for sentinel in ("QUESTION_SENTINEL", "REFERENCE_SENTINEL", "RUBRIC_SENTINEL")
        )
        assert question_id not in [image.id for image in images]
    judgments = [request for request in provider.requests if request[1] is Judgment]
    assert len(judgments) == 1
    assert [image.id for image in judgments[0][3]] == ids + [question_id]
    assert "RUBRIC_SENTINEL" in judgments[0][2]
    with answer_db.session() as session:
        answer = session.get(AnswerRecord, "answer")
        transcript = Transcript.model_validate(answer.result["transcript"])
        assert [line.image_id for line in transcript.lines] == [
            ident for ident in ids for _ in range(2)
        ]
        assert len({line.id for line in transcript.lines}) == len(transcript.lines)
        unclear = next(line for line in transcript.lines if line.status == "unclear")
        assert unclear.alternatives == ["x", "y"]
        assert unclear.issues == ["ambiguous symbol"]
        assert transcript.issues == ["page ambiguity"]
        assert sum(line.status == "crossed_out" for line in transcript.lines) == len(ids)
        assert answer.result["grade"]["total_score"] == "4"


@pytest.mark.parametrize("error", [ConnectionError("temporary"), RateLimitWait(1)])
def test_failed_chunk_drains_and_checkpoints_success_before_retry(
    answer_db, tmp_path, monkeypatch, error
):
    settings = configured(transcription_images_per_call=1, transcription_concurrency=2)
    provider = ChunkModel()
    pipeline, ids, _ = chunk_pipeline(answer_db, tmp_path, settings, provider, pages=4)
    successful_started = threading.Event()
    failed = threading.Event()
    release_success = threading.Event()
    success_finished = threading.Event()
    request = provider.request
    attempts = Counter()
    original_wait = pipeline_module.wait

    def observe_failure(futures, **kwargs):
        done, pending = original_wait(futures, **kwargs)
        if any(future.exception() is error for future in done):
            failed.set()
        return done, pending

    monkeypatch.setattr(pipeline_module, "wait", observe_failure)

    def controlled_request(role, schema, prompt, images):
        if schema is Transcript:
            attempts[images[0].id] += 1
            if images[0].id == ids[0] and attempts[images[0].id] == 1:
                assert successful_started.wait(timeout=5)
                raise error
            if images[0].id == ids[1]:
                successful_started.set()
                assert release_success.wait(timeout=5)
                value = request(role, schema, prompt, images)
                success_finished.set()
                return value
        return request(role, schema, prompt, images)

    provider.request = controlled_request
    with ThreadPoolExecutor(max_workers=1) as executor:
        unit = executor.submit(pipeline.grade_answer, "answer", "owner")
        try:
            assert failed.wait(timeout=5)
            assert not unit.done()
            assert not success_finished.is_set()
        finally:
            release_success.set()
        with pytest.raises(type(error)):
            unit.result(timeout=5)
    assert success_finished.is_set()
    assert attempts == Counter({ids[0]: 1, ids[1]: 1})
    with answer_db.session() as session:
        checkpoint = session.get(AnswerRecord, "answer").intermediate
        parts = [
            value for key, value in checkpoint.items() if key.startswith("primary_transcript_part_")
        ]
        assert len(parts) == 1
        assert parts[0]["lines"][0]["image_id"] == ids[1]
        assert "primary_transcript" not in checkpoint
        assert session.get(AnswerRecord, "answer").execution_token == "owner"
    pipeline.grade_answer("answer", "owner")
    assert attempts == Counter({ids[0]: 2, **dict.fromkeys(ids[1:], 1)})
    with answer_db.session() as session:
        assert session.get(AnswerRecord, "answer").state == "scored"
        assert (
            len(session.scalars(select(ModelCall).where(ModelCall.state == "succeeded")).all()) == 5
        )


@pytest.mark.parametrize("mode", ["all_blank", "part_blank", "all_unclear"])
def test_chunk_merge_preserves_blank_and_unclear_semantics(answer_db, tmp_path, mode):
    settings = configured(transcription_images_per_call=1, transcription_concurrency=2)
    provider = ChunkModel()
    pipeline, ids, _ = chunk_pipeline(answer_db, tmp_path, settings, provider, pages=2)
    provider.blank_ids = set(
        ids if mode == "all_blank" else ids[:1] if mode == "part_blank" else []
    )
    provider.unclear_ids = set(ids if mode == "all_unclear" else [])
    pipeline.grade_answer("answer", "owner")
    with answer_db.session() as session:
        result = session.get(AnswerRecord, "answer").result
    transcript = Transcript.model_validate(result["transcript"])
    assert sum(request[1] is Transcript for request in provider.requests) == 2
    assert transcript.is_blank is (mode == "all_blank")
    judgments = [request for request in provider.requests if request[1] is Judgment]
    assert len(judgments) == (0 if mode == "all_blank" else 1)
    if mode == "all_blank":
        assert result["grade"]["total_score"] == "0"
    elif mode == "part_blank":
        assert {line.image_id for line in transcript.lines} == {ids[1]}
        assert result["grade"]["total_score"] == "4"
    else:
        assert result["grade"]["total_score"] is None
        assert result["grade"]["status"] == "ungradable"
        assert sum(line.status == "unclear" for line in transcript.lines) == 2


def test_unconfigured_transcription_retains_single_call_and_evidence_ids(answer_db, tmp_path):
    settings = configured()
    provider = ChunkModel()
    pipeline, ids, _ = chunk_pipeline(answer_db, tmp_path, settings, provider, pages=3)
    pipeline.grade_answer("answer", "owner")
    transcriptions = [request for request in provider.requests if request[1] is Transcript]
    assert len(transcriptions) == 1
    assert [image.id for image in transcriptions[0][3]] == ids
    with answer_db.session() as session:
        answer = session.get(AnswerRecord, "answer")
        assert not any(key.startswith("primary_transcript_part_") for key in answer.intermediate)
        assert {line["id"] for line in answer.result["transcript"]["lines"]} == {
            f"{prefix}{number}" for prefix in ("L", "X") for number in range(1, 4)
        }


def test_verified_transcription_has_independent_part_checkpoints(answer_db, tmp_path):
    settings = configured(
        transcription_images_per_call=1,
        transcription_concurrency=2,
        verification_policy="all",
    )
    provider = ChunkModel(first_wave=2)
    pipeline, ids, question_id = chunk_pipeline(answer_db, tmp_path, settings, provider, pages=3)
    pipeline.grade_answer("answer", "owner")
    transcriptions = [request for request in provider.requests if request[1] is Transcript]
    assert Counter(request[0] for request in transcriptions) == {"transcribe": 3, "verify": 3}
    assert all(request[2] == prompts.transcription_prompt() for request in transcriptions)
    judgments = [request for request in provider.requests if request[1] is Judgment]
    assert [request[0] for request in judgments] == ["grade", "verify"]
    assert all([image.id for image in request[3]] == ids + [question_id] for request in judgments)
    with answer_db.session() as session:
        answer = session.get(AnswerRecord, "answer")
        for prefix in ("primary", "verified"):
            assert all(
                f"{prefix}_transcript_part_{index}" in answer.intermediate for index in range(1, 4)
            )
        assert answer.result["verified"] is True


def test_prompt_version_guard_precedes_chunk_checkpoint_reuse(answer_db, tmp_path):
    settings = configured(transcription_images_per_call=1)
    provider = ChunkModel()
    pipeline, ids, _ = chunk_pipeline(answer_db, tmp_path, settings, provider, pages=2)
    with answer_db.session() as session:
        job = session.get(JobRecord, "job")
        job.execution_config = job.execution_config | {"prompt_version": "obsolete"}
        session.get(AnswerRecord, "answer").intermediate = {
            "primary_transcript_part_1": {
                "lines": [{"id": "L1", "image_id": ids[0], "text": "proof"}]
            }
        }
        session.commit()
    with pytest.raises(ValueError, match="提示词版本"):
        pipeline.grade_answer("answer", "owner")
    assert provider.requests == []


def test_input_limit_allows_exact_conservative_estimate(answer_db):
    requests = []
    prompt = "read"
    image = ModelImage("page", b"png", "image/png", 100, 100)
    estimate = len(prompt.encode()) + len(json.dumps(Transcript.model_json_schema()).encode()) + 130
    settings = configured(max_input_tokens=estimate)
    limiter = RecordingLimiter()

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response({"lines": []}))

    client = BailianClient(settings, transport=httpx.MockTransport(handle))
    try:
        calls_for(answer_db, settings, client, limiter).complete(
            "transcribe", Transcript, prompt, [image]
        )
    finally:
        client.close()
    assert len(requests) == 1
    assert limiter.reservations == [("test", estimate + 4096)]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (RateLimitWait(1), ConnectionError("provider failed"), ConnectionError),
        (RateLimitWait(1), StaleExecution("ownership moved"), StaleExecution),
        (ConnectionError("provider failed"), StaleExecution("ownership moved"), StaleExecution),
        (RateLimitWait(1), RateLimitWait(5), RateLimitWait),
    ],
)
def test_chunk_failures_prioritize_stale_then_fatal_then_longest_capacity_wait(
    answer_db, tmp_path, monkeypatch, first, second, expected
):
    settings = configured(transcription_images_per_call=1, transcription_concurrency=2)
    provider = ChunkModel()
    pipeline, ids, _ = chunk_pipeline(answer_db, tmp_path, settings, provider, pages=2)
    first_observed = threading.Event()
    second_finished = threading.Event()
    original_wait = pipeline_module.wait

    def observe_first_completion(futures, **kwargs):
        done, pending = original_wait(futures, **kwargs)
        if any(future.exception() is first for future in done):
            first_observed.set()
        return done, pending

    monkeypatch.setattr(pipeline_module, "wait", observe_first_completion)

    def fail_in_order(role, schema, prompt, images):
        assert schema is Transcript
        if images[0].id == ids[0]:
            raise first
        assert first_observed.wait(timeout=5)
        second_finished.set()
        raise second

    provider.request = fail_in_order
    with pytest.raises(expected) as raised:
        pipeline.grade_answer("answer", "owner")
    assert second_finished.is_set()
    if expected is RateLimitWait:
        assert raised.value.seconds == 5
