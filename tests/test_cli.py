import json

import httpx

from ai4edu import cli
from ai4edu.config import Settings


def test_submit_reuses_exam_and_uploads_only_selected_candidates(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "exam": {
                    "title": "test",
                    "questions": [{"id": "15", "stem": "q", "reference": "a", "max_score": 1}],
                },
                "candidates": [
                    {"candidate_id": "c1", "kind": "paper", "pages": ["one.png"]},
                    {"candidate_id": "c2", "kind": "paper", "pages": ["must-not-open.png"]},
                ],
            }
        )
    )
    (tmp_path / "one.png").write_bytes(b"image")
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/v1/assets":
            return httpx.Response(201, json=[{"id": "asset1"}])
        if request.url.path == "/v1/exams/existing":
            return httpx.Response(200, json={"state": "ready"})
        if request.url.path == "/v1/jobs":
            payload = json.loads(request.content)
            assert payload["exam_id"] == "existing"
            assert [item["candidate_id"] for item in payload["candidates"]] == ["c1"]
            return httpx.Response(202, json={"id": "job1", "exam_id": "existing"})
        raise AssertionError(request.url.path)

    real_client = httpx.Client
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    output = tmp_path / "submitted.json"
    result = cli.submit_manifest(
        manifest, "http://test", 5, "2", exam_id="existing", candidate_ids={"c1"}, output=output
    )
    assert result["id"] == "job1"
    assert json.loads(output.read_text())["id"] == "job1"
    assert "/v1/exams" not in calls


def test_submit_applies_shared_exam_budget_before_any_preparation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "exam": {
                    "title": "test",
                    "questions": [{"id": "15", "stem": "q", "reference": "a", "max_score": 1}],
                },
                "candidates": [{"candidate_id": "c1", "kind": "paper", "pages": ["one.png"]}],
            }
        )
    )
    (tmp_path / "one.png").write_bytes(b"image")

    def handle(request):
        if request.url.path == "/v1/assets":
            return httpx.Response(201, json=[{"id": "asset1"}])
        if request.url.path == "/v1/exams":
            assert json.loads(request.content)["budget_cny"] == "50"
            return httpx.Response(201, json={"id": "new"})
        if request.url.path == "/v1/exams/new":
            return httpx.Response(200, json={"state": "ready"})
        return httpx.Response(202, json={"id": "job1"})

    real_client = httpx.Client
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    assert cli.submit_manifest(manifest, "http://test", 5, "5", exam_budget="50")["id"] == "job1"


def test_uploads_overlap_without_changing_page_order_or_reuploading_shared_files(
    tmp_path, monkeypatch
):
    import threading

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "exam": {},
                "candidates": [
                    {"candidate_id": "c1", "kind": "paper", "pages": ["first.pdf", "second.png"]},
                    {"candidate_id": "c2", "kind": "paper", "pages": ["second.png"]},
                ],
            }
        )
    )
    (tmp_path / "first.pdf").write_bytes(b"first")
    (tmp_path / "second.png").write_bytes(b"second")
    barrier = threading.Barrier(2)
    uploaded = []
    lock = threading.Lock()

    def handle(request):
        if request.url.path == "/v1/assets":
            first = b'filename="first.pdf"' in request.content
            with lock:
                uploaded.append("first" if first else "second")
            try:
                barrier.wait(timeout=3)
            except threading.BrokenBarrierError as exc:
                raise AssertionError("uploads must overlap") from exc
            return httpx.Response(
                201, json=[{"id": "p1"}, {"id": "p2"}] if first else [{"id": "p3"}]
            )
        if request.url.path == "/v1/exams/existing":
            return httpx.Response(200, json={"state": "ready"})
        payload = json.loads(request.content)
        assert payload["candidates"][0]["pages"] == ["p1", "p2", "p3"]
        assert payload["candidates"][1]["pages"] == ["p3"]
        return httpx.Response(202, json={"id": "parallel"})

    real_client = httpx.Client
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    assert (
        cli.submit_manifest(manifest, "http://test", 5, None, exam_id="existing")["id"]
        == "parallel"
    )
    assert sorted(uploaded) == ["first", "second"]


def test_failed_upload_stops_scheduling_later_files_while_earlier_file_is_slow(
    tmp_path, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    manifest = tmp_path / "manifest.json"
    paths = ["first.png", "second.png", "third.png"]
    manifest.write_text(
        json.dumps(
            {"exam": {}, "candidates": [{"candidate_id": "c1", "kind": "paper", "pages": paths}]}
        )
    )
    for path in paths:
        (tmp_path / path).write_bytes(b"image")
    release_first = threading.Event()
    third_started = threading.Event()

    class DrainingPool(ThreadPoolExecutor):
        def shutdown(self, *args, **kwargs):
            release_first.set()
            return super().shutdown(*args, **kwargs)

    def handle(request):
        assert request.url.path == "/v1/assets"
        if b'filename="first.png"' in request.content:
            assert release_first.wait(timeout=5)
            return httpx.Response(201, json=[{"id": "first"}])
        if b'filename="second.png"' in request.content:
            return httpx.Response(500, json={"error": "upload failed"})
        third_started.set()
        release_first.set()
        return httpx.Response(201, json=[{"id": "third"}])

    real_client = httpx.Client
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "ThreadPoolExecutor", DrainingPool)
    with pytest.raises(httpx.HTTPStatusError):
        cli.submit_manifest(
            manifest, "http://test", 5, None, exam_id="existing", upload_concurrency=2
        )
    assert not third_started.is_set()
