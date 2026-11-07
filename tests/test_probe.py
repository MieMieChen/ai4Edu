import json

import httpx

from ai4edu.config import Settings
from ai4edu.probe import run_probe


def test_probe_reports_usage_and_never_prints_credential_or_error_url(tmp_path):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["model"])
        if body["model"] == "model-b":
            return httpx.Response(403, json={"error": "secret-value"})
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "model": "snapshot-a",
                "choices": [{"message": {"content": '{"text":"2x+3=7"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    output = tmp_path / "probe.json"
    settings = Settings(_env_file=None, api_key="secret-value", base_url="https://example.test/v1")
    result = run_probe(
        settings, ["model-a", "model-b"], output, transport=httpx.MockTransport(handler)
    )
    assert seen == ["model-a", "model-b"]
    assert result["calls"][0]["status"] == "ok"
    assert result["calls"][0]["usage"]["total_tokens"] == 110
    assert result["calls"][1]["http_status"] == 403
    assert "secret-value" not in output.read_text()
    assert "example.test" not in output.read_text()
