import json

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from ai4edu.bailian import BailianClient, ModelImage
from ai4edu.config import Settings


class Result(BaseModel):
    text: str


@pytest.mark.parametrize("thinking", [False, True])
def test_bailian_uses_configured_url_model_and_image(thinking):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "choices": [{"message": {"content": '{"text":"answer"}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

    settings = Settings(
        _env_file=None,
        api_key=SecretStr("secret"),
        base_url="https://example.test/v1",
        model="chosen",
        thinking_roles=["grade"] if thinking else [],
        thinking_budget=128,
    )
    client = BailianClient(settings, transport=httpx.MockTransport(handler))
    response = client.request(
        "grade", Result, "grade", [ModelImage("image-1", b"png", "image/png", 100, 200)]
    )
    assert response["usage"]["total_tokens"] == 25
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://example.test/v1/chat/completions"
    assert body["model"] == "chosen"
    assert body["response_format"] == {"type": "json_object"}
    assert body["enable_thinking"] is thinking
    assert (body.get("thinking_budget") == 128) is thinking
    content = body["messages"][1]["content"]
    assert any(block["type"] == "image_url" for block in content)
    assert any(block.get("text") == "image_id=image-1" for block in content)
    assert "secret" not in json.dumps(body)


def test_provider_error_propagates_without_model_fallback():
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(429, json={"error": {"message": "limited"}})

    client = BailianClient(
        Settings(_env_file=None, api_key=SecretStr("secret"), model="chosen"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.request("grade", Result, "grade", [])
    assert len(attempts) == 1


def test_provider_error_explains_rejection_without_exposing_api_key():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "budget invalid secret"}},
        )

    client = BailianClient(
        Settings(_env_file=None, api_key=SecretStr("secret"), model="chosen"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            client.request("grade", Result, "grade", [])
    finally:
        client.close()
    message = str(caught.value)
    assert "InvalidParameter" in message
    assert "budget invalid" in message
    assert "secret" not in message
    assert caught.value.response.status_code == 400
