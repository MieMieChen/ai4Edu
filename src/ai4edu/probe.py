import json
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
from pydantic import BaseModel

from ai4edu.bailian import BailianClient, ModelImage
from ai4edu.config import Settings


class ProbeOutput(BaseModel):
    text: str


def run_probe(
    settings: Settings,
    models: list[str],
    output: Path,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    if not 1 <= len(models) <= 2:
        raise ValueError("连通性探测每轮限1至2个模型；正式对照实验应单独设置预算")
    image = Image.new("RGB", (256, 96), "white")
    ImageDraw.Draw(image).text((24, 36), "2x + 3 = 7", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    inputs = [ModelImage("synthetic-equation", buffer.getvalue(), "image/png", 256, 96)]
    report = {
        "kind": "connectivity_probe",
        "plan_version": "v1.1",
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {"thinking": False, "max_output_tokens": 128, "image": "synthetic"},
        "calls": [],
    }
    for model in models:
        config = settings.model_copy(
            update={
                "grade_model": model,
                "max_output_tokens": 128,
                "http_timeout": min(settings.http_timeout, 30),
                "thinking_roles": [],
            }
        )
        client = BailianClient(config, transport=transport)
        entry = {"model": model, "usage": None, "reference_cost_cny": None}
        started = time.monotonic()
        try:
            raw = client.request(
                "grade", ProbeOutput, "识别图片中的等式，仅返回JSON，text为识别结果。", inputs
            )
            entry["usage"] = raw["usage"]
            price = settings.model_prices.get(model)
            if price:
                entry["reference_cost_cny"] = str(
                    (
                        raw["usage"]["prompt_tokens"] * price.input
                        + raw["usage"]["completion_tokens"] * price.output
                    )
                    / 1000000
                )
            ProbeOutput.model_validate_json(raw["choices"][0]["message"]["content"])
            entry.update(
                status="ok",
                http_status=200,
                returned_model=raw.get("model"),
                request_id=raw.get("id"),
            )
        except httpx.HTTPStatusError as exc:
            entry.update(
                status="http_error",
                http_status=exc.response.status_code,
                error_type=type(exc).__name__,
            )
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
        finally:
            entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
            client.close()
        report["calls"].append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
