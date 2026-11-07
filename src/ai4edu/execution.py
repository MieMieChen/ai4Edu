"""Immutable, nonsecret model execution settings attached to each batch."""

from typing import Literal

from ai4edu import prompts
from ai4edu.config import Settings

EXECUTION_FIELDS = {
    "model",
    "layout_model",
    "rubric_model",
    "transcribe_model",
    "grade_model",
    "verify_model",
    "model_prices",
    "max_output_tokens",
    "max_transcription_tokens",
    "max_grading_tokens",
    "max_input_tokens",
    "transcription_images_per_call",
    "transcription_concurrency",
    "thinking_roles",
    "thinking_budget",
    "http_timeout",
    "max_pixels",
    "region_padding_ratio",
    "verification_policy",
    "audit_rate",
    "prompt_version",
}


def freeze_settings(settings: Settings) -> dict:
    frozen = settings.model_dump(mode="json", include=EXECUTION_FIELDS)
    for role in ("layout", "rubric", "transcribe", "grade", "verify"):
        frozen[f"{role}_model"] = settings.model_for(role)
    frozen["prompt_version"] = prompts.VERSION
    return frozen


def execution_settings(settings: Settings, frozen: dict) -> Settings:
    if frozen.keys() - EXECUTION_FIELDS:
        raise ValueError("批次执行配置包含不支持的字段")
    # Snapshots created before these options existed must retain the single-call path
    # and generic output cap, even after an operator enables a longer-answer profile.
    legacy_defaults = dict.fromkeys(
        (
            "max_transcription_tokens",
            "max_grading_tokens",
            "max_input_tokens",
            "transcription_images_per_call",
        )
    ) | {"transcription_concurrency": 2, "region_padding_ratio": 0.0}
    return Settings(
        _env_file=None,
        **(
            settings.model_dump()
            | legacy_defaults
            | {key: value for key, value in frozen.items() if key != "prompt_version"}
        ),
    )


def strategy_settings(settings: Settings, strategy: Literal["A", "B", "C"]) -> Settings:
    model = settings.model_for("verify") if strategy == "A" else settings.model
    return settings.model_copy(
        update={
            "transcribe_model": model,
            "grade_model": model,
            "verification_policy": "risk" if strategy == "C" else "off",
        }
    )
