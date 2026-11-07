import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPrice(BaseModel):
    input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI4EDU_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )
    database_url: str = "postgresql+psycopg://ai4edu:ai4edu@localhost:5432/ai4edu"
    redis_url: str = "redis://localhost:6379/0"
    storage_backend: Literal["local", "oss"] = "local"
    storage_dir: Path = Path("data/assets")
    service_api_key: SecretStr | None = None
    credentials_file: Path | None = None
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "AI4EDU_API_KEY",
        ),
    )
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("DASHSCOPE_BASE_URL", "OPENAI_BASE_URL", "AI4EDU_BASE_URL"),
    )
    model: str = ""
    layout_model: str = ""
    rubric_model: str = ""
    transcribe_model: str = ""
    grade_model: str = ""
    verify_model: str = ""
    model_prices: dict[str, ModelPrice] = Field(default_factory=dict)
    max_output_tokens: int = Field(default=4096, gt=0)
    max_transcription_tokens: int | None = Field(default=None, gt=0)
    max_grading_tokens: int | None = Field(default=None, gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    transcription_images_per_call: int | None = Field(default=None, gt=0)
    transcription_concurrency: int = Field(default=2, gt=0)
    thinking_roles: list[Literal["layout", "rubric", "transcribe", "grade", "verify"]] = Field(
        default_factory=list
    )
    thinking_budget: int = Field(default=1024, gt=0)
    http_timeout: float = Field(default=120, gt=0)
    max_pixels: int = Field(default=2621440, ge=65536)
    region_padding_ratio: float = Field(default=0.04, ge=0, le=0.25)
    orient_question_images: bool = True
    max_upload_mb: int = Field(default=30, gt=0)
    model_concurrency: int = Field(default=8, gt=0)
    model_rpm: int = Field(default=60, gt=0)
    model_tpm: int = Field(default=100000, gt=0)
    verification_policy: Literal["all", "risk", "off"] = "risk"
    audit_rate: float = Field(default=0.1, ge=0, le=1)
    task_lease_seconds: int = Field(default=900, ge=300)
    oss_endpoint: str = ""
    oss_bucket: str = ""
    oss_access_key_id: str = ""
    oss_access_key_secret: SecretStr | None = None

    @field_validator(
        "api_key", "service_api_key", "oss_access_key_secret", "credentials_file", mode="before"
    )
    @classmethod
    def empty_secret(cls, value):
        return None if value == "" else value

    def model_for(self, role: str) -> str:
        model = getattr(self, f"{role}_model") or self.model
        if not model:
            raise ValueError(f"请配置 AI4EDU_MODEL 或 AI4EDU_{role.upper()}_MODEL")
        return model

    def output_limit_for(self, schema_name: str) -> int:
        # Verification emits either schema, so its role alone cannot select the cap.
        override = {
            "Transcript": self.max_transcription_tokens,
            "Judgment": self.max_grading_tokens,
        }.get(schema_name)
        return self.max_output_tokens if override is None else override


@lru_cache
def get_settings() -> Settings:
    return resolve_credentials(Settings())


def resolve_credentials(settings: Settings) -> Settings:
    if settings.credentials_file is None:
        return settings
    values = {}
    for line in settings.credentials_file.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"\s*(api_key|base_url)\s*[:=]\s*(.+?)\s*", line)
        if match is None:
            raise ValueError("凭据文件只接受 api_key 和 base_url 两个配置项")
        values[match[1]] = match[2].strip().strip("\"'")
    if set(values) != {"api_key", "base_url"} or not all(values.values()):
        raise ValueError("凭据文件需要非空的 api_key 和 base_url")
    updates = {}
    if settings.api_key is None:
        updates["api_key"] = SecretStr(values["api_key"])
    if "base_url" not in settings.model_fields_set or not settings.base_url:
        updates["base_url"] = values["base_url"]
    return settings.model_copy(update=updates)
