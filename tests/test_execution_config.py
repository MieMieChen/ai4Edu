import json

from pydantic import SecretStr

from ai4edu.config import Settings
from ai4edu.db import JobRecord


def test_execution_settings_are_frozen_without_credentials():
    from ai4edu.execution import execution_settings, freeze_settings

    settings = Settings(
        _env_file=None,
        model="plus",
        verify_model="max",
        api_key=SecretStr("private-key"),
        base_url="https://private.example/v1",
        verification_policy="risk",
        model_prices={"plus": {"input": 2, "output": 8}},
    )
    frozen = freeze_settings(settings)
    assert "private-key" not in json.dumps(frozen)
    assert "private.example" not in json.dumps(frozen)
    changed = settings.model_copy(update={"model": "new", "max_output_tokens": 999})
    effective = execution_settings(changed, frozen)
    assert effective.model_for("grade") == "plus"
    assert effective.model_for("verify") == "max"
    assert effective.max_output_tokens == settings.max_output_tokens
    assert effective.api_key == settings.api_key
    from ai4edu import prompts

    assert frozen["prompt_version"] == prompts.VERSION


def test_job_model_persists_nonsecret_execution_settings():
    assert "execution_config" in JobRecord.__table__.columns


def test_strategy_settings_change_only_grading_roles():
    from ai4edu.execution import strategy_settings

    settings = Settings(
        _env_file=None, model="plus", verify_model="max", rubric_model="max", audit_rate=0.1
    )
    a = strategy_settings(settings, "A")
    b = strategy_settings(settings, "B")
    c = strategy_settings(settings, "C")
    assert a.model_for("transcribe") == a.model_for("grade") == "max"
    assert b.model_for("transcribe") == b.model_for("grade") == "plus"
    assert a.model_for("layout") == b.model_for("layout") == "plus"
    assert a.verification_policy == b.verification_policy == "off"
    assert c.verification_policy == "risk"
    assert c.audit_rate == 0.1


def test_explicit_database_initialization_upgrades_existing_job_table(tmp_path):
    from sqlalchemy import inspect, text

    from ai4edu.db import Database

    db = Database(f"sqlite:///{tmp_path / 'legacy.db'}")
    with db.engine.begin() as connection:
        connection.execute(text("CREATE TABLE jobs (id VARCHAR(32) PRIMARY KEY)"))
        connection.execute(text("INSERT INTO jobs (id) VALUES ('existing')"))
    db.create_all()
    assert "execution_config" in {
        column["name"] for column in inspect(db.engine).get_columns("jobs")
    }
    with db.engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT execution_config FROM jobs WHERE id='existing'")
            ).scalar_one()
            == "{}"
        )
