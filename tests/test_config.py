from ai4edu.config import Settings, resolve_credentials


def test_empty_optional_credentials_do_not_enable_authentication():
    settings = Settings(_env_file=None, service_api_key="", api_key="", oss_access_key_secret="")
    assert settings.service_api_key is None
    assert settings.api_key is None
    assert settings.oss_access_key_secret is None


def test_declared_credentials_file_is_loaded_without_exposing_secret(tmp_path):
    path = tmp_path / "bailian.txt"
    path.write_text('api_key = "test-private-key"\nbase_url = https://example.test/v1\n')
    settings = resolve_credentials(Settings(_env_file=None, credentials_file=path, model="chosen"))
    assert settings.api_key.get_secret_value() == "test-private-key"
    assert settings.base_url == "https://example.test/v1"
    assert "test-private-key" not in repr(settings)


def test_explicit_environment_settings_override_credentials_file(tmp_path):
    path = tmp_path / "bailian.txt"
    path.write_text("api_key=file-key\nbase_url=https://file.test/v1\n")
    settings = resolve_credentials(
        Settings(
            _env_file=None,
            credentials_file=path,
            api_key="explicit-key",
            base_url="https://explicit.test/v1",
        )
    )
    assert settings.api_key.get_secret_value() == "explicit-key"
    assert settings.base_url == "https://explicit.test/v1"
