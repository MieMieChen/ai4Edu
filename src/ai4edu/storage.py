from pathlib import Path
from typing import Protocol

from ai4edu.config import Settings


class Storage(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def read(self, key: str) -> bytes: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root

    def put(self, key: str, data: bytes):
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


class OSSStorage:
    def __init__(self, settings: Settings):
        import oss2

        if not all(
            (
                settings.oss_endpoint,
                settings.oss_bucket,
                settings.oss_access_key_id,
                settings.oss_access_key_secret,
            )
        ):
            raise ValueError("OSS 后端需要配置 endpoint、bucket 和 access key")
        self.bucket = oss2.Bucket(
            oss2.Auth(
                settings.oss_access_key_id, settings.oss_access_key_secret.get_secret_value()
            ),
            settings.oss_endpoint,
            settings.oss_bucket,
        )

    def put(self, key: str, data: bytes):
        self.bucket.put_object(key, data)

    def read(self, key: str) -> bytes:
        return self.bucket.get_object(key).read()


def make_storage(settings: Settings) -> Storage:
    return (
        OSSStorage(settings)
        if settings.storage_backend == "oss"
        else LocalStorage(settings.storage_dir)
    )
