import hashlib

from sqlalchemy import select

from ai4edu.bailian import ModelImage
from ai4edu.db import Asset, Database, uid
from ai4edu.images import image_info, render_upload
from ai4edu.storage import Storage


class AssetStore:
    def __init__(self, db: Database, storage: Storage):
        self.db, self.storage = db, storage

    def put_image(self, data: bytes) -> Asset:
        digest = hashlib.sha256(data).hexdigest()
        with self.db.session() as session:
            existing = session.scalar(select(Asset).where(Asset.sha256 == digest))
            if existing is not None:
                return existing
            _, width, height = image_info(data)
            asset = Asset(
                id=uid(), key=f"images/{uid()}.png", sha256=digest, width=width, height=height
            )
            self.storage.put(asset.key, data)
            session.add(asset)
            session.commit()
            return asset

    def upload(self, data: bytes, mime: str) -> list[Asset]:
        return [self.put_image(page) for page in render_upload(data, mime)]

    def image(self, ident: str) -> ModelImage:
        with self.db.session() as session:
            asset = session.get(Asset, ident)
            if asset is None:
                raise ValueError(f"图片不存在：{ident}")
            return ModelImage(
                asset.id, self.storage.read(asset.key), "image/png", asset.width, asset.height
            )
