"""Decode uploads and transform regions in EXIF-corrected image coordinates."""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps

_MIMES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def _decode(data: bytes) -> tuple[str, Image.Image]:
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in _MIMES:
            raise ValueError(f"Unsupported image format: {image.format}")
        mime = _MIMES[image.format]
        normalized = ImageOps.exif_transpose(image)
        if "A" in normalized.getbands() or "transparency" in normalized.info:
            normalized = Image.alpha_composite(
                Image.new("RGBA", normalized.size, "white"), normalized.convert("RGBA")
            )
        return mime, normalized.convert("RGB")


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_info(data: bytes) -> tuple[str, int, int]:
    """Return the actual MIME type and the EXIF-corrected pixel dimensions."""
    mime, image = _decode(data)
    return mime, image.width, image.height


def render_upload(data: bytes, content_type: str) -> list[bytes]:
    """Normalize a supported raster or every PDF page to an RGB PNG on white."""
    if content_type == "application/pdf":
        with pymupdf.open(stream=data, filetype="pdf") as document:
            return [
                page.get_pixmap(dpi=150, colorspace=pymupdf.csRGB, alpha=False).tobytes("png")
                for page in document
            ]
    if content_type not in _MIMES.values():
        raise ValueError(f"Unsupported content type: {content_type}")
    _, image = _decode(data)
    return [_png(image)]


def _rotated_size(size: tuple[int, int], rotation: float) -> tuple[int, int]:
    width, height = size
    angle = math.radians(rotation % 360)
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    if rotation % 90 == 0:
        # Exact quarter-turns must not gain a pixel from trigonometric roundoff.
        cosine, sine = round(cosine), round(sine)
    return math.ceil(width * cosine + height * sine), math.ceil(width * sine + height * cosine)


def _resize_for_canvas(
    image: Image.Image, canvas_size: tuple[int, int], max_pixels: int
) -> Image.Image:
    canvas_pixels = math.prod(canvas_size)
    if canvas_pixels <= max_pixels:
        return image
    scale = min(math.sqrt(max_pixels / canvas_pixels), max_pixels / max(image.size))
    size = tuple(max(1, int(dimension * scale)) for dimension in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _rotate(image: Image.Image, rotation: float) -> Image.Image:
    angle = rotation % 360
    if angle == 0:
        return image

    width, height = image.size
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), -angle, 1)
    output_width, output_height = _rotated_size(image.size, rotation)
    matrix[0, 2] += (output_width - width) / 2
    matrix[1, 2] += (output_height - height) / 2
    pixels = cv2.warpAffine(
        np.asarray(image),
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(pixels)


def pad_region_boxes(
    regions: list[tuple[str, list[float]]],
    image_size: tuple[int, int],
    ratio: float,
) -> list[list[float]]:
    """Expand original-image boxes with equal pixel margins and neighbor limits.

    Margins are a fraction of each box's shorter pixel dimension. Different
    questions sharing a row or column split their original gap at its midpoint.
    Diagonal neighbors are limited on one axis only if their padding intersects.
    Existing overlap is retained without enlarging its intersection; regions
    belonging to the same question do not constrain one another. Input boxes
    remain unchanged, and returned boxes retain their input order.
    """
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("padding ratio must be finite and nonnegative")
    width, height = image_size
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError("image dimensions must be finite and positive")
    for _, bbox in regions:
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("bbox must have ordered, nonempty bounds within [0, 1]")
    if ratio == 0:
        return [list(bbox) for _, bbox in regions]

    padded = []
    for question_id, (x0, y0, x1, y1) in regions:
        margin = ratio * min((x1 - x0) * width, (y1 - y0) * height)
        dx, dy = margin / width, margin / height
        box = [max(0, x0 - dx), max(0, y0 - dy), min(1, x1 + dx), min(1, y1 + dy)]
        for neighbor_id, (nx0, ny0, nx1, ny1) in regions:
            if neighbor_id == question_id:
                continue
            overlap_x = x0 < nx1 and nx0 < x1
            overlap_y = y0 < ny1 and ny0 < y1
            if overlap_y:
                if nx1 <= x0:
                    box[0] = max(box[0], (nx1 + x0) / 2)
                elif x1 <= nx0:
                    box[2] = min(box[2], (x1 + nx0) / 2)
            if overlap_x:
                if ny1 <= y0:
                    box[1] = max(box[1], (ny1 + y0) / 2)
                elif y1 <= ny0:
                    box[3] = min(box[3], (y1 + ny0) / 2)
            if overlap_x and overlap_y:
                # An edge inside the neighbor must not add new shared pixels.
                if nx0 <= x0:
                    box[0] = x0
                if ny0 <= y0:
                    box[1] = y0
                if x1 <= nx1:
                    box[2] = x1
                if y1 <= ny1:
                    box[3] = y1
        padded.append(box)

    for index, (question_id, original) in enumerate(regions):
        for neighbor_index in range(index + 1, len(regions)):
            neighbor_id, neighbor = regions[neighbor_index]
            if neighbor_id == question_id:
                continue
            x_gap = max(neighbor[0] - original[2], original[0] - neighbor[2])
            y_gap = max(neighbor[1] - original[3], original[1] - neighbor[3])
            if x_gap < 0 or y_gap < 0:
                continue
            box, other = padded[index], padded[neighbor_index]
            if not (
                box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]
            ):
                continue
            # Prefer a positive horizontal gap, then vertical, then a shared corner.
            axis = 0 if x_gap > 0 or y_gap == 0 else 1
            before, after = index, neighbor_index
            if original[axis] > neighbor[axis]:
                before, after = after, before
            midpoint = (regions[before][1][axis + 2] + regions[after][1][axis]) / 2
            padded[before][axis + 2] = min(padded[before][axis + 2], midpoint)
            padded[after][axis] = max(padded[after][axis], midpoint)
    return padded


def crop_and_rotate(
    data: bytes,
    bbox: tuple[float, float, float, float] | None,
    rotation: float,
    max_pixels: int,
) -> bytes:
    """Crop normalized original-image bounds, scale, then rotate clockwise.

    Fractional crop edges include every intersected pixel. Rotation expands onto
    a white canvas. Scaling uses its predicted size before allocating the affine
    image, and a final resize accounts for integer rounding of the canvas edges.
    """
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    if not math.isfinite(rotation):
        raise ValueError("rotation must be finite")
    _, image = _decode(data)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("bbox must have ordered, nonempty bounds within [0, 1]")
        width, height = image.size
        image = image.crop(
            (
                math.floor(x0 * width),
                math.floor(y0 * height),
                math.ceil(x1 * width),
                math.ceil(y1 * height),
            )
        )
    image = _resize_for_canvas(image, _rotated_size(image.size, rotation), max_pixels)
    image = _rotate(image, rotation)
    return _png(_resize_for_canvas(image, image.size, max_pixels))


class PaddleOrientation:
    """Optional local classifier returning a clockwise correction in degrees."""

    def __init__(self) -> None:
        from paddleocr import DocImgOrientationClassification

        self.model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")

    def angle(self, data: bytes) -> int:
        _, image = _decode(data)
        pixels = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        prediction = self.model.predict(pixels, batch_size=1)[0]
        # Paddle corrects its orientation label counterclockwise.
        return -int(prediction["label_names"][0]) % 360
