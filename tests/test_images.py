from __future__ import annotations

import builtins
import io
import math
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image, UnidentifiedImageError

from ai4edu.images import PaddleOrientation, crop_and_rotate, image_info, render_upload


def encoded(image: Image.Image, file_format: str = "PNG", **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=file_format, **options)
    return buffer.getvalue()


def decoded(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.copy()


@pytest.mark.parametrize(
    ("file_format", "mime"),
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_image_info_reports_actual_format_and_dimensions(file_format, mime):
    data = encoded(Image.new("RGB", (60, 40), "white"), file_format)

    assert image_info(data) == (mime, 60, 40)


@pytest.mark.parametrize(
    ("file_format", "mime", "mode"),
    [("PNG", "image/png", "RGBA"), ("JPEG", "image/jpeg", "L"), ("WEBP", "image/webp", "RGB")],
)
def test_render_upload_normalizes_raster_upload_to_rgb_png(file_format, mime, mode):
    data = encoded(Image.new(mode, (60, 40), "white"), file_format)

    pages = render_upload(data, mime)

    assert len(pages) == 1
    with Image.open(io.BytesIO(pages[0])) as page:
        assert page.format == "PNG"
        assert page.mode == "RGB"
        assert page.size == (60, 40)


@pytest.mark.parametrize("mode", ["RGBA", "P"])
def test_transparent_handwriting_is_composited_onto_white(mode):
    if mode == "RGBA":
        image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        ink, translucent_ink = (0, 0, 0, 255), (0, 0, 0, 128)
    else:
        image = Image.new("P", (20, 10), 0)
        image.putpalette([0, 0, 0] * 256)
        image.info["transparency"] = bytes((0, 255, 128))
        ink, translucent_ink = 1, 2
    image.paste(ink, (4, 2, 6, 8))
    image.paste(ink, (4, 6, 12, 8))
    image.putpixel((12, 7), translucent_ink)
    data = encoded(image)
    expected = np.full((10, 20, 3), 255, dtype=np.uint8)
    expected[2:8, 4:6] = 0
    expected[6:8, 4:12] = 0
    expected[7, 12] = 127

    normalized = decoded(render_upload(data, "image/png")[0])
    transformed = decoded(crop_and_rotate(data, None, 0, 1000))

    assert normalized.mode == transformed.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(normalized), expected)
    np.testing.assert_array_equal(np.asarray(transformed), expected)


def test_decode_applies_exif_orientation_before_reporting_or_cropping():
    image = Image.new("RGB", (60, 40), "white")
    image.paste((255, 0, 0), (0, 0, 20, 20))
    exif = Image.Exif()
    exif[274] = 6  # Stored pixels need a clockwise quarter-turn.
    data = encoded(image, "JPEG", exif=exif, quality=100, subsampling=0)

    assert image_info(data) == ("image/jpeg", 40, 60)
    page = decoded(render_upload(data, "image/jpeg")[0])
    assert page.size == (40, 60)
    assert page.getpixel((30, 10))[0] > 240
    assert page.getpixel((30, 10))[1] < 20
    assert page.getexif().get(274) is None

    crop = decoded(crop_and_rotate(data, (0.5, 0, 1, 1 / 3), 0, 10_000))
    assert crop.size == (20, 20)
    assert crop.getpixel((10, 10))[0] > 240
    assert crop.getpixel((10, 10))[1] < 20


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_clockwise_quarter_turns_preserve_actual_marked_pixels(rotation):
    pixels = np.full((5, 7, 3), 255, dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    pixels[0, -1] = (0, 255, 0)
    pixels[-1, 0] = (0, 0, 255)
    pixels[-1, -1] = (0, 0, 0)
    pixels[2, 3] = (255, 0, 255)

    result = decoded(crop_and_rotate(encoded(Image.fromarray(pixels)), None, rotation, 10_000))

    np.testing.assert_array_equal(np.asarray(result), np.rot90(pixels, -(rotation // 90)))


def test_crop_coordinates_refer_to_original_image_before_rotation():
    pixels = np.full((40, 80, 3), 255, dtype=np.uint8)
    pixels[:20, :40] = (255, 0, 0)
    pixels[:20, 40:] = (0, 255, 0)
    pixels[20:, :40] = (0, 0, 255)
    pixels[20:, 40:] = (0, 0, 0)
    pixels[5:10, 45:50] = (255, 0, 255)

    result = decoded(
        crop_and_rotate(encoded(Image.fromarray(pixels)), (0.5, 0, 1, 0.5), 90, 10_000)
    )

    assert result.size == (20, 40)
    np.testing.assert_array_equal(np.asarray(result), np.rot90(pixels[:20, 40:], -1))


@pytest.mark.parametrize("rotation", [12.5, -12.5, 42.7])
def test_arbitrary_rotations_expand_canvas_and_preserve_corner_ink(rotation):
    pixels = np.full((80, 160, 3), 255, dtype=np.uint8)
    pixels[:12, :12] = (0, 0, 0)
    pixels[:12, -12:] = (0, 0, 0)
    pixels[-12:, :12] = (0, 0, 0)
    pixels[-12:, -12:] = (0, 0, 0)

    result = decoded(crop_and_rotate(encoded(Image.fromarray(pixels)), None, rotation, 100_000))

    radians = math.radians(rotation)
    assert result.size == (
        math.ceil(160 * abs(math.cos(radians)) + 80 * abs(math.sin(radians))),
        math.ceil(160 * abs(math.sin(radians)) + 80 * abs(math.cos(radians))),
    )
    original_ink = (255 - pixels.astype(float)).sum()
    rotated_ink = (255 - np.asarray(result).astype(float)).sum()
    assert rotated_ink == pytest.approx(original_ink, rel=0.005)


@pytest.mark.parametrize(
    ("size", "rotation", "max_pixels"),
    [((1200, 800), 0, 120_000), ((1200, 800), 13.5, 120_000), ((2000, 1), 0, 100)],
)
def test_resize_preserves_aspect_ratio_within_pixel_budget(size, rotation, max_pixels):
    data = encoded(Image.new("RGB", size, "white"))
    reference = decoded(crop_and_rotate(data, None, rotation, 10_000_000))

    result = decoded(crop_and_rotate(data, None, rotation, max_pixels))

    assert result.width * result.height <= max_pixels
    assert result.width <= reference.width
    assert result.height <= reference.height
    # Integer dimensions may differ from the exact aspect ratio by one pixel.
    assert (
        abs(result.width - result.height * reference.width / reference.height) <= 1
        or abs(result.height - result.width * reference.height / reference.width) <= 1
    )


@pytest.mark.parametrize("rotation", [13.5, 90])
def test_large_page_is_scaled_for_expanded_canvas_before_affine_warp(monkeypatch, rotation):
    max_pixels = 2_621_440
    data = encoded(Image.new("RGBA", (4096, 2896), "white"))
    original_warp = cv2.warpAffine
    observed = []

    def bounded_warp(pixels, matrix, output_size, **options):
        height, width = pixels.shape[:2]
        observed.append(((width, height), output_size))
        assert width * height <= max_pixels, "Affine input must be scaled before rotation"
        # Ceiling expanded canvas edges can add less than one row and column.
        assert math.prod(output_size) <= max_pixels + sum(output_size)
        return original_warp(pixels, matrix, output_size, **options)

    monkeypatch.setattr(cv2, "warpAffine", bounded_warp)

    result = decoded(crop_and_rotate(data, None, rotation, max_pixels))

    assert len(observed) == 1
    assert result.width * result.height <= max_pixels


@pytest.mark.parametrize(
    "bbox",
    [
        (-0.1, 0, 1, 1),
        (0, 0, 1.1, 1),
        (0, -0.1, 1, 1),
        (0, 0, 1, 1.1),
        (0.8, 0, 0.2, 1),
        (0, 0.8, 1, 0.2),
        (0.5, 0, 0.5, 1),
        (0, 0.5, 1, 0.5),
        (0, 0, float("nan"), 1),
        (0, 0, float("inf"), 1),
        (0, 0, 1),
    ],
)
def test_invalid_crop_bounds_raise_instead_of_being_clamped(bbox):
    data = encoded(Image.new("RGB", (20, 10), "white"))

    with pytest.raises(ValueError):
        crop_and_rotate(data, bbox, 0, 1000)


@pytest.mark.parametrize("max_pixels", [0, -1])
def test_nonpositive_pixel_budgets_raise(max_pixels):
    data = encoded(Image.new("RGB", (20, 10), "white"))

    with pytest.raises(ValueError):
        crop_and_rotate(data, None, 0, max_pixels)


@pytest.mark.parametrize("rotation", [float("nan"), float("inf")])
def test_nonfinite_rotation_raises(rotation):
    data = encoded(Image.new("RGB", (20, 10), "white"))

    with pytest.raises(ValueError):
        crop_and_rotate(data, None, rotation, 1000)


def test_unsupported_declared_or_actual_raster_formats_raise():
    png = encoded(Image.new("RGB", (20, 10), "white"))
    gif = encoded(Image.new("RGB", (20, 10), "white"), "GIF")

    with pytest.raises(ValueError):
        render_upload(png, "image/gif")
    with pytest.raises(ValueError):
        render_upload(gif, "image/png")
    with pytest.raises(ValueError):
        image_info(gif)
    with pytest.raises(ValueError):
        crop_and_rotate(gif, None, 0, 1000)


@pytest.mark.parametrize(
    "operation",
    [
        image_info,
        lambda data: render_upload(data, "image/png"),
        lambda data: crop_and_rotate(data, None, 0, 1000),
    ],
)
def test_invalid_image_bytes_raise(operation):
    with pytest.raises((ValueError, UnidentifiedImageError)):
        operation(b"not an image")


def test_two_page_pdf_renders_each_page_at_150_dpi_in_order():
    with pymupdf.open() as document:
        first = document.new_page(width=72, height=144)
        first.draw_rect(first.rect, color=None, fill=(1, 0, 0))
        second = document.new_page(width=144, height=72)
        second.draw_rect(second.rect, color=None, fill=(0, 0, 1))
        data = document.tobytes()

    pages = render_upload(data, "application/pdf")

    assert len(pages) == 2
    assert image_info(pages[0]) == ("image/png", 150, 300)
    assert image_info(pages[1]) == ("image/png", 300, 150)
    assert decoded(pages[0]).getpixel((75, 150)) == (255, 0, 0)
    assert decoded(pages[1]).getpixel((150, 75)) == (0, 0, 255)


def test_invalid_pdf_raises():
    with pytest.raises(pymupdf.FileDataError):
        render_upload(b"not a PDF", "application/pdf")


def test_optional_paddle_orientation_propagates_missing_dependency(monkeypatch):
    original_import = builtins.__import__

    def without_paddle(name, *args, **kwargs):
        if name == "paddleocr":
            raise ImportError("PaddleOCR is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_paddle)

    with pytest.raises(ImportError, match="PaddleOCR is not installed"):
        PaddleOrientation()


@pytest.mark.parametrize(
    ("label", "correction"), [("0", 0), ("90", 270), ("180", 180), ("270", 90)]
)
def test_paddle_adapter_supplies_bgr_pixels_and_returns_clockwise_correction(
    monkeypatch, label, correction
):
    observed = []

    class OrientationModel:
        def __init__(self, **kwargs):
            pass

        def predict(self, image, batch_size):
            observed.append(image)
            return [{"label_names": [label]}]

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(DocImgOrientationClassification=OrientationModel),
    )
    data = encoded(Image.new("RGB", (20, 10), (255, 0, 0)))

    assert PaddleOrientation().angle(data) == correction
    assert observed[0].shape == (10, 20, 3)
    np.testing.assert_array_equal(observed[0][0, 0], (0, 0, 255))
