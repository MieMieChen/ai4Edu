from __future__ import annotations

import copy
import io
import math

import numpy as np
import pytest
from PIL import Image

from ai4edu import images


def encoded(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decoded(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image).copy()


def test_padding_recovers_clipped_letter_without_including_neighbor_ink():
    page = Image.new("RGB", (1000, 1000), "white")
    # The detected left edge clips the upright of this L and part of its foot.
    page.paste("black", (666, 200, 670, 241))
    page.paste("black", (666, 237, 699, 241))
    page.paste("red", (647, 200, 655, 241))
    regions = [
        ("q18", [0.345, 0.065, 0.655, 0.945]),
        ("q19", [0.675, 0.065, 0.975, 0.945]),
    ]
    data = encoded(page)
    clipped = decoded(images.crop_and_rotate(data, tuple(regions[1][1]), 0, 1_000_000))

    boxes = images.pad_region_boxes(regions, page.size, 0.04)
    recovered = decoded(images.crop_and_rotate(data, tuple(boxes[1]), 0, 1_000_000))

    expected_ink = np.all(np.asarray(page) == 0, axis=2).sum()
    assert np.all(clipped == 0, axis=2).sum() < expected_ink
    assert np.all(recovered == 0, axis=2).sum() == expected_ink
    assert not np.any(np.all(recovered == (255, 0, 0), axis=2))
    assert boxes[1][0] == pytest.approx(0.665)


@pytest.mark.parametrize(
    ("size", "bbox", "expected"),
    [
        ((1000, 2000), [0.2, 0.3, 0.6, 0.6], [0.16, 0.28, 0.64, 0.62]),
        ((1000, 500), [0.2, 0.3, 0.8, 0.7], [0.18, 0.26, 0.82, 0.74]),
    ],
)
def test_padding_uses_the_shorter_pixel_dimension(size, bbox, expected):
    assert images.pad_region_boxes([("q1", bbox)], size, 0.1)[0] == pytest.approx(expected)


@pytest.mark.parametrize("gap", [0, 0.02])
def test_horizontal_neighbors_share_the_original_gap_midpoint(gap):
    regions = [
        ("left", [0.1, 0.2, 0.4, 0.8]),
        ("right", [0.4 + gap, 0.3, 0.7, 0.9]),
    ]

    left, right = images.pad_region_boxes(regions, (1000, 1000), 0.2)

    assert left[2] == pytest.approx(0.4 + gap / 2)
    assert right[0] == pytest.approx(0.4 + gap / 2)
    assert left[0] < regions[0][1][0]
    assert right[2] > regions[1][1][2]


def test_nearest_neighbors_bound_padding_independently_of_input_order():
    regions = [
        ("middle", [0.35, 0.2, 0.55, 0.8]),
        ("far_left", [0, 0.2, 0.1, 0.8]),
        ("right", [0.65, 0.2, 0.75, 0.8]),
        ("near_left", [0.15, 0.2, 0.25, 0.8]),
    ]

    padded = images.pad_region_boxes(regions, (1000, 1000), 1)
    reversed_padded = images.pad_region_boxes(list(reversed(regions)), (1000, 1000), 1)

    assert padded[0][0] == pytest.approx(0.3)
    assert padded[0][2] == pytest.approx(0.6)
    assert padded == list(reversed(reversed_padded))


def test_vertical_neighbors_share_the_original_gap_midpoint():
    regions = [
        ("top", [0.1, 0.05, 0.9, 0.45]),
        ("bottom", [0.2, 0.47, 0.8, 0.9]),
    ]

    top, bottom = images.pad_region_boxes(regions, (1000, 1000), 0.2)

    assert top[3] == pytest.approx(0.46)
    assert bottom[1] == pytest.approx(0.46)
    assert top[1] < regions[0][1][1]
    assert bottom[3] > regions[1][1][3]


@pytest.mark.parametrize(
    ("first", "second", "axis"),
    [
        ([0.1, 0.1, 0.4, 0.4], [0.42, 0.42, 0.7, 0.7], 0),
        ([0.42, 0.42, 0.7, 0.7], [0.1, 0.1, 0.4, 0.4], 0),
        ([0.42, 0.1, 0.7, 0.4], [0.1, 0.42, 0.4, 0.7], 0),
        ([0.1, 0.1, 0.4, 0.4], [0.4, 0.42, 0.7, 0.7], 1),
        ([0.1, 0.1, 0.4, 0.4], [0.4, 0.4, 0.7, 0.7], 0),
    ],
)
def test_colliding_diagonal_padding_stops_at_one_original_gap_midpoint(first, second, axis):
    regions = [("q1", first), ("q2", second)]
    unrestricted = [images.pad_region_boxes([region], (1000, 1000), 0.1)[0] for region in regions]

    padded = images.pad_region_boxes(regions, (1000, 1000), 0.1)

    before, after = sorted(range(2), key=lambda index: regions[index][1][axis])
    midpoint = (regions[before][1][axis + 2] + regions[after][1][axis]) / 2
    assert padded[before][axis + 2] == pytest.approx(midpoint)
    assert padded[after][axis] == pytest.approx(midpoint)
    other_axis = 1 - axis
    for limited, original_padding in zip(padded, unrestricted, strict=True):
        assert limited[other_axis] == original_padding[other_axis]
        assert limited[other_axis + 2] == original_padding[other_axis + 2]


@pytest.mark.parametrize("second", [[0.42, 0.6, 0.7, 0.9], [0.6, 0.42, 0.9, 0.7]])
def test_noncolliding_diagonal_neighbors_do_not_reduce_padding(second):
    regions = [("q1", [0.1, 0.1, 0.4, 0.4]), ("q2", second)]

    padded = images.pad_region_boxes(regions, (1000, 1000), 0.1)

    assert padded == [images.pad_region_boxes([region], (1000, 1000), 0.1)[0] for region in regions]


def test_same_question_diagonal_regions_can_still_overlap():
    regions = [("q1", [0.1, 0.1, 0.4, 0.4]), ("q1", [0.42, 0.42, 0.7, 0.7])]

    padded = images.pad_region_boxes(regions, (1000, 1000), 0.1)

    assert padded == [images.pad_region_boxes([region], (1000, 1000), 0.1)[0] for region in regions]
    assert padded[0][2] > padded[1][0]
    assert padded[0][3] > padded[1][1]


def test_diagonal_limits_retain_existing_overlap_with_a_third_question():
    regions = [
        ("q1", [0.1, 0.1, 0.4, 0.4]),
        ("q2", [0.42, 0.42, 0.7, 0.7]),
        ("q3", [0.3, 0.2, 0.5, 0.4]),
    ]

    first, diagonal, overlapping = images.pad_region_boxes(regions, (1000, 1000), 0.1)

    assert first[2] <= diagonal[0] or first[3] <= diagonal[1]
    assert [
        max(first[0], overlapping[0]),
        max(first[1], overlapping[1]),
        min(first[2], overlapping[2]),
        min(first[3], overlapping[3]),
    ] == [0.3, 0.2, 0.4, 0.4]


def test_same_question_regions_do_not_restrict_each_other():
    regions = [
        ("q1", [0.2, 0.2, 0.4, 0.8]),
        ("q1", [0.41, 0.2, 0.7, 0.8]),
    ]

    padded = images.pad_region_boxes(regions, (1000, 1000), 0.25)

    assert padded == [
        images.pad_region_boxes([region], (1000, 1000), 0.25)[0] for region in regions
    ]
    assert padded[0][2] > padded[1][0]


@pytest.mark.parametrize(
    "boxes",
    [
        [[0.1, 0.1, 0.55, 0.6], [0.45, 0.4, 0.9, 0.9]],
        [[0.1, 0.1, 0.9, 0.9], [0.3, 0.3, 0.7, 0.7]],
        [[0.1, 0.1, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9]],
    ],
)
def test_existing_overlap_is_preserved_without_growing_it(boxes):
    padded = images.pad_region_boxes([("a", boxes[0]), ("b", boxes[1])], (1000, 1000), 0.2)

    def intersection(pair):
        a, b = pair
        return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]

    assert intersection(padded) == intersection(boxes)
    for original, expanded in zip(boxes, padded, strict=True):
        assert expanded[0] <= original[0] < original[2] <= expanded[2]
        assert expanded[1] <= original[1] < original[3] <= expanded[3]


def test_padding_stops_at_image_bounds():
    assert images.pad_region_boxes([("q1", [0.001, 0.002, 0.999, 0.998])], (1000, 500), 1) == [
        [0, 0, 1, 1]
    ]


@pytest.mark.parametrize("ratio", [0, 0.04])
def test_input_is_not_mutated_or_aliased(ratio):
    regions = [("q2", [0.7, 0.2, 0.9, 0.8]), ("q1", [0.1, 0.2, 0.4, 0.8])]
    original = copy.deepcopy(regions)

    padded = images.pad_region_boxes(regions, (1000, 1000), ratio)

    assert regions == original
    if ratio == 0:
        assert padded == [box for _, box in original]
    assert all(result is not source for result, (_, source) in zip(padded, regions, strict=True))
    padded[0][0] = 0
    assert regions == original


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_padding_stays_in_original_coordinates_before_rotation(rotation):
    pixels = np.full((40, 80, 3), 255, dtype=np.uint8)
    pixels[8:16, 37:42] = 0
    regions = [("q1", [0.5, 0.2, 0.8, 0.6])]

    padded = images.pad_region_boxes(regions, (80, 40), 0.25)[0]
    result = decoded(
        images.crop_and_rotate(encoded(Image.fromarray(pixels)), tuple(padded), rotation, 10_000)
    )

    assert padded == pytest.approx([0.45, 0.1, 0.85, 0.7])
    np.testing.assert_array_equal(result, np.rot90(pixels[4:28, 36:68], -(rotation // 90)))


def test_empty_regions_return_an_empty_list():
    assert images.pad_region_boxes([], (1000, 500), 0.04) == []


@pytest.mark.parametrize("ratio", [-0.01, math.nan, math.inf])
def test_invalid_padding_ratios_raise(ratio):
    with pytest.raises(ValueError):
        images.pad_region_boxes([("q1", [0.1, 0.2, 0.9, 0.8])], (1000, 500), ratio)


@pytest.mark.parametrize("size", [(0, 100), (100, 0), (-1, 100)])
def test_nonpositive_image_dimensions_raise(size):
    with pytest.raises(ValueError):
        images.pad_region_boxes([], size, 0.04)


@pytest.mark.parametrize(
    "bbox",
    [
        [-0.1, 0, 1, 1],
        [0, 0, 1.1, 1],
        [0.5, 0, 0.5, 1],
        [0, 0.8, 1, 0.2],
        [0, 0, math.nan, 1],
        [0, 0, math.inf, 1],
        [0, 0, 1],
    ],
)
def test_invalid_region_bounds_raise_even_with_zero_padding(bbox):
    with pytest.raises(ValueError):
        images.pad_region_boxes([("q1", bbox)], (1000, 500), 0)
