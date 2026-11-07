import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from ai4edu import schemas
from ai4edu.schemas import AnswerInput, ExamInput


def box(box_id="q1-a", question_id="q1", order=0, bbox=(0, 0, 1, 0.4), **extra):
    return {"id": box_id, "question_id": question_id, "order": order, "bbox": bbox} | extra


def template_data():
    return {
        "pages": [
            {
                "width": 1000,
                "height": 1500,
                "boxes": [
                    box("q2-a", "q2", bbox=(0, 0.6, 1, 1)),
                    box(subquestion_id="a"),
                ],
            },
            {
                "width": 1000,
                "height": 1500,
                "boxes": [box("q1-b", order=1, subquestion_id="a")],
            },
            {"width": 1000, "height": 1500},
        ]
    }


def exam_data():
    return {
        "title": "fixed answer sheet",
        "questions": [
            {"id": qid, "stem": "question", "reference": "solution", "max_score": 4}
            for qid in ["q1", "q2"]
        ],
        "answer_template": template_data(),
    }


def test_exam_accepts_multiple_boxes_and_same_subquestion_across_pages():
    exam = ExamInput.model_validate(exam_data())
    assert exam.answer_template.pages[1].boxes[0].subquestion_id == "a"
    assert exam.answer_template.pages[2].boxes == []


@pytest.mark.parametrize("bbox", [
    (-0.1, 0, 1, 1), (0, 0, 1.1, 1), (0, -0.1, 1, 1), (0, 0, 1, 1.1),
    (0.5, 0, 0.5, 1), (0.8, 0, 0.2, 1), (0, 0.5, 1, 0.5),
    (0, 0.8, 1, 0.2), (0, 0, float("nan"), 1), (0, 0, 1, float("inf")),
])
def test_boxes_reject_invalid_normalized_geometry(bbox):
    with pytest.raises(ValidationError):
        schemas.AnswerBox.model_validate(box(bbox=bbox))


@pytest.mark.parametrize("change", [
    {"id": ""}, {"id": "a" * 65}, {"question_id": ""},
    {"question_id": "a" * 65}, {"subquestion_id": ""}, {"order": -1},
])
def test_boxes_reject_invalid_identifiers_or_order(change):
    with pytest.raises(ValidationError):
        schemas.AnswerBox.model_validate(box() | change)


@pytest.mark.parametrize("change", [{"width": 0}, {"height": 0}, {"height": -1}])
def test_template_pages_require_positive_pixel_dimensions(change):
    with pytest.raises(ValidationError):
        schemas.TemplatePage.model_validate({"width": 1000, "height": 1500} | change)


@pytest.mark.parametrize("value", [{"pages": []}, {"pages": [{"width": 1, "height": 1}]}])
def test_templates_require_at_least_one_page_and_one_box(value):
    with pytest.raises(ValidationError):
        schemas.AnswerTemplate.model_validate(value)


@pytest.mark.parametrize("field,value", [("id", "q1-a"), ("order", 0)])
def test_template_rejects_duplicate_box_ids_or_question_order_across_pages(field, value):
    data = template_data()
    data["pages"][1]["boxes"][0][field] = value
    with pytest.raises(ValidationError):
        schemas.AnswerTemplate.model_validate(data)


def test_template_rejects_positive_area_overlap_on_same_page():
    data = template_data()
    data["pages"][0]["boxes"][0]["bbox"] = [0, 0.3, 1, 1]
    with pytest.raises(ValidationError, match="重叠"):
        schemas.AnswerTemplate.model_validate(data)


def test_template_allows_touching_edges_and_reused_coordinates_on_other_pages():
    data = template_data()
    data["pages"][0]["boxes"][0]["bbox"] = [0, 0.4, 1, 1]
    assert len(schemas.AnswerTemplate.model_validate(data).pages) == 3


@pytest.mark.parametrize("question_ids", [{"q1"}, {"q1", "q2", "q3"}])
def test_validate_questions_requires_exact_exam_coverage(question_ids):
    template = schemas.AnswerTemplate.model_validate(template_data())
    with pytest.raises(ValueError, match="题"):
        template.validate_questions(question_ids)


def test_exam_rejects_template_that_omits_a_question():
    data = exam_data()
    data["answer_template"]["pages"][0]["boxes"] = [box()]
    with pytest.raises(ValidationError, match="题"):
        ExamInput.model_validate(data)


def test_exam_rejects_template_with_unknown_question():
    data = exam_data()
    data["answer_template"]["pages"][0]["boxes"][0]["question_id"] = "unknown"
    with pytest.raises(ValidationError, match="题"):
        ExamInput.model_validate(data)


def test_exam_without_template_and_answer_without_box_ids_remain_valid():
    data = exam_data()
    del data["answer_template"]
    assert ExamInput.model_validate(data).answer_template is None
    assert AnswerInput(question_id="q1", assets=["asset"]).box_ids is None


def test_template_regions_are_sorted_with_page_dimensions_and_question_filter():
    template = schemas.AnswerTemplate.model_validate(template_data())
    regions = template.regions()
    assert regions == [
        {
            "box_id": "q1-a", "question_id": "q1", "subquestion_id": "a",
            "bbox": [0.0, 0.0, 1.0, 0.4], "order": 0, "page_index": 0,
            "page_width": 1000, "page_height": 1500,
        },
        {
            "box_id": "q1-b", "question_id": "q1", "subquestion_id": "a",
            "bbox": [0.0, 0.0, 1.0, 0.4], "order": 1, "page_index": 1,
            "page_width": 1000, "page_height": 1500,
        },
        {
            "box_id": "q2-a", "question_id": "q2", "subquestion_id": None,
            "bbox": [0.0, 0.6, 1.0, 1.0], "order": 0, "page_index": 0,
            "page_width": 1000, "page_height": 1500,
        },
    ]
    assert template.regions({"q1"}) == regions[:2]
    assert template.regions(set()) == []


def test_template_version_is_canonical_sha256_and_changes_with_geometry():
    template = schemas.AnswerTemplate.model_validate(template_data())
    canonical = json.dumps(template.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert template.version == hashlib.sha256(canonical.encode()).hexdigest()
    reordered = json.loads(json.dumps(template_data(), sort_keys=True))
    assert schemas.AnswerTemplate.model_validate(reordered).version == template.version
    changed = deepcopy(template_data())
    changed["pages"][1]["boxes"][0]["bbox"] = [0, 0, 1, 0.5]
    assert schemas.AnswerTemplate.model_validate(changed).version != template.version


@pytest.mark.parametrize("box_ids", [[], ["q1-a"], ["q1-a", "q1-a"]])
def test_answer_box_ids_require_nonempty_unique_one_per_asset(box_ids):
    with pytest.raises(ValidationError):
        AnswerInput(question_id="q1", assets=["a", "b"], box_ids=box_ids)


def test_answer_assets_are_still_required_with_box_ids():
    with pytest.raises(ValidationError):
        AnswerInput.model_validate({"question_id": "q1", "box_ids": ["q1-a"]})


def test_answer_accepts_explicit_box_ids_in_asset_order():
    answer = AnswerInput(question_id="q1", assets=["b", "a"], box_ids=["q1-b", "q1-a"])
    assert answer.box_ids == ["q1-b", "q1-a"]


@pytest.mark.parametrize("box_ids", [None, [], ["q1-a"], ["q1-a", "q1-a"], ["q1-a", "q2-a"], ["q1-a", "q1-b", "q1-b"]])
def test_box_id_validation_rejects_missing_duplicate_or_foreign_frames(box_ids):
    from ai4edu.answer_templates import validate_box_ids

    template = schemas.AnswerTemplate.model_validate(template_data())
    with pytest.raises(ValueError, match="答题框"):
        validate_box_ids(template, "q1", box_ids)


def test_box_id_validation_accepts_complete_frames_in_any_order():
    from ai4edu.answer_templates import validate_box_ids

    template = schemas.AnswerTemplate.model_validate(template_data())
    validate_box_ids(template, "q1", ["q1-b", "q1-a"])


@pytest.mark.parametrize("regions", [[], [{}], [{"box_id": "q1-a"}], [{"box_id": "q1-a"}, {"box_id": "q1-a"}], [{"box_id": "q1-a"}, {"box_id": "q2-a"}], [{"box_id": "q1-a"}, {"box_id": "q1-b"}, {"box_id": "q1-b"}]])
def test_completeness_rejects_missing_duplicate_or_foreign_regions(regions):
    from ai4edu.answer_templates import ensure_complete_regions

    template = schemas.AnswerTemplate.model_validate(template_data())
    with pytest.raises(ValueError, match="答题框"):
        ensure_complete_regions(template, "q1", regions)


def test_completeness_accepts_all_regions_and_rejects_zero_expected_boxes():
    from ai4edu.answer_templates import ensure_complete_regions, validate_box_ids

    template = schemas.AnswerTemplate.model_validate(template_data())
    ensure_complete_regions(template, "q1", list(reversed(template.regions({"q1"}))))
    with pytest.raises(ValueError, match="答题框"):
        ensure_complete_regions(template, "unknown", [])
    with pytest.raises(ValueError, match="答题框"):
        validate_box_ids(template, "unknown", [])


@pytest.mark.parametrize("size", [(1000, 1500), (2000, 3000), (1020, 1500), (980, 1500), (1019, 1500), (981, 1500)])
def test_page_size_accepts_rescaling_and_small_relative_aspect_variation(size):
    from ai4edu.answer_templates import validate_page_size

    page = schemas.TemplatePage(width=1000, height=1500)
    validate_page_size(page, *size)


@pytest.mark.parametrize("size", [(1021, 1500), (979, 1500), (1500, 1000), (0, 1500), (1000, 0)])
def test_page_size_rejects_aspect_mismatch_or_nonpositive_dimensions(size):
    from ai4edu.answer_templates import validate_page_size

    page = schemas.TemplatePage(width=1000, height=1500)
    with pytest.raises(ValueError, match="尺寸|比例"):
        validate_page_size(page, *size)
