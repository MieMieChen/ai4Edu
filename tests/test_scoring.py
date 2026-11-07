from decimal import Decimal

import pytest
from pydantic import ValidationError

from ai4edu.schemas import Criterion, CriterionScore, Judgment, Rubric, Transcript, TranscriptLine
from ai4edu.scoring import finalize_grade, validate_rubric, validate_transcript


@pytest.fixture
def rubric():
    return Rubric(
        question_id="15",
        max_score=4,
        solution="有效证明",
        corrections=[],
        criteria=[
            Criterion(id="R1", description="建立关系", max_score=2, rules="允许等价解法"),
            Criterion(id="R2", description="完成证明", max_score=2, rules="按有效推理给分"),
        ],
    )


@pytest.fixture
def transcript():
    return Transcript(
        lines=[
            TranscriptLine(id="L1", image_id="image-1", text="BD perpendicular to PE"),
            TranscriptLine(id="L2", image_id="image-1", text="BD perpendicular to plane PEC"),
        ]
    )


def judgment(score=1):
    return Judgment(
        scores=[
            CriterionScore(criterion_id="R1", score=2, evidence_ids=["L1"], reason="关系成立"),
            CriterionScore(
                criterion_id="R2", score=score, evidence_ids=["L2"], reason="论证部分完成"
            ),
        ]
    )


def test_computes_total_from_criteria(rubric, transcript):
    result = finalize_grade(rubric, transcript, judgment(), Decimal("0.5"))
    assert result.total_score == Decimal(3)
    assert result.status == "scored"


def test_per_line_issues_are_preserved_and_trigger_independent_read(rubric):
    from ai4edu.scoring import needs_verification

    transcript = Transcript.model_validate(
        {
            "lines": [
                {
                    "id": "L1",
                    "image_id": "image-1",
                    "text": "x=1",
                    "issues": ["a nearby symbol is ambiguous"],
                },
                {"id": "L2", "image_id": "image-1", "text": "therefore"},
            ]
        }
    )
    validate_transcript(transcript, {"image-1"})
    grade = finalize_grade(rubric, transcript, judgment(), Decimal("0.5"))
    assert transcript.lines[0].issues == ["a nearby symbol is ambiguous"]
    assert needs_verification(transcript, grade, "stable-sample", 0) is True


def test_blank_cannot_hide_per_line_recognition_issues():
    transcript = Transcript.model_validate(
        {
            "is_blank": True,
            "lines": [{"id": "L1", "image_id": "image-1", "text": "", "issues": ["unreadable"]}],
        }
    )
    with pytest.raises(ValueError, match="空白"):
        validate_transcript(transcript, {"image-1"})


def test_missing_criterion_is_an_error(rubric, transcript):
    incomplete = judgment()
    incomplete.scores.pop()
    with pytest.raises(ValueError, match="评分点"):
        finalize_grade(rubric, transcript, incomplete, Decimal("0.5"))


def test_nonexistent_evidence_is_an_error(rubric, transcript):
    value = judgment()
    value.scores[0].evidence_ids = ["invented"]
    with pytest.raises(ValueError, match="证据"):
        finalize_grade(rubric, transcript, value, Decimal("0.5"))


def test_over_maximum_is_rejected_not_clamped(rubric, transcript):
    with pytest.raises(ValueError, match="分值"):
        finalize_grade(rubric, transcript, judgment(3), Decimal("0.5"))


def test_scores_obey_point_increment(rubric, transcript):
    with pytest.raises(ValueError, match="精度"):
        finalize_grade(rubric, transcript, judgment("1.3"), Decimal("0.5"))


def test_unreadable_is_null_not_zero(rubric, transcript):
    result = finalize_grade(rubric, transcript, judgment(None), Decimal("0.5"))
    assert result.status == "ungradable"
    assert result.total_score is None


def test_rubric_maximum_must_match_question(rubric):
    with pytest.raises(ValueError, match="满分"):
        validate_rubric(rubric, "15", Decimal(5), Decimal("0.5"))


def test_rubric_points_must_sum_to_maximum(rubric):
    rubric.criteria[0].max_score = Decimal(3)
    with pytest.raises(ValueError, match="合计"):
        validate_rubric(rubric, "15", Decimal(4), Decimal("0.5"))


def test_duplicate_criterion_ids_are_rejected(rubric):
    rubric.criteria[1].id = "R1"
    with pytest.raises(ValueError, match="重复"):
        validate_rubric(rubric, "15", Decimal(4), Decimal("0.5"))


def test_negative_points_fail_at_input_boundary():
    with pytest.raises(ValidationError):
        CriterionScore(criterion_id="R1", score=-1, evidence_ids=[], reason="invalid")


def test_unclear_blank_cannot_be_accepted_as_zero_score():
    transcript = Transcript(
        is_blank=True,
        lines=[
            TranscriptLine(id="L1", image_id="image-1", text="", status="unclear"),
        ],
    )
    with pytest.raises(ValueError, match="空白"):
        validate_transcript(transcript, {"image-1"})
