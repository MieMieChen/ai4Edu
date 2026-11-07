import hashlib
import json
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AnswerBox(Schema):
    id: str = Field(min_length=1, max_length=64)
    question_id: str = Field(min_length=1, max_length=64)
    subquestion_id: str | None = Field(default=None, min_length=1)
    bbox: tuple[float, float, float, float]
    order: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_box(self):
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("答题框坐标必须为原图上的有效归一化矩形")
        return self


class TemplatePage(Schema):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    boxes: list[AnswerBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonoverlapping_boxes(self):
        for index, first in enumerate(self.boxes):
            x0, y0, x1, y1 = first.bbox
            for second in self.boxes[index + 1 :]:
                other_x0, other_y0, other_x1, other_y1 = second.bbox
                if (
                    max(x0, other_x0) < min(x1, other_x1)
                    and max(y0, other_y0) < min(y1, other_y1)
                ):
                    raise ValueError(f"同页答题框不能重叠：{first.id}、{second.id}")
        return self


class AnswerTemplate(Schema):
    pages: list[TemplatePage] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_boxes(self):
        boxes = [box for page in self.pages for box in page.boxes]
        if not boxes:
            raise ValueError("答题卡模板必须包含至少一个答题框")
        if len({box.id for box in boxes}) != len(boxes):
            raise ValueError("答题框编号不能重复")
        if len({(box.question_id, box.order) for box in boxes}) != len(boxes):
            raise ValueError("同一题目的答题框顺序不能重复")
        return self

    def validate_questions(self, question_ids: set[str]) -> None:
        covered = {box.question_id for page in self.pages for box in page.boxes}
        if unknown := covered - question_ids:
            raise ValueError(f"答题卡模板包含未知题号：{', '.join(sorted(unknown))}")
        if missing := question_ids - covered:
            raise ValueError(f"答题卡模板缺少题目的答题框：{', '.join(sorted(missing))}")

    @property
    def version(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def regions(self, question_ids: set[str] | None = None) -> list[dict]:
        regions = [
            {
                "box_id": box.id,
                "question_id": box.question_id,
                "subquestion_id": box.subquestion_id,
                "bbox": list(box.bbox),
                "order": box.order,
                "page_index": page_index,
                "page_width": page.width,
                "page_height": page.height,
            }
            for page_index, page in enumerate(self.pages)
            for box in page.boxes
            if question_ids is None or box.question_id in question_ids
        ]
        return sorted(regions, key=lambda region: (region["question_id"], region["order"]))


class Question(Schema):
    id: str = Field(min_length=1, max_length=64)
    stem: str = ""
    question_assets: list[str] = Field(default_factory=list)
    reference: str = ""
    reference_assets: list[str] = Field(default_factory=list)
    max_score: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def has_material(self):
        if not (self.stem or self.question_assets):
            raise ValueError("题目必须提供文字或图片")
        if not (self.reference or self.reference_assets):
            raise ValueError("参考解答必须提供文字或图片")
        return self


class ExamInput(Schema):
    title: str = Field(min_length=1, max_length=200)
    questions: list[Question] = Field(min_length=1)
    score_step: Decimal = Field(default=Decimal("0.5"), gt=0, max_digits=8, decimal_places=3)
    budget_cny: Decimal | None = Field(default=None, gt=0, max_digits=16, decimal_places=8)
    answer_template: AnswerTemplate | None = None

    @model_validator(mode="after")
    def unique_questions(self):
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("题号重复")
        if any(q.max_score % self.score_step for q in self.questions):
            raise ValueError("题目满分必须符合评分精度")
        if self.answer_template is not None:
            self.answer_template.validate_questions({q.id for q in self.questions})
        return self


class AnswerInput(Schema):
    question_id: str
    assets: list[str] = Field(min_length=1)
    box_ids: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def valid_box_ids(self):
        if self.box_ids is not None:
            if len(self.box_ids) != len(self.assets):
                raise ValueError("答题框编号必须与答案图片一一对应")
            if len(set(self.box_ids)) != len(self.box_ids):
                raise ValueError("答题框编号不能重复")
        return self


class QuestionCandidate(Schema):
    candidate_id: str = Field(min_length=1, max_length=128)
    kind: Literal["question"]
    answers: list[AnswerInput] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_answers(self):
        if len({a.question_id for a in self.answers}) != len(self.answers):
            raise ValueError("同一考生的题号重复")
        return self


class PaperCandidate(Schema):
    candidate_id: str = Field(min_length=1, max_length=128)
    kind: Literal["paper"]
    pages: list[str] = Field(min_length=1)


CandidateInput = Annotated[QuestionCandidate | PaperCandidate, Field(discriminator="kind")]


class JobInput(Schema):
    exam_id: str
    question_ids: list[str] | None = Field(default=None, min_length=1)
    candidates: list[CandidateInput] = Field(min_length=1, max_length=1000)
    budget_cny: Decimal | None = Field(default=None, gt=0, max_digits=16, decimal_places=8)
    strategy: Literal["A", "B", "C"] | None = None

    @model_validator(mode="after")
    def unique_candidates(self):
        if len({c.candidate_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("考生编号重复")
        if self.question_ids is not None and len(set(self.question_ids)) != len(self.question_ids):
            raise ValueError("选题不能重复")
        return self


class Criterion(Schema):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    max_score: Decimal = Field(gt=0)
    rules: str = Field(min_length=1)
    alternatives: list[str] = Field(default_factory=list)


class Rubric(Schema):
    question_id: str
    max_score: Decimal = Field(gt=0)
    solution: str = Field(min_length=1)
    criteria: list[Criterion] = Field(min_length=1)
    corrections: list[str] = Field(default_factory=list)


class TranscriptLine(Schema):
    id: str
    image_id: str
    text: str
    status: Literal["clear", "unclear", "crossed_out"] = "clear"
    alternatives: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class Transcript(Schema):
    lines: list[TranscriptLine]
    is_blank: bool = False
    issues: list[str] = Field(default_factory=list)


class CriterionScore(Schema):
    criterion_id: str
    score: Decimal | None = Field(ge=0)
    evidence_ids: list[str]
    reason: str = Field(min_length=1)


class Judgment(Schema):
    scores: list[CriterionScore] = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)


class Grade(Schema):
    status: Literal["scored", "ungradable"]
    scores: list[CriterionScore]
    total_score: Decimal | None
    issues: list[str]


class Region(Schema):
    question_id: str
    bbox: tuple[float, float, float, float]
    order: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_box(self):
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("裁切坐标必须为原图上的有效归一化矩形")
        return self


class PageLayout(Schema):
    rotation: float = Field(ge=-180, le=180, description="将原图转正所需的顺时针角度")
    regions: list[Region]
