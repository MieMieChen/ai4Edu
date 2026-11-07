"""Validate fixed answer-sheet frame identities before preparing or grading answers."""

from ai4edu.schemas import AnswerTemplate, TemplatePage


def validate_box_ids(
    template: AnswerTemplate, question_id: str, box_ids: list[str] | None
) -> None:
    expected = {region["box_id"] for region in template.regions({question_id})}
    if not expected or box_ids is None or len(box_ids) != len(expected) or set(box_ids) != expected:
        raise ValueError(f"题目 {question_id} 的答题框必须完整提供，不能缺失、重复或包含未知答题框")


def ensure_complete_regions(
    template: AnswerTemplate, question_id: str, regions: list[dict]
) -> None:
    box_ids = [region.get("box_id") for region in regions]
    if any(not isinstance(box_id, str) for box_id in box_ids):
        raise ValueError(f"题目 {question_id} 的答案区域缺少有效答题框编号")
    validate_box_ids(template, question_id, box_ids)


def validate_page_size(page: TemplatePage, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("答题卡页面尺寸必须为正数")
    # Compare integer products so that a difference of exactly 2% remains allowed.
    if abs(width * page.height - height * page.width) * 100 > 2 * height * page.width:
        raise ValueError("答题卡页面宽高比例与模板不符，偏差不能超过 2%")
