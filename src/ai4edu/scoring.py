import hashlib
from decimal import Decimal

from ai4edu.schemas import Grade, Judgment, Rubric, Transcript


def validate_rubric(
    rubric: Rubric,
    question_id: str,
    max_score: Decimal,
    step: Decimal,
    *,
    require_total: bool = True,
):
    if rubric.question_id != question_id or rubric.max_score != max_score:
        raise ValueError("评分细则题号或满分与题目不一致")
    ids = [c.id for c in rubric.criteria]
    if len(ids) != len(set(ids)):
        raise ValueError("评分点编号重复")
    if require_total and sum(c.max_score for c in rubric.criteria) != max_score:
        raise ValueError("评分点分值合计不等于题目满分")
    if any(c.max_score % step for c in rubric.criteria):
        raise ValueError("评分点分值不符合评分精度")


def validate_transcript(transcript: Transcript, image_ids: set[str]):
    ids = [line.id for line in transcript.lines]
    if len(ids) != len(set(ids)):
        raise ValueError("转写证据编号重复")
    if any(line.image_id not in image_ids for line in transcript.lines):
        raise ValueError("转写引用了不存在的作答图片")
    if transcript.is_blank and (
        transcript.issues
        or any(
            line.status == "unclear"
            or line.issues
            or (line.status == "clear" and line.text.strip())
            for line in transcript.lines
        )
    ):
        raise ValueError("空白作答标记与转写内容冲突")


def finalize_grade(
    rubric: Rubric, transcript: Transcript, judgment: Judgment, step: Decimal
) -> Grade:
    expected = {c.id: c for c in rubric.criteria}
    returned = [s.criterion_id for s in judgment.scores]
    if len(returned) != len(set(returned)) or set(returned) != expected.keys():
        raise ValueError("评分点必须完整覆盖且不能重复")
    evidence = {line.id: line for line in transcript.lines}
    for item in judgment.scores:
        if not set(item.evidence_ids) <= evidence.keys():
            raise ValueError("评分引用了不存在的证据")
        if item.score is not None:
            if item.score > expected[item.criterion_id].max_score:
                raise ValueError("评分点分值超过满分")
            if item.score % step:
                raise ValueError("得分不符合评分精度")
            if item.score > 0 and not item.evidence_ids:
                raise ValueError("正向给分必须引用学生作答证据")
            if item.score > 0 and any(evidence[e].status != "clear" for e in item.evidence_ids):
                raise ValueError("不能根据划掉或不清楚的证据确定给分")
    unresolved = any(s.score is None for s in judgment.scores)
    return Grade(
        status="ungradable" if unresolved else "scored",
        scores=judgment.scores,
        total_score=None if unresolved else sum((s.score for s in judgment.scores), Decimal(0)),
        issues=judgment.issues,
    )


def needs_verification(
    transcript: Transcript, grade: Grade, unit_id: str, audit_rate: float
) -> bool:
    sample = int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16) / 2**32
    return (
        grade.status == "ungradable"
        or bool(grade.issues or transcript.issues)
        or any(
            line.status == "unclear" or line.alternatives or line.issues
            for line in transcript.lines
        )
        or sample < audit_rate
    )
