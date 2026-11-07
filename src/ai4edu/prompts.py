import json

from ai4edu.schemas import Question, Rubric, Transcript

VERSION = "v1.2"


def rubric_prompt(question: Question, step: str) -> str:
    return (
        "为这道主观题制定自动阅卷评分细则。只使用题目和参考解答，不接收学生分数。"
        "独立检查参考解答的正确性，将笔误及修正记录在 corrections。"
        "评分点必须有稳定唯一的 id，分值总和恰好等于满分，分值精度为 " + step + "。"
        "围绕数学意义和有效论证制定评分点，允许其他正确解法、不同坐标系和等价表达。"
        "rules 必须说明部分得分、计算错误和错误延续如何处理，避免重复扣分。"
        "不要奖励学生没有写出的论证，不按字迹美观给分。\n题目："
        + question.model_dump_json()
        + "\n图片顺序：题目图片，然后参考解答图片。"
    )


def check_rubric_prompt(question: Question, rubric: Rubric, step: str) -> str:
    return (
        "校核并返回完整的最终评分细则。独立验证题意、数学结论和参考解答，修正错误。"
        "保持题号和满分不变，所有评分点分值之和必须等于满分。"
        f"草稿评分点当前合计为 {sum(c.max_score for c in rubric.criteria)}，题目满分为 {question.max_score}。"
        "若两者不同，必须重新分配各评分点上限及其部分得分规则，逐项加总检查后返回；不能修改题目满分。"
        "检查等价解法、部分给分、错误延续规则，评分精度为 "
        + step
        + "。没有教师审核环节，不输出待教师确认状态。\n题目："
        + question.model_dump_json()
        + "\n草稿："
        + rubric.model_dump_json()
    )


def layout_prompt(question_ids: list[str], *, cropped: bool) -> str:
    if cropped:
        return (
            "输入是已经裁好的单题完整作答。判断将图片转正所需的顺时针 rotation 角度，"
            "支持90/180/270度及小角度倾斜，范围[-180,180]。regions 返回空数组，不再次裁切。"
        )
    return (
        "在这一张学生答卷页面上定位指定主观题的实际答题区域。"
        "返回将页面转正所需的顺时针 rotation 角度，范围[-180,180]。"
        "bbox 为输入原图（旋转前）的 [x0,y0,x1,y1] 归一化矩形。"
        "区域必须完整覆盖手写步骤、涂改和相关图形，不包含其他题目、学生姓名或总分。"
        "同一道题可有多块区域，order 表示阅读顺序。页面没有某题时不输出该题区域，"
        "不要臆造位置，也不要输出客观题。指定题号：" + json.dumps(question_ids)
    )


def transcription_prompt() -> str:
    return (
        "忠实转写学生实际保留的解题内容，不评分、不求解、不补写、不纠正。"
        "按图片顺序和阅读顺序为每行或图形标注分配唯一证据 id，关联实际 image_id。"
        "公式使用 LaTeX。明确区分 clear、unclear、crossed_out；"
        "status 仅表示能否清楚读出，不表示数学是否正确；清晰但写错的内容仍标 clear。"
        "只对确实无法辨认的那一行标 unclear，不把整段清楚的推导一起标为 unclear。"
        "模糊符号记录 alternatives，影响理解的问题写入 issues。"
        "逐行问题放在线的 issues，跨行问题放在顶层 issues；不要输出 Schema 以外的字段。"
        "只记录图形明确标注，不从看起来平行、垂直、等长推断题设。"
        "is_blank 只在确实没有保留的有效作答时为true，且不能同时输出有效文字或识别歧义；看不清不等于空白。"
    )


def grading_prompt(
    question: Question, rubric: Rubric, transcript: Transcript, step: str = "0.5"
) -> str:
    return (
        "结合学生作答原图和转写，按固定细则逐点评分。"
        "必须恰好覆盖每一个 criterion_id，正向给分引用真实且清楚的证据 id。"
        "任何正分项的 evidence_ids 只能引用 status=clear 的行；绝对不要引用 unclear 或 crossed_out。"
        "如果一个评分点必须依赖 unclear 行才能判断，即使你认为原图可能看得清，也将该点评为 null，交给独立重读。"
        "能由其他 clear 行独立证明的评分点照常给分，不要把无关的模糊内容附加为正分证据。"
        f"每点 score 不得超过该点评分上限，且必须是 {step} 的整数倍。"
        "允许其他有效解法，不逐字匹配参考解答，不补全学生缺失步骤。"
        "遵循部分给分和错误延续规则；每点输出简短 reason。"
        "无法判定的分项 score 为 null，说明原因；不得把无法辨认计为零分。"
        "满分和分值由程序计算，不需要输出总分。\n题目："
        + question.model_dump_json()
        + "\n固定评分细则："
        + rubric.model_dump_json()
        + "\n忠实转写："
        + transcript.model_dump_json()
    )
