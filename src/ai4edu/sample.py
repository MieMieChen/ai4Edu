import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path


def decode_zip_name(name: str) -> str:
    # The supplied ZIP stores UTF-8 filenames without setting the UTF-8 flag.
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def prepare_sample(archive: Path, output: Path, limit: int) -> Path:
    """Convert the provided math sample to an anonymous input manifest, excluding CSV labels."""
    output.mkdir(parents=True, exist_ok=True)
    scores = {15: 13, 16: 15, 17: 15, 18: 17, 19: 17}
    with zipfile.ZipFile(archive) as zipped:
        entries = {
            decode_zip_name(info.filename): info
            for info in zipped.infolist()
            if not info.is_dir() and not info.filename.startswith("__MACOSX/")
        }
        questions = []
        for qid, maximum in scores.items():
            question_name = next(
                name for name in entries if Path(name).name == f"数学 {qid} 题.png"
            )
            references = sorted(
                name
                for name in entries
                if re.fullmatch(rf"数学 {qid} 题答案.*\.png", Path(name).name)
            )
            if not references:
                raise ValueError(f"第{qid}题缺少参考解答")
            question_target = f"question_{qid}.png"
            (output / question_target).write_bytes(zipped.read(entries[question_name]))
            reference_paths = []
            for part, source in enumerate(references, 1):
                target = f"reference_{qid}_{part}.png"
                (output / target).write_bytes(zipped.read(entries[source]))
                reference_paths.append(target)
            questions.append(
                {
                    "id": str(qid),
                    "max_score": maximum,
                    "question_assets": [question_target],
                    "reference_assets": reference_paths,
                }
            )
        students = defaultdict(list)
        for name, info in entries.items():
            match = re.fullmatch(r"[0-9.]+_.+_([0-9]+)\(([0-9]+)\)\.jpg", Path(name).name, re.I)
            if match:
                students[match[1]].append((int(match[2]), info))
        candidates = []
        for index, student in enumerate(sorted(students)[:limit], 1):
            pages = []
            for page, info in sorted(students[student], key=lambda item: item[0]):
                target = f"candidate_{index:04}_page_{page}.jpg"
                (output / target).write_bytes(zipped.read(info))
                pages.append(target)
            candidates.append(
                {"candidate_id": f"candidate_{index:04}", "kind": "paper", "pages": pages}
            )
        if not candidates:
            raise ValueError("未找到符合样例文件名约定的学生答卷")
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"exam": {"title": "数学主观题样例", "questions": questions}, "candidates": candidates},
            ensure_ascii=False,
            indent=2,
        )
    )
    return manifest
