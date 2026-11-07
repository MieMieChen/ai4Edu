"""Offline adapter for the supplied 50-student mathematics evaluation archive.

Only ``inputs/`` may be supplied to the grading pipeline. Existing grades live in
the sibling ``labels.jsonl`` and are never included in a model manifest. The
source archive remains unchanged; source identifiers are used only for the
in-memory exact join and are not written to the prepared artifacts.
"""

import csv
import json
import math
import random
import re
import zipfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from ai4edu.sample import decode_zip_name, prepare_sample

QUESTION_MAXIMA = {"15": 13, "16": 15, "17": 15, "18": 17, "19": 17}
STUDENT_COUNT = 50


def _archive_labels(archive: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Validate the source relationships without exposing private source names."""
    with zipfile.ZipFile(archive) as zipped:
        entries = {}
        for info in zipped.infolist():
            name = decode_zip_name(info.filename)
            if info.is_dir() or name.startswith("__MACOSX/") or Path(name).name == ".DS_Store":
                continue
            if name in entries:
                raise ValueError("压缩包包含重复文件条目")
            entries[name] = info

        pages = defaultdict(set)
        for name in entries:
            if Path(name).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            match = re.fullmatch(r"[0-9.]+_.+_([0-9]+)\(([0-9]+)\)\.jpg", Path(name).name, re.I)
            if not match:
                raise ValueError("学生答卷文件不符合已核对的数据集命名约定")
            student, page = match[1], int(match[2])
            if page in pages[student]:
                raise ValueError("学生答卷存在重复页")
            pages[student].add(page)
        if len(pages) != STUDENT_COUNT:
            raise ValueError("评测数据必须包含50名学生答卷")
        if any(student_pages != {1, 2} for student_pages in pages.values()):
            raise ValueError("每名学生答卷必须恰好包含第1、2页")

        for qid in QUESTION_MAXIMA:
            questions = [name for name in entries if Path(name).name == f"数学 {qid} 题.png"]
            references = [
                name for name in entries
                if re.fullmatch(rf"数学 {qid} 题答案.*\.png", Path(name).name)
            ]
            if len(questions) != 1 or not references:
                raise ValueError("主观题题目必须唯一且具有参考解答")
            if len({Path(name).name for name in references}) != len(references):
                raise ValueError("主观题参考解答包含重复文件名")

        csv_entries = [info for name, info in entries.items() if Path(name).suffix.lower() == ".csv"]
        if len(csv_entries) != 1:
            raise ValueError("评测数据必须包含唯一的成绩CSV")
        try:
            contents = zipped.read(csv_entries[0]).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("成绩CSV必须使用UTF-8编码") from None
        reader = csv.DictReader(StringIO(contents, newline=""))
        required = {"学号", *(f"第{qid}题" for qid in QUESTION_MAXIMA)}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("成绩CSV缺少学号或主观题得分列")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("成绩CSV包含重复列名")
        labels = {}
        for row in reader:
            student = row.get("学号")
            if None in row or not isinstance(student, str) or not re.fullmatch(r"[0-9]+", student):
                raise ValueError("成绩CSV包含无效学号或列数")
            if student in labels:
                raise ValueError("成绩CSV包含重复学号")
            question_scores = {}
            for qid, maximum in QUESTION_MAXIMA.items():
                try:
                    score = Decimal(row[f"第{qid}题"])
                except (InvalidOperation, TypeError):
                    raise ValueError("成绩CSV包含无效主观题得分") from None
                if not score.is_finite() or not 0 <= score <= maximum:
                    raise ValueError("成绩CSV主观题得分超出合法范围")
                question_scores[qid] = str(score)
            labels[student] = question_scores
        if set(labels) != set(pages):
            raise ValueError("成绩CSV与学生答卷学号必须完整一一对应")
    return sorted(pages), labels


def _anonymize_pages(inputs: Path, manifest: dict) -> None:
    """Apply the visually verified layout of this archive, not a general detector.

    The identity header occupies x=[0,.35), y=[0,.30) on stored page 1.
    Subjective q15/q16 start to its right. A lossless PNG derivative whites only
    this rectangle; page 2 pixels are unchanged. Fresh images strip metadata from
    both pages. This adapter does not claim to locate identities in other exams.
    """
    for candidate in manifest["candidates"]:
        targets = []
        for page_number, filename in enumerate(candidate["pages"], 1):
            source = inputs / filename
            with Image.open(source) as image:
                clean = Image.new("RGB", image.size)
                clean.paste(image.convert("RGB"))
            if page_number == 1:
                width, height = clean.size
                clean.paste("white", (0, 0, math.ceil(width * 0.35), math.ceil(height * 0.30)))
            target = source.with_suffix(".png")
            clean.save(target, format="PNG")
            source.unlink()
            targets.append(target.name)
        candidate["pages"] = targets


def prepare_evaluation_data(archive: Path, output: Path, seed: int = 20261107) -> dict:
    """Prepare private labels and anonymous inputs with a frozen 5/10/35 split.

    Returns ``inputs``, ``manifest``, ``labels`` and ``splits`` as ``Path`` objects.
    Every page/CSV join is checked before creating outputs. All artifacts are
    built in a private sibling directory and published with one directory rename
    after anonymization and label/split writes succeed. Failed staging is removed.
    Nonempty outputs are refused rather than overwritten or merged with stale inputs.
    Candidate numbering follows ``prepare_sample``'s sorted source student IDs;
    the random seed changes only the student-group split.
    """
    archive, output = Path(archive), Path(output)
    final_paths = {
        "inputs": output / "inputs",
        "manifest": output / "inputs" / "manifest.json",
        "labels": output / "labels.jsonl",
        "splits": output / "splits.json",
    }
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("评测输出已存在，请使用新的输出目录")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("数据划分种子必须为整数")
    students, source_labels = _archive_labels(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temporary:
        staging = Path(temporary)
        paths = {name: staging / path.relative_to(output) for name, path in final_paths.items()}
        prepare_sample(archive, paths["inputs"], STUDENT_COUNT)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        _anonymize_pages(paths["inputs"], manifest)
        paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        candidate_ids = [candidate["candidate_id"] for candidate in manifest["candidates"]]
        shuffled = candidate_ids.copy()
        random.Random(seed).shuffle(shuffled)
        splits = {"seed": seed, "debug": shuffled[:5], "initial": shuffled[5:15], "holdout": shuffled[15:]}
        paths["splits"].write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
        split_for = {candidate: split for split in ("debug", "initial", "holdout") for candidate in splits[split]}
        label_rows = [
            {
                "candidate_id": candidate,
                "question_id": qid,
                "score": source_labels[student][qid],
                "max_score": str(maximum),
                "split": split_for[candidate],
            }
            for student, candidate in zip(students, candidate_ids, strict=True)
            for qid, maximum in QUESTION_MAXIMA.items()
        ]
        paths["labels"].write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in label_rows), encoding="utf-8"
        )
        staging.rename(output)
    return final_paths
