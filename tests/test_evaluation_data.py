import csv
import importlib
import importlib.util
import json
import zipfile
from io import BytesIO, StringIO

import pytest
from PIL import Image, UnidentifiedImageError


def _prepare(*args, **kwargs):
    assert importlib.util.find_spec("ai4edu.evaluation_data") is not None, (
        "Anonymous evaluation data preparation is not implemented"
    )
    return importlib.import_module("ai4edu.evaluation_data").prepare_evaluation_data(*args, **kwargs)


def _make_archive(tmp_path, *, student_count=50, rows=None, missing_page=None, duplicate_page=False, corrupt_page=None):
    image = BytesIO()
    Image.new("RGB", (100, 100), (12, 34, 56)).save(image, format="JPEG")
    image_bytes = image.getvalue()
    archive = tmp_path / "sample.zip"
    if rows is None:
        rows = [
            {
                "学号": str(90000000 + index),
                "姓名": f"学生姓名{index}",
                "总分": "111",
                **{f"第{qid}题": "6.50" for qid in range(15, 20)},
            }
            for index in range(student_count)
        ]
    csv_file = StringIO(newline="")
    writer = csv.DictWriter(csv_file, ["学号", "姓名", "总分", *[f"第{q}题" for q in range(15, 20)]])
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(archive, "w") as zipped:
        for index in range(student_count):
            for page in (1, 2):
                if missing_page == (index, page):
                    continue
                zipped.writestr(
                    f"测试数据集/111_学生姓名{index}_{90000000 + index}({page}).jpg",
                    b"invalid image data" if corrupt_page == (index, page) else image_bytes,
                )
        if duplicate_page:
            zipped.writestr("测试数据集/112_重复姓名_90000000(1).jpg", image_bytes)
        for qid in range(15, 20):
            zipped.writestr(f"测试数据集/试卷/数学 {qid} 题.png", image_bytes)
            zipped.writestr(f"测试数据集/试卷/数学 {qid} 题答案.png", image_bytes)
        zipped.writestr("测试数据集/试卷/数学 19 题答案2.png", image_bytes)
        zipped.writestr("测试数据集/成绩.csv", csv_file.getvalue().encode("utf-8-sig"))
    return archive, rows, image_bytes


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_preparation_has_deterministic_student_splits_and_isolated_decimal_labels(tmp_path):
    archive, _, _ = _make_archive(tmp_path)
    first = _prepare(archive, tmp_path / "first")
    second = _prepare(archive, tmp_path / "second")

    assert set(first) == {"inputs", "manifest", "labels", "splits"}
    assert first["manifest"] == first["inputs"] / "manifest.json"
    assert first["labels"].parent == first["inputs"].parent
    assert first["labels"].parent != first["inputs"]
    manifest = json.loads(first["manifest"].read_text())
    assert set(manifest) == {"exam", "candidates"}
    assert len(manifest["candidates"]) == 50
    assert all(len(candidate["pages"]) == 2 for candidate in manifest["candidates"])
    assert len(manifest["exam"]["questions"][-1]["reference_assets"]) == 2

    splits = json.loads(first["splits"].read_text())
    assert splits["seed"] == 20261107
    assert [len(splits[name]) for name in ("debug", "initial", "holdout")] == [5, 10, 35]
    grouped = [candidate for name in ("debug", "initial", "holdout") for candidate in splits[name]]
    assert len(set(grouped)) == len(grouped) == 50
    assert set(grouped) == {candidate["candidate_id"] for candidate in manifest["candidates"]}
    assert first["splits"].read_bytes() == second["splits"].read_bytes()
    assert first["manifest"].read_bytes() == second["manifest"].read_bytes()
    assert first["labels"].read_bytes() == second["labels"].read_bytes()

    labels = _read_jsonl(first["labels"])
    assert len(labels) == 250
    assert set(labels[0]) == {"candidate_id", "question_id", "score", "max_score", "split"}
    assert labels[0]["score"] == "6.50"
    assert [row["max_score"] for row in labels[:5]] == ["13", "15", "15", "17", "17"]
    for candidate in grouped:
        candidate_labels = [row for row in labels if row["candidate_id"] == candidate]
        assert len(candidate_labels) == 5
        assert len({row["split"] for row in candidate_labels}) == 1
        assert candidate in splits[candidate_labels[0]["split"]]

    visible_text = first["manifest"].read_text() + " ".join(
        str(path.relative_to(first["inputs"])) for path in first["inputs"].rglob("*")
    )
    for private in ("学生姓名", "90000000", "111", '"score"', "split", "labels", "学号"):
        assert private not in visible_text


def test_page_one_identity_header_is_white_and_subjective_pixels_are_unchanged(tmp_path):
    archive, _, original_bytes = _make_archive(tmp_path)
    result = _prepare(archive, tmp_path / "prepared")
    manifest = json.loads(result["manifest"].read_text())
    pages = manifest["candidates"][0]["pages"]
    original = Image.open(BytesIO(original_bytes)).convert("RGB")
    with Image.open(result["inputs"] / pages[0]) as page:
        assert page.format == "PNG"
        assert page.size == original.size
        assert page.getpixel((34, 29)) == (255, 255, 255)
        assert page.getpixel((35, 29)) == original.getpixel((35, 29))
        assert page.getpixel((20, 30)) == original.getpixel((20, 30))
    with Image.open(result["inputs"] / pages[1]) as page:
        assert page.getpixel((20, 20)) == original.getpixel((20, 20))
    assert not list(result["inputs"].glob("*.jpg"))


def test_label_join_uses_student_id_order_not_csv_order(tmp_path):
    _, rows, _ = _make_archive(tmp_path)
    rows[0]["第15题"] = "0.125"
    archive, _, _ = _make_archive(tmp_path, rows=list(reversed(rows)))
    result = _prepare(archive, tmp_path / "prepared", seed=7)
    labels = _read_jsonl(result["labels"])
    assert labels[0]["candidate_id"] == "candidate_0001"
    assert labels[0]["score"] == "0.125"
    assert json.loads(result["splits"].read_text())["seed"] == 7


@pytest.mark.parametrize("problem", ["duplicate", "missing", "unmatched"])
def test_invalid_csv_join_fails_without_logging_private_identifiers(tmp_path, problem):
    _, rows, _ = _make_archive(tmp_path)
    if problem == "duplicate":
        rows[-1] = rows[0].copy()
    elif problem == "missing":
        rows.pop()
    else:
        rows[-1]["学号"] = "99999999"
    archive, _, _ = _make_archive(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="CSV|成绩|对应|学号") as error:
        _prepare(archive, tmp_path / "prepared")
    assert "90000000" not in str(error.value)
    assert "99999999" not in str(error.value)
    assert "学生姓名" not in str(error.value)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("problem", ["duplicate_page", "missing_page", "student_count"])
def test_invalid_student_pages_fail_before_writing_inputs(tmp_path, problem):
    options = {
        "duplicate_page": {"duplicate_page": True},
        "missing_page": {"missing_page": (0, 2)},
        "student_count": {"student_count": 49},
    }
    archive, _, _ = _make_archive(tmp_path, **options[problem])
    with pytest.raises(ValueError, match="答卷|页|50"):
        _prepare(archive, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("score", ["", "NaN", "Infinity", "-0.1", "13.01", "wrong"])
def test_bad_labels_are_rejected_instead_of_repaired(tmp_path, score):
    _, rows, _ = _make_archive(tmp_path)
    rows[0]["第15题"] = score
    archive, _, _ = _make_archive(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="得分|成绩|分数"):
        _prepare(archive, tmp_path / "prepared")


def test_preparation_refuses_to_overwrite_existing_artifacts(tmp_path):
    archive, _, _ = _make_archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "labels.jsonl").write_text("private existing labels")
    with pytest.raises(FileExistsError):
        _prepare(archive, output)
    assert (output / "labels.jsonl").read_text() == "private existing labels"


def test_corrupt_student_image_leaves_no_published_or_staged_inputs(tmp_path):
    archive, _, _ = _make_archive(tmp_path, corrupt_page=(0, 1))
    output = tmp_path / "prepared"
    original_archive = archive.read_bytes()
    with pytest.raises(UnidentifiedImageError):
        _prepare(archive, output)
    assert not (output / "inputs" / "manifest.json").exists()
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.staging-*"))
    assert archive.read_bytes() == original_archive


@pytest.mark.parametrize("existing_empty_directory", [False, True])
def test_final_manifest_is_private_until_anonymization_finishes(tmp_path, monkeypatch, existing_empty_directory):
    archive, _, _ = _make_archive(tmp_path)
    output = tmp_path / "prepared"
    if existing_empty_directory:
        output.mkdir()
    module = importlib.import_module("ai4edu.evaluation_data")
    anonymize = module._anonymize_pages

    def verify_unpublished(inputs, manifest):
        assert inputs != output / "inputs"
        assert not (output / "inputs" / "manifest.json").exists()
        anonymize(inputs, manifest)
        assert not (output / "inputs" / "manifest.json").exists()

    monkeypatch.setattr(module, "_anonymize_pages", verify_unpublished)
    paths = _prepare(archive, output)
    assert all(path.exists() for path in paths.values())
    assert not list(tmp_path.glob(".prepared.staging-*"))
