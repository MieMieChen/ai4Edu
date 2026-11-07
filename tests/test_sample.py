import json
import zipfile
from io import BytesIO

from PIL import Image

from ai4edu.sample import prepare_sample


def test_sample_import_anonymizes_names_and_keeps_labels_out(tmp_path):
    image = BytesIO()
    Image.new("RGB", (100, 100), "white").save(image, format="JPEG")
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for page in (1, 2):
            zipped.writestr(f"测试数据集/136_学生姓名_12345678({page}).jpg", image.getvalue())
        for qid in range(15, 20):
            zipped.writestr(f"测试数据集/试卷/数学 {qid} 题.png", image.getvalue())
            zipped.writestr(f"测试数据集/试卷/数学 {qid} 题答案.png", image.getvalue())
        zipped.writestr("测试数据集/成绩.csv", "姓名,总分\n学生姓名,136")
    manifest = prepare_sample(archive, tmp_path / "output", 1)
    data = json.loads(manifest.read_text())
    assert len(data["exam"]["questions"]) == 5
    assert len(data["candidates"]) == 1
    assert len(data["candidates"][0]["pages"]) == 2
    text = manifest.read_text()
    assert "学生姓名" not in text and "12345678" not in text and "136" not in text
