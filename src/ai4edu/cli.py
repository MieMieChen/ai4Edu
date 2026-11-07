import argparse
import json
import mimetypes
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx

from ai4edu.config import get_settings
from ai4edu.db import Database
from ai4edu.sample import prepare_sample


def submit_manifest(
    manifest: Path,
    server: str,
    wait_seconds: int,
    budget: str | None,
    *,
    exam_id: str | None = None,
    candidate_ids: set[str] | None = None,
    output: Path | None = None,
    exam_budget: str | None = None,
    strategy: str | None = None,
    upload_concurrency: int = 8,
):
    if upload_concurrency < 1:
        raise ValueError("上传并发必须为正整数")
    if exam_id is not None and exam_budget is not None:
        raise ValueError("复用试卷时沿用已有总预算，不能通过提交批次修改")
    data = json.loads(manifest.read_text())
    if exam_budget is not None:
        data["exam"]["budget_cny"] = exam_budget
    if candidate_ids is not None:
        available = {candidate["candidate_id"] for candidate in data["candidates"]}
        if not candidate_ids or candidate_ids - available:
            raise ValueError("所选考生必须存在于输入清单中")
        data["candidates"] = [
            candidate
            for candidate in data["candidates"]
            if candidate["candidate_id"] in candidate_ids
        ]
    settings = get_settings()
    headers = (
        {"Authorization": f"Bearer {settings.service_api_key.get_secret_value()}"}
        if settings.service_api_key
        else {}
    )
    uploaded = {}
    with httpx.Client(base_url=server, headers=headers, timeout=120) as client:

        def upload(path: str) -> list[str]:
            source = manifest.parent / path
            with source.open("rb") as stream:
                response = client.post(
                    "/v1/assets",
                    files={"file": (source.name, stream, mimetypes.guess_type(source)[0])},
                )
            response.raise_for_status()
            return [image["id"] for image in response.json()]

        paths = []
        if exam_id is None:
            paths.extend(
                path
                for question in data["exam"]["questions"]
                for field in ("question_assets", "reference_assets")
                for path in question.get(field, [])
            )
        for candidate in data["candidates"]:
            paths.extend(
                candidate["pages"]
                if candidate["kind"] == "paper"
                else [path for answer in candidate["answers"] for path in answer["assets"]]
            )
        paths = list(dict.fromkeys(paths))
        for path in paths:
            if not (manifest.parent / path).is_file():
                raise FileNotFoundError(manifest.parent / path)
        pool = ThreadPoolExecutor(max_workers=upload_concurrency)
        try:
            remaining = iter(paths)
            pending = {
                pool.submit(upload, path): path
                for path in [next(remaining) for _ in range(min(upload_concurrency, len(paths)))]
            }
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    uploaded[pending.pop(future)] = future.result()
                while len(pending) < upload_concurrency:
                    path = next(remaining, None)
                    if path is None:
                        break
                    pending[pool.submit(upload, path)] = path
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        if exam_id is None:
            for question in data["exam"]["questions"]:
                for field in ("question_assets", "reference_assets"):
                    question[field] = [
                        ident for path in question.get(field, []) for ident in uploaded[path]
                    ]
        for candidate in data["candidates"]:
            if candidate["kind"] == "paper":
                candidate["pages"] = [
                    ident for path in candidate["pages"] for ident in uploaded[path]
                ]
            else:
                for answer in candidate["answers"]:
                    answer["assets"] = [
                        ident for path in answer["assets"] for ident in uploaded[path]
                    ]
        if exam_id is None:
            response = client.post("/v1/exams", json=data["exam"])
            response.raise_for_status()
            exam_id = response.json()["id"]
        print(json.dumps({"exam_id": exam_id, "state": "preparing"}), flush=True)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            response = client.get(f"/v1/exams/{exam_id}")
            response.raise_for_status()
            exam = response.json()
            if exam["state"] == "failed":
                raise RuntimeError(exam["error"])
            if exam["state"] == "ready":
                payload = {
                    "exam_id": exam_id,
                    "candidates": data["candidates"],
                    "budget_cny": budget,
                }
                if strategy is not None:
                    payload["strategy"] = strategy
                response = client.post("/v1/jobs", json=payload)
                response.raise_for_status()
                result = response.json()
                if output is not None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return result
            time.sleep(2)
        raise TimeoutError(f"试卷准备仍在后台运行，可查询 /v1/exams/{exam_id}")


def main():
    parser = argparse.ArgumentParser(description="AI4Edu 主观题评分后端工具")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="显式创建数据库表")
    sample = commands.add_parser("prepare-sample", help="匿名整理原始测试ZIP，不读取成绩CSV")
    sample.add_argument("--archive", type=Path, required=True)
    sample.add_argument("--output", type=Path, default=Path("data/sample"))
    sample.add_argument("--limit", type=int, default=2)
    submit = commands.add_parser("submit", help="上传 manifest，准备评分细则后提交批量评分")
    submit.add_argument("manifest", type=Path)
    submit.add_argument("--server", default="http://127.0.0.1:8000")
    submit.add_argument("--wait-seconds", type=int, default=900)
    submit.add_argument("--budget", default=None, help="学生作答处理的费用预算，单位人民币")
    submit.add_argument("--exam-budget", help="新试卷的累计总预算，包含细则及其全部批次")
    submit.add_argument(
        "--strategy", choices=["A", "B", "C"], help="该批次显式使用评测策略，不改变默认模型配置"
    )
    submit.add_argument("--exam-id", help="复用已经冻结的评分细则")
    submit.add_argument("--candidate-ids", nargs="+", help="仅提交这些匿名考生")
    submit.add_argument("--output", type=Path, help="保存批次ID与生效配置")
    submit.add_argument("--upload-concurrency", type=int, default=8)
    probe = commands.add_parser("probe", help="用合成图片探测至多两个模型的权限、JSON和Token统计")
    probe.add_argument("--models", nargs="+", default=["qwen3.7-plus", "qwen3.8-max"])
    probe.add_argument("--output", type=Path, default=Path("data/probe.json"))
    evaluation_data = commands.add_parser(
        "prepare-eval", help="生成匿名输入、隔离成绩和固定学生分组"
    )
    evaluation_data.add_argument("--archive", type=Path, required=True)
    evaluation_data.add_argument("--output", type=Path, required=True)
    evaluation_data.add_argument("--seed", type=int, default=20261107)
    benchmark = commands.add_parser(
        "benchmark", help="复用实际C批次的固定细则和裁图，提交A对照并生成A/B/C报告"
    )
    benchmark.add_argument("--source-job", required=True)
    benchmark.add_argument("--labels", type=Path, required=True)
    benchmark.add_argument("--split", choices=["debug", "initial", "holdout"])
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--budget", required=True)
    benchmark.add_argument("--wait-seconds", type=int, default=3600)
    report = commands.add_parser("evaluate", help="从已完成批次重新生成离线报告，不发起模型请求")
    report.add_argument("--source-job", required=True)
    report.add_argument("--max-job", required=True)
    report.add_argument("--labels", type=Path, required=True)
    report.add_argument("--split", choices=["debug", "initial", "holdout"])
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init-db":
        Database(get_settings().database_url).create_all()
        print("数据库表已创建")
    elif args.command == "prepare-sample":
        if args.limit < 1:
            parser.error("--limit 必须大于0")
        print(prepare_sample(args.archive, args.output, args.limit))
    elif args.command == "submit":
        submit_manifest(
            args.manifest,
            args.server,
            args.wait_seconds,
            args.budget,
            exam_id=args.exam_id,
            candidate_ids=set(args.candidate_ids) if args.candidate_ids else None,
            output=args.output,
            exam_budget=args.exam_budget,
            strategy=args.strategy,
            upload_concurrency=args.upload_concurrency,
        )
    elif args.command == "probe":
        from ai4edu.probe import run_probe

        print(
            json.dumps(
                run_probe(get_settings(), args.models, args.output), ensure_ascii=False, indent=2
            )
        )
    elif args.command == "prepare-eval":
        from ai4edu.evaluation_data import prepare_evaluation_data

        print(
            json.dumps(
                {
                    key: str(value)
                    for key, value in prepare_evaluation_data(
                        args.archive, args.output, args.seed
                    ).items()
                },
                ensure_ascii=False,
            )
        )
    elif args.command in {"benchmark", "evaluate"}:
        from ai4edu.api import job_summary
        from ai4edu.benchmark import create_comparison_job, write_evaluation_report
        from ai4edu.db import JobRecord

        settings = get_settings()
        db = Database(settings.database_url)
        if args.command == "benchmark":
            max_job = create_comparison_job(
                db, settings, args.source_job, "A", budget_cny=args.budget
            )
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "submission.json").write_text(
                json.dumps({"source_job_id": args.source_job, "max_job_id": max_job}, indent=2)
            )
            print(json.dumps({"source_job_id": args.source_job, "max_job_id": max_job}), flush=True)
            deadline, previous = time.monotonic() + args.wait_seconds, None
            while time.monotonic() < deadline:
                with db.session() as session:
                    summaries = [
                        job_summary(db, session.get(JobRecord, ident))
                        for ident in [args.source_job, max_job]
                    ]
                counts = [item["counts"] for item in summaries]
                if counts != previous:
                    print(json.dumps({"counts_C_A": counts}, ensure_ascii=False), flush=True)
                    previous = counts
                if all(item["completed"] == item["total"] for item in summaries):
                    break
                time.sleep(2)
            else:
                raise TimeoutError("对照仍在后台运行，保留 submission.json，可用 evaluate 重建报告")
        else:
            max_job = args.max_job
        results = write_evaluation_report(
            db, args.source_job, max_job, args.labels, args.output, split=args.split
        )
        print(
            json.dumps(
                {
                    "selected_strategy": results["selection"]["strategy"],
                    "report": str(args.output / "report.md"),
                    "cost_accounting": results["cost_accounting"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
