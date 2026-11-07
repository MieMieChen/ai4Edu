"""Opt-in real-stack load harness; never imported by the production API or workers.

The model substitute is a separate HTTP service. Production dispatch, reservations,
limits, image preparation, checkpoints, scoring and persistence remain unmodified.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import psycopg
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from psycopg import sql

FAULT_TYPES = ("429", "503", "timeout", "bad_json", "invalid_evidence")
QUESTIONS = [str(n) for n in range(15, 20)]
MODEL = "local-loadtest-only"


@dataclass
class FaultPlan:
    seed: str = "loadtest"
    probability: float = 0
    by_question: dict[str, str] = field(default_factory=dict)

    def select(self, schema: str, identity: str, question_id: str) -> str | None:
        if schema != "Judgment":
            return None
        if question_id in self.by_question:
            fault = self.by_question[question_id]
            if fault not in FAULT_TYPES:
                raise ValueError("Unsupported local fault")
            return fault
        digest = hashlib.sha256(f"{self.seed}:{schema}:{identity}".encode()).digest()
        if int.from_bytes(digest[:8], "big") / (2**64) >= self.probability:
            return None
        return FAULT_TYPES[int.from_bytes(digest[8:16], "big") % len(FAULT_TYPES)]


def _after(prompt: str, marker: str):
    offset = prompt.index(marker) + len(marker)
    return json.JSONDecoder().raw_decode(prompt[offset:].lstrip())[0]


def _protocol(body: dict) -> tuple[str, str, list[str], str, dict]:
    messages = body["messages"]
    system = next(m["content"] for m in messages if m["role"] == "system")
    schema = json.loads(system[system.index("{") :])["title"]
    content = next(m["content"] for m in messages if m["role"] == "user")
    texts = [c["text"] for c in content if c.get("type") == "text"]
    prompt = texts[0]
    images = [text[len("image_id=") :] for text in texts[1:] if text.startswith("image_id=")]
    qid = ""
    if schema == "Rubric":
        question = _after(prompt, "题目：")
        qid = question["id"]
        result = {
            "question_id": qid,
            "max_score": question["max_score"],
            "solution": question.get("reference") or "Synthetic correct solution",
            "criteria": [
                {
                    "id": f"{qid}.c1",
                    "description": "Synthetic complete correct solution",
                    "max_score": question["max_score"],
                    "rules": "完整正确得满分；缺少有效论证得零分。",
                }
            ],
        }
    elif schema == "PageLayout":
        ids = _after(prompt, "指定题号：") if "指定题号：" in prompt else []
        result = {
            "rotation": 0,
            "regions": [
                {
                    "question_id": ident,
                    "bbox": [0, index / len(ids), 1, (index + 1) / len(ids)],
                    "order": index,
                }
                for index, ident in enumerate(ids)
            ],
        }
    elif schema == "Transcript":
        result = {
            "is_blank": False,
            "lines": [
                {
                    "id": f"e{index}",
                    "image_id": ident,
                    "text": "Synthetic complete correct solution",
                    "status": "clear",
                }
                for index, ident in enumerate(images)
            ],
        }
    elif schema == "Judgment":
        rubric = _after(prompt, "固定评分细则：")
        transcript = _after(prompt, "忠实转写：")
        qid = rubric["question_id"]
        evidence = [
            line["id"] for line in transcript["lines"] if line.get("status", "clear") == "clear"
        ]
        result = {
            "scores": [
                {
                    "criterion_id": criterion["id"],
                    "score": criterion["max_score"],
                    "evidence_ids": evidence,
                    "reason": "Synthetic correct evidence",
                }
                for criterion in rubric["criteria"]
            ]
        }
    else:
        raise ValueError(f"Unsupported JSON Schema title: {schema}")
    return schema, prompt, images, qid, result


def create_mock_app(*, delay_seconds: float = 0.2, journal: Path | None = None) -> FastAPI:
    app = FastAPI(title="Isolated AI4Edu compatible load-test model")
    state: dict[str, Any] = {
        "delay_seconds": delay_seconds,
        "timeout_seconds": 2.0,
        "plan": FaultPlan(),
        "active": 0,
        "peak_inflight": 0,
        "records": [],
        "started": time.monotonic(),
        "usage_enabled": True,
    }

    @app.get("/health")
    async def health():
        return {"status": "local-mock-only"}

    @app.post("/control")
    async def control(payload: dict):
        if payload.get("reset"):
            if state["active"]:
                raise HTTPException(409, "Wait for mock requests to finish before reset")
            state["records"], state["peak_inflight"] = [], 0
            state["started"] = time.monotonic()
        for key in ("delay_seconds", "timeout_seconds", "usage_enabled"):
            if key in payload:
                state[key] = payload[key]
        if "faults" in payload:
            state["plan"] = FaultPlan(**payload["faults"])
        return {"configured": True}

    @app.get("/stats")
    async def stats():
        records = state["records"]
        return {
            "total_calls": len(records),
            "active": state["active"],
            "peak_inflight": state["peak_inflight"],
            "calls_by_schema": dict(Counter(r["schema"] for r in records)),
            "calls_by_status": dict(Counter(str(r["status"]) for r in records)),
            "faults": dict(Counter(r["fault"] for r in records if r["fault"])),
            "wall_seconds": round(time.monotonic() - state["started"], 3),
        }

    @app.post("/v1/chat/completions")
    async def complete(body: dict):
        try:
            schema, _, images, qid, value = _protocol(body)
        except (KeyError, ValueError, StopIteration, TypeError) as exc:
            raise HTTPException(422, f"Local mock protocol error: {exc}") from exc
        visual_inputs = [
            item["image_url"]["url"]
            for message in body["messages"]
            if message["role"] == "user"
            for item in message["content"]
            if item.get("type") == "image_url"
        ]
        # Raster content makes seeded faults reproducible across new random asset IDs.
        identity = hashlib.sha256(json.dumps([schema, visual_inputs, qid]).encode()).hexdigest()
        fault = state["plan"].select(schema, identity, qid)
        state["active"] += 1
        state["peak_inflight"] = max(state["peak_inflight"], state["active"])
        record = {
            "sequence": len(state["records"]) + 1,
            "schema": schema,
            "question_id": qid,
            "image_ids": images,
            "identity": identity,
            "fault": fault,
            "status": int(fault) if fault in {"429", "503"} else 200,
            "started_at": datetime.now(UTC).isoformat(),
        }
        state["records"].append(record)
        start = time.monotonic()
        try:
            await asyncio.sleep(
                state["timeout_seconds"] if fault == "timeout" else state["delay_seconds"]
            )
            if fault in {"429", "503"}:
                return JSONResponse(
                    {"error": {"message": f"seeded local {fault}"}}, status_code=int(fault)
                )
            if fault == "invalid_evidence":
                for score in value["scores"]:
                    score["evidence_ids"] = ["does-not-exist"]
            result = {
                "id": f"local-{record['sequence']}",
                "object": "chat.completion",
                "model": body.get("model", MODEL),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "{bad JSON" if fault == "bad_json" else json.dumps(value),
                        },
                    }
                ],
            }
            if state["usage_enabled"]:
                result["usage"] = {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            return result
        finally:
            state["active"] -= 1
            record["elapsed_seconds"] = round(time.monotonic() - start, 6)
            if journal is not None:
                with journal.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    return app


def isolated_environment(run_id: str, directory: Path, *, concurrency: int) -> dict[str, str]:
    if not re.fullmatch(r"[a-z0-9_]{1,35}", run_id):
        raise ValueError("Run id must contain 1-35 lowercase ASCII letters, digits or underscores")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AI4EDU_", "DASHSCOPE_", "OPENAI_"))
    }
    fake_key, base = "local-loadtest-fake-key", "http://127.0.0.1:18091/v1"
    env.update(
        {
            "AI4EDU_DATABASE_URL": f"postgresql+psycopg://ai4edu:ai4edu@127.0.0.1:15432/ai4edu_loadtest_{run_id}",
            "AI4EDU_REDIS_URL": "redis://127.0.0.1:16379/3",
            "AI4EDU_STORAGE_BACKEND": "local",
            "AI4EDU_STORAGE_DIR": str(directory / "assets"),
            "AI4EDU_SERVICE_API_KEY": "local-loadtest-service-key",
            "AI4EDU_CREDENTIALS_FILE": "",
            "AI4EDU_MODEL": MODEL,
            "AI4EDU_MAX_OUTPUT_TOKENS": "256",
            "AI4EDU_THINKING_ROLES": "[]",
            "AI4EDU_THINKING_BUDGET": "128",
            "AI4EDU_HTTP_TIMEOUT": "1",
            "AI4EDU_MAX_PIXELS": "65536",
            "AI4EDU_MODEL_CONCURRENCY": str(concurrency),
            "AI4EDU_MODEL_RPM": "1000000",
            "AI4EDU_MODEL_TPM": "10000000000",
            "AI4EDU_VERIFICATION_POLICY": "off",
            "AI4EDU_AUDIT_RATE": "0",
            "AI4EDU_TASK_LEASE_SECONDS": "300",
            "AI4EDU_MODEL_PRICES": json.dumps({MODEL: {"input": "1", "output": "1"}}),
            "AI4EDU_OSS_ENDPOINT": "",
            "AI4EDU_OSS_BUCKET": "",
            "AI4EDU_OSS_ACCESS_KEY_ID": "",
            "AI4EDU_OSS_ACCESS_KEY_SECRET": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
    for prefix in ("AI4EDU", "DASHSCOPE", "OPENAI"):
        env[f"{prefix}_API_KEY"], env[f"{prefix}_BASE_URL"] = fake_key, base
    for role in ("LAYOUT", "RUBRIC", "TRANSCRIBE", "GRADE", "VERIFY"):
        env[f"AI4EDU_{role}_MODEL"] = MODEL
    return env


def synthetic_paper(candidate: int) -> bytes:
    image = Image.new("RGB", (96, 160), "white")
    draw = ImageDraw.Draw(image)
    for index, qid in enumerate(QUESTIONS):
        y = index * 32
        draw.text((2, y + 8), f"C{candidate:06} Q{qid}", fill="black")
        # Each cropped answer contains a lossless candidate and question identifier.
        for byte in range(4):
            image.putpixel((byte, y), ((candidate >> (byte * 8)) & 255, index, 127))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def assess_full_run(measurements: dict) -> dict[str, bool]:
    return {
        "1000_unique_candidates": measurements["candidates"]
        == measurements["unique_candidate_ids"]
        == 1000,
        "5000_unique_answers": measurements["answers"] == measurements["unique_answer_ids"] == 5000,
        "all_answers_scored_correctly": measurements["answer_states"] == {"scored": 5000}
        and measurements["correct_scores"] == 5000,
        "all_results_queryable_via_api": measurements["api_queryable_candidates"] == 1000
        and measurements["api_queryable_answers"] == 5000,
        "actual_http_foundation_calls": all(
            measurements["http_calls"].get(schema) == count
            for schema, count in {"Transcript": 5000, "Judgment": 5000, "PageLayout": 1000}.items()
        ),
        "concurrent_within_configured_cap": 1
        < measurements["peak_inflight"]
        <= measurements["model_cap"],
        "no_pending_outbox": measurements["pending_outbox"] == 0,
        "idempotency_and_duplicate_delivery": measurements["duplicate_submit_same_job"]
        and measurements["duplicate_delivery_extra_calls"] == 0,
    }


class IsolatedStack:
    def __init__(self, root: Path, run_id: str, *, smoke: bool = False):
        if not re.fullmatch(r"[a-z0-9_]{1,35}", run_id):
            raise ValueError("Unsafe isolated run id")
        self.root, self.run_id, self.smoke = root, run_id, smoke
        self.directory = root / "data" / "loadtest" / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "inputs").mkdir()
        self.env = isolated_environment(run_id, self.directory, concurrency=16)
        self.database = "ai4edu_loadtest_" + run_id
        self.dsn = f"postgresql://ai4edu:ai4edu@127.0.0.1:15432/{self.database}"
        self.conn = None
        self.processes: dict[str, subprocess.Popen] = {}
        self.process_log: list[dict] = []
        self.log_handles = []
        self.baseline_redis_keys: set[bytes] = set()
        self.owns_redis = False
        self.api = httpx.Client(
            base_url="http://127.0.0.1:18090",
            timeout=120,
            trust_env=False,
            headers={"Authorization": "Bearer local-loadtest-service-key"},
        )
        self.mock = httpx.Client(base_url="http://127.0.0.1:18091", timeout=10, trust_env=False)
        self.started = time.monotonic()
        self.report: dict[str, Any] = {
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "database": self.database,
            "redis_logical_db": 3,
            "api_port": 18090,
            "model_port": 18091,
            "synthetic_provider": True,
            "production_throughput_claim": False,
            "smoke": smoke,
            "phases": {},
            "processes": self.process_log,
            "configuration": {
                "task_lease_seconds": 300,
                "model_rpm": 1000000,
                "model_tpm": 10000000000,
                "grader_threads": 48,
                "preparation_threads": 16,
                "mock_delay_seconds": 0.2,
                "http_timeout_seconds": 1,
                "verification_policy": "off",
                "price_cny_per_million_input_output": [1, 1],
            },
            "resource_samples": [],
        }

    def save(self):
        self.report["wall_seconds"] = round(time.monotonic() - self.started, 3)
        (self.directory / "report.json").write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, default=str)
        )

    def event(self, event: str, **values):
        item = {"at": datetime.now(UTC).isoformat(), "event": event, **values}
        print(json.dumps(item, ensure_ascii=False, default=str), flush=True)
        with (self.directory / "runner-events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def spawn(self, role: str, args: list[str]):
        log = (self.directory / f"{role}-{len(self.process_log)}.log").open("w")
        self.log_handles.append(log)
        # Working in the fresh artifact directory also excludes the project's .env.
        process = subprocess.Popen(
            [sys.executable, *args],
            cwd=self.directory,
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes[role] = process
        self.process_log.append(
            {
                "role": role,
                "pid": process.pid,
                "started_at": datetime.now(UTC).isoformat(),
                "log": log.name,
            }
        )
        self.event("process_started", role=role, pid=process.pid)
        return process

    def stop(self, role: str, *, abrupt: bool = False):
        process = self.processes.pop(role, None)
        if process is None:
            return
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL if abrupt else signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for item in self.process_log:
            if item["pid"] == process.pid:
                item.update(
                    stopped_at=datetime.now(UTC).isoformat(),
                    exit_code=process.returncode,
                    abrupt=abrupt,
                )
        self.event(
            "process_stopped",
            role=role,
            pid=process.pid,
            abrupt=abrupt,
            exit_code=process.returncode,
        )

    def worker(self, role: str, queue: str, concurrency: int):
        return self.spawn(
            role,
            [
                "-m",
                "celery",
                "-A",
                "ai4edu.tasks:celery",
                "worker",
                "--pool=threads",
                f"--concurrency={concurrency}",
                f"--queues={queue}",
                f"--hostname=loadtest_{self.run_id}_{role}@%h",
                "--without-gossip",
                "--without-mingle",
                "--without-heartbeat",
                "--loglevel=WARNING",
            ],
        )

    def start(self):
        from redis import Redis

        for port in (18090, 18091):
            with socket.socket() as check:
                if check.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Refusing to reuse occupied isolated port {port}")
        redis = Redis.from_url(self.env["AI4EDU_REDIS_URL"])
        self.baseline_redis_keys = set(redis.scan_iter())
        # Any existing queue could belong to another isolated run; do not consume it.
        if any(redis.type(key) == b"list" and redis.llen(key) for key in self.baseline_redis_keys):
            raise RuntimeError(
                "Redis DB 3 already has queued messages; refusing to consume another run"
            )
        with psycopg.connect(
            "postgresql://ai4edu:ai4edu@127.0.0.1:15432/postgres", autocommit=True
        ) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database)))
        self.owns_redis = True
        subprocess.run(
            [sys.executable, "-m", "ai4edu.cli", "init-db"],
            cwd=self.directory,
            env=self.env,
            check=True,
            capture_output=True,
        )
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.spawn(
            "mock",
            [
                "-m",
                "ai4edu.loadtest",
                "mock",
                "--journal",
                str(self.directory / "model-http-calls.jsonl"),
            ],
        )
        self.spawn(
            "api",
            [
                "-m",
                "uvicorn",
                "ai4edu.api:create_app",
                "--factory",
                "--host=127.0.0.1",
                "--port=18090",
                "--log-level=warning",
                "--no-access-log",
            ],
        )
        self.worker("control", "control", 1)
        self.worker("preparation", "preparation", 16)
        self.worker("grading", "grading", 48)
        self.spawn(
            "beat",
            [
                "-m",
                "celery",
                "-A",
                "ai4edu.tasks:celery",
                "beat",
                "--loglevel=WARNING",
                "--schedule",
                str(self.directory / "celerybeat-schedule"),
                "--pidfile",
                str(self.directory / "celerybeat.pid"),
            ],
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                self.api.get("/health").raise_for_status()
                self.mock.get("/health").raise_for_status()
                self.event("isolated_stack_ready", database=self.database, redis_db=3)
                return
            except httpx.HTTPError:
                time.sleep(0.2)
        raise RuntimeError("Isolated API/mock failed startup; inspect owned logs")

    def configure_mock(self, **values):
        self.mock.post("/control", json=values).raise_for_status()

    def idle_mock(self):
        deadline = time.monotonic() + 10
        while self.mock.get("/stats").json()["active"]:
            if time.monotonic() > deadline:
                raise RuntimeError("Mock requests did not finish")
            time.sleep(0.1)

    def change_cap(self, cap: int):
        self.stop("preparation")
        self.stop("grading")
        self.env["AI4EDU_MODEL_CONCURRENCY"] = str(cap)
        self.worker("preparation", "preparation", 16)
        self.worker("grading", "grading", 48)
        self.event("model_cap_changed", cap=cap)

    def query(self, statement: str, args=()):
        return self.conn.execute(statement, args).fetchall()

    def states(self, job_id: str):
        return dict(
            self.query(
                "SELECT state, count(*) FROM answers WHERE job_id=%s GROUP BY state", (job_id,)
            )
        )

    def sample_resources(self):
        if "resource_sampling_unavailable" in self.report:
            return
        pids = [str(p.pid) for p in self.processes.values() if p.poll() is None]
        try:
            result = subprocess.run(
                ["ps", "-o", "pid=,rss=,%cpu=", "-p", ",".join(pids)],
                text=True,
                capture_output=True,
            )
        except OSError as exc:
            self.report["resource_sampling_unavailable"] = f"{type(exc).__name__}: {exc}"
            return
        if result.returncode:
            self.report["resource_sampling_unavailable"] = result.stderr.strip()
            return
        rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
        if rows:
            sample = {
                "elapsed_seconds": round(time.monotonic() - self.started, 2),
                "owned_rss_mb": round(sum(int(r[1]) for r in rows) / 1024, 2),
                "owned_cpu_percent": round(sum(float(r[2]) for r in rows), 2),
            }
            self.report["resource_samples"].append(sample)

    def wait_job(self, job_id: str, *, timeout: int = 900):
        deadline, last_progress = time.monotonic() + timeout, 0
        while time.monotonic() < deadline:
            states = self.states(job_id)
            if states and all(state in {"scored", "failed", "ungradable"} for state in states):
                self.idle_mock()
                return states
            if time.monotonic() - last_progress > 20:
                self.sample_resources()
                self.event(
                    "job_progress",
                    job_id=job_id,
                    states=states,
                    model=self.mock.get("/stats").json(),
                )
                self.save()
                last_progress = time.monotonic()
            time.sleep(0.3)
        raise RuntimeError(f"Job did not terminate: {job_id}, states={self.states(job_id)}")

    def create_exam(self, name: str, *, budget: str | None = None):
        payload = {
            "title": f"Isolated {self.run_id} {name}",
            "questions": [
                {
                    "id": qid,
                    "stem": f"Synthetic question {qid}",
                    "reference": "Synthetic complete correct solution",
                    "max_score": "10",
                }
                for qid in QUESTIONS
            ],
        }
        if budget:
            payload["budget_cny"] = budget
        response = self.api.post("/v1/exams", json=payload)
        response.raise_for_status()
        exam_id = response.json()["id"]
        for _ in range(600):
            exam = self.api.get(f"/v1/exams/{exam_id}").json()
            if exam["state"] == "ready":
                return exam_id
            if exam["state"] == "failed":
                raise RuntimeError(f"Exam preparation failed: {exam['error']}")
            time.sleep(0.1)
        raise RuntimeError("Exam did not become ready")

    def candidates(self, name: str, count: int, *, offset: int):
        result = []
        start = time.monotonic()
        for index in range(count):
            image = synthetic_paper(offset + index)
            filename = f"{name}-{index:04}.png"
            (self.directory / "inputs" / filename).write_bytes(image)
            response = self.api.post("/v1/assets", files={"file": (filename, image, "image/png")})
            response.raise_for_status()
            result.append(
                {
                    "candidate_id": f"{self.run_id}:{name}:{index:04}",
                    "kind": "paper",
                    "pages": [response.json()[0]["id"]],
                }
            )
        self.event(
            "assets_uploaded",
            phase=name,
            count=count,
            wall_seconds=round(time.monotonic() - start, 3),
        )
        return result

    def submit(self, name: str, exam_id: str, candidates: list[dict]):
        payload = {"exam_id": exam_id, "candidates": candidates}
        (self.directory / f"{name}-submission.json").write_text(json.dumps(payload, indent=2))
        start = time.monotonic()
        response = self.api.post(
            "/v1/jobs", json=payload, headers={"Idempotency-Key": f"{self.run_id}:{name}"}
        )
        elapsed = time.monotonic() - start
        response.raise_for_status()
        return response.json()["id"], elapsed, payload

    def snapshot(self, job_id: str):
        counts = self.query(
            "SELECT count(*), count(DISTINCT candidate_id) FROM candidates WHERE job_id=%s",
            (job_id,),
        )[0]
        answers = self.query(
            "SELECT id, state, result, intermediate, error FROM answers WHERE job_id=%s", (job_id,)
        )
        good = sum(
            state == "scored" and Decimal(result["grade"]["total_score"]) == 10
            for _, state, result, _, _ in answers
        )
        return {
            "candidates": counts[0],
            "unique_candidate_ids": counts[1],
            "answers": len(answers),
            "unique_answer_ids": len({row[0] for row in answers}),
            "answer_states": self.states(job_id),
            "correct_scores": good,
            "model_calls_by_role_state": [
                {"role": role, "state": state, "count": count}
                for role, state, count in self.query(
                    "SELECT role,state,count(*) FROM model_calls WHERE job_id=%s GROUP BY role,state ORDER BY role,state",
                    (job_id,),
                )
            ],
            "error_counts": dict(
                Counter((error or "").split(":", 1)[0] for _, _, _, _, error in answers if error)
            ),
        }

    def query_results(self, name: str, job_id: str):
        candidates, answer_ids = set(), set()
        output_path = self.directory / f"{name}-api-results.jsonl"
        with output_path.open("w") as output:
            for offset in range(0, 1000, 100):
                response = self.api.get(
                    f"/v1/jobs/{job_id}/results", params={"offset": offset, "limit": 100}
                )
                response.raise_for_status()
                rows = response.json()
                for candidate in rows:
                    candidates.add(candidate["candidate_id"])
                    answer_ids.update(answer["id"] for answer in candidate["answers"])
                    output.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                if len(rows) < 100:
                    break
        return {
            "api_queryable_candidates": len(candidates),
            "api_queryable_answers": len(answer_ids),
        }

    def run_success(self, name: str, count: int, cap: int, offset: int):
        self.idle_mock()
        self.configure_mock(reset=True, faults={}, usage_enabled=True, delay_seconds=0.2)
        start = time.monotonic()
        exam_id = self.create_exam(name)
        candidates = self.candidates(name, count, offset=offset)
        submitted = time.monotonic()
        job_id, latency, payload = self.submit(name, exam_id, candidates)
        self.wait_job(job_id)
        completed = time.monotonic()
        duplicate = self.api.post(
            "/v1/jobs", json=payload, headers={"Idempotency-Key": f"{self.run_id}:{name}"}
        )
        duplicate.raise_for_status()
        before = self.query("SELECT count(*) FROM model_calls WHERE job_id=%s", (job_id,))[0][0]
        # Re-publish both unit types through the real transactional Outbox.
        candidate_id = self.query("SELECT id FROM candidates WHERE job_id=%s LIMIT 1", (job_id,))[
            0
        ][0]
        answer_id = self.query("SELECT id FROM answers WHERE job_id=%s LIMIT 1", (job_id,))[0][0]
        for task, unit in (("prepare_candidate", candidate_id), ("grade_answer", answer_id)):
            self.conn.execute(
                "INSERT INTO outbox(id,task,unit_id,available_at,created_at) VALUES (%s,%s,%s,now(),now())",
                (hashlib.md5(f"{name}:{unit}".encode()).hexdigest(), task, unit),
            )
        for _ in range(50):
            if not self.query("SELECT count(*) FROM outbox")[0][0]:
                break
            time.sleep(0.1)
        time.sleep(0.5)
        after = self.query("SELECT count(*) FROM model_calls WHERE job_id=%s", (job_id,))[0][0]
        mock = self.mock.get("/stats").json()
        result = (
            self.snapshot(job_id)
            | self.query_results(name, job_id)
            | {
                "job_id": job_id,
                "exam_id": exam_id,
                "model_cap": cap,
                "http_calls": mock["calls_by_schema"],
                "peak_inflight": mock["peak_inflight"],
                "mock": mock,
                "submit_latency_seconds": round(latency, 4),
                "job_wall_seconds": round(completed - submitted, 3),
                "phase_wall_seconds": round(time.monotonic() - start, 3),
                "answers_per_second": round(5 * count / (completed - submitted), 3),
                "duplicate_submit_same_job": duplicate.json()["id"] == job_id,
                "duplicate_delivery_extra_calls": after - before,
                "pending_outbox": self.query("SELECT count(*) FROM outbox")[0][0],
            }
        )
        result["acceptance"] = (
            assess_full_run(result)
            if count == 1000
            else {
                "all_answers_scored": result["answer_states"] == {"scored": 5 * count},
                "actual_call_volume": mock["calls_by_schema"]
                == {
                    "Rubric": 10,
                    "PageLayout": count,
                    "Transcript": 5 * count,
                    "Judgment": 5 * count,
                },
                "concurrent_within_cap": 1 < mock["peak_inflight"] <= cap,
                "correct_scores": result["correct_scores"] == 5 * count,
            }
        )
        self.report["phases"][name] = result
        self.event("phase_completed", phase=name, result=result)
        self.save()

    def run_faults(self):
        name = "fault_retry_32"
        self.idle_mock()
        config = {"seed": "fault-replay-v1", "probability": 0.65}
        self.configure_mock(
            reset=True, faults=config, usage_enabled=True, delay_seconds=0.05, timeout_seconds=2
        )
        exam_id = self.create_exam(name)
        candidates = self.candidates(name, 20, offset=20000)
        job_id, latency, _ = self.submit(name, exam_id, candidates)
        self.wait_job(job_id)
        before = self.snapshot(job_id)
        stats_before = self.mock.get("/stats").json()
        records_before = self.query(
            "SELECT id,state,intermediate,result FROM answers WHERE job_id=%s", (job_id,)
        )
        failed = {
            ident: intermediate
            for ident, state, intermediate, _ in records_before
            if state == "failed"
        }
        succeeded = {
            ident: result for ident, state, _, result in records_before if state == "scored"
        }
        self.configure_mock(faults={})
        response = self.api.post(f"/v1/jobs/{job_id}/retry")
        response.raise_for_status()
        self.wait_job(job_id)
        stats_after = self.mock.get("/stats").json()
        after_rows = {
            ident: (state, intermediate, result)
            for ident, state, intermediate, result in self.query(
                "SELECT id,state,intermediate,result FROM answers WHERE job_id=%s", (job_id,)
            )
        }
        delta = {
            schema: stats_after["calls_by_schema"].get(schema, 0)
            - stats_before["calls_by_schema"].get(schema, 0)
            for schema in ("Rubric", "PageLayout", "Transcript", "Judgment")
        }
        result = {
            "job_id": job_id,
            "exam_id": exam_id,
            "fault_config": config,
            "submit_latency_seconds": round(latency, 4),
            "before_retry": before,
            "after_retry": self.snapshot(job_id),
            "mock_before_retry": stats_before,
            "mock_after_retry": stats_after,
            "retry_http_status": response.status_code,
            "retry_call_delta": delta,
            "failed_units": len(failed),
            "previous_successes": len(succeeded),
        }
        result["acceptance"] = {
            "all_five_fault_types_observed": set(stats_before["faults"]) == set(FAULT_TYPES),
            "failed_states_retained_until_explicit_retry": bool(failed) and bool(succeeded),
            "only_failed_judgments_recalled": delta
            == {"Rubric": 0, "PageLayout": 0, "Transcript": 0, "Judgment": len(failed)},
            "successful_transcript_checkpoints_reused": all(
                after_rows[ident][1]["primary_transcript"] == intermediate["primary_transcript"]
                for ident, intermediate in failed.items()
            ),
            "successful_answers_unchanged": all(
                after_rows[ident][2] == result for ident, result in succeeded.items()
            ),
            "all_answers_correct_after_retry": result["after_retry"]["correct_scores"] == 100,
            "failed_call_audit_retained": self.query(
                "SELECT count(*) FROM model_calls WHERE job_id=%s AND state='failed'", (job_id,)
            )[0][0]
            == len(failed),
        }
        self.report["phases"][name] = result
        self.event("phase_completed", phase=name, result=result)
        self.save()

    def run_budget(self):
        name = "exam_budget_concurrent_32"
        self.idle_mock()
        self.configure_mock(reset=True, faults={}, usage_enabled=True, delay_seconds=0.2)
        exam_id = self.create_exam(name, budget="0.02000000")
        groups = [self.candidates(f"{name}_{n}", 12, offset=30000 + n * 100) for n in range(2)]
        self.configure_mock(usage_enabled=False, delay_seconds=0.5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            submissions = list(
                pool.map(
                    lambda item: self.submit(f"{name}_{item[0]}", exam_id, item[1]),
                    enumerate(groups),
                )
            )
        samples = []
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            committed, budget = self.query(
                "SELECT committed_cost,budget_cny FROM exams WHERE id=%s", (exam_id,)
            )[0]
            running = self.query(
                "SELECT count(*) FROM model_calls WHERE exam_id=%s AND state='running'", (exam_id,)
            )[0][0]
            samples.append({"committed": str(committed), "running_calls": running})
            if all(
                all(state in {"failed", "scored", "ungradable"} for state in self.states(job_id))
                and self.states(job_id)
                for job_id, _, _ in submissions
            ):
                break
            time.sleep(0.1)
        for job_id, _, _ in submissions:
            self.wait_job(job_id)
        self.idle_mock()
        committed, budget = self.query(
            "SELECT committed_cost,budget_cny FROM exams WHERE id=%s", (exam_id,)
        )[0]
        unknown_count, unknown_reserved = self.query(
            "SELECT count(*),COALESCE(sum(reserved_cost),0) FROM model_calls WHERE exam_id=%s AND cost IS NULL",
            (exam_id,),
        )[0]
        known = self.query(
            "SELECT COALESCE(sum(cost),0) FROM model_calls WHERE exam_id=%s", (exam_id,)
        )[0][0]
        jobs = [
            self.snapshot(job_id) | {"job_id": job_id, "submit_latency_seconds": round(latency, 4)}
            for job_id, latency, _ in submissions
        ]
        budget_errors = self.query(
            "SELECT count(*) FROM candidates WHERE job_id = ANY(%s) AND error LIKE 'BudgetExceeded:%%'",
            ([job_id for job_id, _, _ in submissions],),
        )[0][0]
        result = {
            "exam_id": exam_id,
            "budget_cny": str(budget),
            "committed_cost_cny": str(committed),
            "known_cost_cny": str(known),
            "unknown_usage_calls": unknown_count,
            "unknown_usage_reserved_cny": str(unknown_reserved),
            "budget_exceeded_candidate_units": budget_errors,
            "concurrent_reservation_samples": samples,
            "jobs": jobs,
            "mock": self.mock.get("/stats").json(),
            "fault_config": {"usage_enabled": False, "delay_seconds": 0.5},
        }
        result["acceptance"] = {
            "shared_exam_budget_never_exceeded": all(
                Decimal(sample["committed"]) <= budget for sample in samples
            )
            and committed <= budget,
            "concurrent_reservations_observed": max(sample["running_calls"] for sample in samples)
            > 1,
            "budget_stopped_work_across_two_jobs": len(jobs) == 2
            and budget_errors > 0
            and all(job["answer_states"] == {"failed": 60} for job in jobs),
            "unknown_usage_reservation_retained": unknown_count > 0
            and unknown_reserved > 0
            and committed == known + unknown_reserved,
        }
        self.report["phases"][name] = result
        self.event("phase_completed", phase=name, result=result)
        self.save()
        self.configure_mock(usage_enabled=True)

    def start_recovery(self):
        from redis import Redis

        name = "worker_abrupt_recovery_32"
        self.idle_mock()
        self.configure_mock(reset=True, faults={}, usage_enabled=True, delay_seconds=0.7)
        exam_id = self.create_exam(name)
        candidates = self.candidates(name, 2, offset=40000)
        job_id, latency, _ = self.submit(name, exam_id, candidates)
        deadline = time.monotonic() + 30
        running = []
        while time.monotonic() < deadline:
            running = self.query(
                "SELECT id FROM answers WHERE job_id=%s AND state='running'", (job_id,)
            )
            running_calls = self.query(
                "SELECT count(*) FROM model_calls WHERE job_id=%s AND state='running'", (job_id,)
            )[0][0]
            if len(running) == 10 and running_calls >= 2:
                break
            time.sleep(0.02)
        if len(running) < 2:
            raise RuntimeError("No running answers observed for abrupt worker test")
        killed_pid = self.processes["grading"].pid
        self.stop("grading", abrupt=True)
        leases = [
            {"unit_id": ident, "lease_until": lease.isoformat()}
            for ident, lease in self.query(
                "SELECT id,lease_until FROM answers WHERE job_id=%s AND state='running'", (job_id,)
            )
        ]
        active_key = "ai4edu:limit:{" + hashlib.sha256(MODEL.encode()).hexdigest()[:16] + "}:active"
        redis = Redis.from_url(self.env["AI4EDU_REDIS_URL"])
        tickets = [
            {"ticket": ident.decode(), "expires_epoch": expiry}
            for ident, expiry in redis.zrange(active_key, 0, -1, withscores=True)
        ]
        self.event(
            "natural_lease_expiry_wait_started",
            job_id=job_id,
            leases=leases,
            note="Both SQL task leases and Redis admission tickets keep their real 300-second timestamps; no forced expiry",
        )
        restarted_pid = self.worker("grading", "grading", 48).pid
        self.idle_mock()
        return {
            "job_id": job_id,
            "exam_id": exam_id,
            "submit_latency_seconds": round(latency, 4),
            "killed_worker_pid": killed_pid,
            "restarted_worker_pid": restarted_pid,
            "leases_at_kill": leases,
            "admission_tickets_at_kill": tickets,
            "admission_key": active_key,
            "kill_monotonic": time.monotonic(),
            "mock_at_kill": self.mock.get("/stats").json(),
        }

    def finish_recovery(self, context: dict):
        from redis import Redis

        name, job_id = "worker_abrupt_recovery_32", context["job_id"]
        self.wait_job(job_id, timeout=360)
        retained_running = self.query(
            "SELECT count(*) FROM model_calls WHERE job_id=%s AND state='running'", (job_id,)
        )[0][0]
        elapsed = round(time.monotonic() - context.pop("kill_monotonic"), 3)
        redis = Redis.from_url(self.env["AI4EDU_REDIS_URL"])
        remaining_tickets = [
            ticket["ticket"]
            for ticket in context["admission_tickets_at_kill"]
            if redis.zscore(context["admission_key"], ticket["ticket"]) is not None
        ]
        image_ids = {
            asset
            for (prepared,) in self.query("SELECT assets FROM answers WHERE job_id=%s", (job_id,))
            for region in prepared
            for asset in (region["asset_id"], region["source_asset_id"])
        }
        http_records = [
            json.loads(line)
            for line in (self.directory / "model-http-calls.jsonl").read_text().splitlines()
        ]
        own_records = [record for record in http_records if set(record["image_ids"]) & image_ids]
        result = (
            self.snapshot(job_id)
            | context
            | {
                "natural_recovery_wall_seconds": elapsed,
                "real_configured_lease_seconds": 300,
                "sql_lease_accelerated": False,
                "redis_lease_accelerated": False,
                "old_admission_tickets_remaining": remaining_tickets,
                "retained_unknown_running_call_audits": retained_running,
                "actual_http_calls_by_schema": dict(
                    Counter(record["schema"] for record in own_records)
                ),
            }
        )
        result["acceptance"] = {
            "own_worker_actually_sigkilled_and_restarted": context["killed_worker_pid"]
            != context["restarted_worker_pid"],
            "real_beat_recovered_naturally_expired_units": bool(context["leases_at_kill"])
            and result["answer_states"] == {"scored": 10}
            and elapsed >= 295,
            "admission_tickets_naturally_expired": bool(context["admission_tickets_at_kill"])
            and not remaining_tickets,
            "all_recovered_scores_correct": result["correct_scores"] == 10,
            "unique_results_after_redelivery": result["unique_answer_ids"] == 10,
        }
        self.report["phases"][name] = result
        self.event("phase_completed", phase=name, result=result)
        self.save()

    def cleanup(self):
        for role in ("beat", "preparation", "grading", "control", "api", "mock"):
            self.stop(role)
        self.api.close()
        self.mock.close()
        if self.conn is not None:
            self.conn.close()
        for stream in self.log_handles:
            stream.close()
        from redis import Redis

        created = set()
        if self.owns_redis:
            redis = Redis.from_url(self.env["AI4EDU_REDIS_URL"])
            created = set(redis.scan_iter()) - self.baseline_redis_keys
            if created:
                redis.delete(*created)
        self.report["cleanup"] = {
            "all_owned_processes_stopped": all("stopped_at" in item for item in self.process_log),
            "removed_own_new_redis_keys": len(created),
            "database_and_artifacts_retained": True,
        }

    def run(self):
        try:
            self.start()
            self.run_success("diagnostic_16", 16, 16, 0)
            self.change_cap(32)
            self.run_success("full_32", 8 if self.smoke else 1000, 32, 1000)
            recovery = self.start_recovery()
            self.run_faults()
            self.run_budget()
            self.finish_recovery(recovery)
            self.report["all_acceptance_passed"] = all(
                all(phase["acceptance"].values()) for phase in self.report["phases"].values()
            )
            self.report["database_row_counts"] = {
                table: self.query(f"SELECT count(*) FROM {table}")[0][0]
                for table in (
                    "assets",
                    "exams",
                    "jobs",
                    "candidates",
                    "answers",
                    "model_calls",
                    "outbox",
                )
            }
            self.sample_resources()
        except BaseException as exc:
            self.report["error"] = f"{type(exc).__name__}: {exc}"
            self.event("run_failed", error=self.report["error"])
            raise
        finally:
            self.cleanup()
            self.save()
            self.write_markdown()
        return self.report

    def write_markdown(self):
        lines = [
            f"# 真实栈并发与稳定性测试：{self.run_id}",
            "",
            "本报告使用独立 HTTP 兼容模型服务和加速合成响应，仅验证真实 API→Outbox→Celery→PostgreSQL/Redis 执行链路。吞吐量不能用于推算真实付费模型性能。",
            "",
            f"数据库：`{self.database}`；Redis：DB 3；API：18090；模型服务：18091。",
            "",
            f"总墙钟时间：{self.report.get('wall_seconds')} 秒。",
            "",
            "| 场景 | 答案状态 | 实测模型 HTTP 峰值 | 提交耗时（秒） | 验收 |",
            "|---|---|---:|---:|---|",
        ]
        for name, result in self.report["phases"].items():
            states = result.get(
                "answer_states", result.get("after_retry", {}).get("answer_states", "见 JSON")
            )
            peak = result.get(
                "peak_inflight",
                result.get("mock", result.get("mock_after_retry", {})).get("peak_inflight", "—"),
            )
            passed = all(result["acceptance"].values())
            lines.append(
                f"| {name} | {states} | {peak} | {result.get('submit_latency_seconds', '—')} | {'通过' if passed else '未通过'} |"
            )
        lines.extend(
            [
                "",
                "完整数据、每个验收条件、故障配置、重试前后状态、预算预留、进程 PID/停止记录、资源采样均保存在 `report.json`。逐调用日志为 `model-http-calls.jsonl`，真实 API 查询结果为 `*-api-results.jsonl`。",
                "",
                "故障场景覆盖确定性 429、503、超时、非法 JSON 和无效证据；失败保留后仅通过 `/v1/jobs/{id}/retry` 重试，已成功的转写检查点应复用。预算场景使用同一 ExamInput.budget_cny 下并发提交两个批次，缺少 usage 的请求继续保留费用预留。",
                "",
                "崩溃恢复执行了独立评分 Worker 的 SIGKILL 和重启；SQL 任务租约与 Redis 模型准入票据均等待真实 300 秒自然到期，没有加速或修改到期时间，由真实 Beat 恢复任务。崩溃中的模型调用保留未知用量/预留审计记录。",
                "",
                "复现命令：",
                "",
                "```bash",
                ".venv/bin/python scripts/run_loadtest.py",
                "```",
                "",
                f"进程清理：{json.dumps(self.report.get('cleanup', {}), ensure_ascii=False)}。数据库和报告保留以便独立查询。",
            ]
        )
        if "error" in self.report:
            lines.extend(["", f"运行错误：`{self.report['error']}`。"])
        (self.directory / "report.zh-CN.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mock = sub.add_parser("mock", help="Independent local HTTP model; no provider access")
    mock.add_argument("--journal", type=Path)
    run = sub.add_parser("run", help="Start and stop only an isolated local stack")
    run.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    run.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "mock":
        uvicorn.run(
            create_mock_app(journal=args.journal),
            host="127.0.0.1",
            port=18091,
            log_level="warning",
            access_log=False,
        )
    else:
        stack = IsolatedStack(Path(__file__).resolve().parents[2], args.run_id, smoke=args.smoke)
        result = stack.run()
        print(
            json.dumps(
                {
                    "report": str(stack.directory / "report.json"),
                    "all_acceptance_passed": result.get("all_acceptance_passed"),
                }
            ),
            flush=True,
        )
        if not result.get("all_acceptance_passed"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
