"""Run the local API, three Celery queues and scheduler against explicit dependencies."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from redis import Redis

from ai4edu.config import get_settings
from ai4edu.db import Database


def main():
    parser = argparse.ArgumentParser(description="Run or expand the local grading workers")
    parser.add_argument(
        "--add-workers",
        action="store_true",
        help="Add workers without interrupting active requests",
    )
    parser.add_argument("--worker-tag", default="extra")
    parser.add_argument("--grading-concurrency", type=int, default=8)
    parser.add_argument("--preparation-concurrency", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    settings = get_settings()
    Database(settings.database_url).create_all()
    Redis.from_url(settings.redis_url).ping()
    directory = root / "data/runtime"
    directory.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    services = {
        "api": [
            python,
            "-m",
            "uvicorn",
            "ai4edu.api:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            "18000",
        ],
        **{
            name: [
                python,
                "-m",
                "celery",
                "-A",
                "ai4edu.tasks:celery",
                "worker",
                "-Q",
                name,
                "--pool",
                "threads",
                "--concurrency",
                str(concurrency),
                "--hostname",
                f"ai4edu-{name}@%h",
                "--loglevel",
                "INFO",
            ]
            for name, concurrency in [("preparation", 4), ("grading", 32), ("control", 1)]
        },
        "scheduler": [
            python,
            "-m",
            "celery",
            "-A",
            "ai4edu.tasks:celery",
            "beat",
            "--schedule",
            str(directory / "celerybeat-schedule"),
            "--loglevel",
            "INFO",
        ],
    }
    if args.add_workers:
        services = {
            f"{queue}_{args.worker_tag}": [
                python,
                "-m",
                "celery",
                "-A",
                "ai4edu.tasks:celery",
                "worker",
                "-Q",
                queue,
                "--pool",
                "threads",
                "--concurrency",
                str(concurrency),
                "--hostname",
                f"ai4edu-{queue}-{args.worker_tag}@%h",
                "--loglevel",
                "INFO",
            ]
            for queue, concurrency in [
                ("preparation", args.preparation_concurrency),
                ("grading", args.grading_concurrency),
            ]
            if concurrency > 0
        }
    registry_name = (
        "additional-services.json"
        if args.worker_tag == "extra"
        else f"additional-{args.worker_tag}-services.json"
    )
    registry = directory / (registry_name if args.add_workers else "services.json")
    children = {}

    def stop(_signum=None, _frame=None):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for name, command in services.items():
            with (directory / f"{name}.log").open("a") as log:
                children[name] = subprocess.Popen(
                    command, cwd=root, stdout=log, stderr=subprocess.STDOUT
                )
        registry.write_text(
            json.dumps(
                {
                    "supervisor_pid": os.getpid(),
                    "url": "http://127.0.0.1:18000",
                    "services": {name: process.pid for name, process in children.items()},
                },
                indent=2,
            )
        )
        print(
            "Additional workers started; existing requests continue."
            if args.add_workers
            else "Local services started at http://127.0.0.1:18000; logs: data/runtime; Ctrl-C drains application processes.",
            flush=True,
        )
        while True:
            for name, process in children.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{name} exited with status {process.returncode}; see data/runtime/{name}.log"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Draining active worker tasks before exit.", flush=True)
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        for process in children.values():
            try:
                process.wait(timeout=settings.http_timeout * 4 + 60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        registry.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
