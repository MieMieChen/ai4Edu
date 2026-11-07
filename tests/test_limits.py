import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest

from ai4edu.limits import RateLimitWait, RedisLimiter


def test_request_and_token_limits_are_shared_between_instances():
    redis = fakeredis.FakeRedis()
    a = RedisLimiter(redis, concurrency=10, rpm=2, tpm=100, lease_seconds=60)
    b = RedisLimiter(redis, concurrency=10, rpm=2, tpm=100, lease_seconds=60)
    first = a.acquire("model", 60)
    with pytest.raises(RateLimitWait):
        b.acquire("model", 60)
    a.release("model", first, 20)
    second = b.acquire("model", 60)
    b.release("model", second, 20)
    with pytest.raises(RateLimitWait):
        a.acquire("model", 1)


def test_global_concurrency_under_one_thousand_tasks():
    redis = fakeredis.FakeRedis()
    limiters = [
        RedisLimiter(redis, concurrency=3, rpm=10000, tpm=100000, lease_seconds=60)
        for _ in range(4)
    ]
    lock = threading.Lock()
    active = maximum = 0

    def work(index):
        nonlocal active, maximum
        limiter = limiters[index % len(limiters)]
        while True:
            try:
                token = limiter.acquire("shared-model", 1)
                break
            except RateLimitWait:
                time.sleep(0.001)
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.001)
        with lock:
            active -= 1
        limiter.release("shared-model", token, 1)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(work, range(1000)))
    assert maximum <= 3
    assert maximum > 1


def test_single_request_over_token_limit_fails_immediately():
    limiter = RedisLimiter(fakeredis.FakeRedis(), concurrency=1, rpm=10, tpm=10, lease_seconds=60)
    with pytest.raises(ValueError, match="TPM"):
        limiter.acquire("model", 11)
