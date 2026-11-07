import hashlib
import time
from uuid import uuid4

from redis import Redis


class RateLimitWait(Exception):
    def __init__(self, seconds: float):
        self.seconds = seconds
        super().__init__(f"模型调度容量不足，{seconds:.1f}秒后重新入队")


ACQUIRE = """
local active, window, sizes, total = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local now, ident, tokens = tonumber(ARGV[1]), ARGV[2], tonumber(ARGV[3])
local expired = redis.call('ZRANGEBYSCORE', window, '-inf', now - 60)
for _, id in ipairs(expired) do
  local n = tonumber(redis.call('HGET', sizes, id) or '0')
  redis.call('DECRBY', total, n)
  redis.call('HDEL', sizes, id)
end
redis.call('ZREMRANGEBYSCORE', window, '-inf', now - 60)
redis.call('ZREMRANGEBYSCORE', active, '-inf', now)
if redis.call('ZCARD', active) >= tonumber(ARGV[4]) then return 1 end
if redis.call('ZCARD', window) >= tonumber(ARGV[5]) or tonumber(redis.call('GET', total) or '0') + tokens > tonumber(ARGV[6]) then
  local first = redis.call('ZRANGE', window, 0, 0, 'WITHSCORES')
  return math.max(1, math.ceil(tonumber(first[2]) + 60 - now))
end
redis.call('ZADD', active, now + tonumber(ARGV[7]), ident)
redis.call('ZADD', window, now, ident)
redis.call('HSET', sizes, ident, tokens)
redis.call('INCRBY', total, tokens)
for _, key in ipairs(KEYS) do redis.call('EXPIRE', key, math.ceil(tonumber(ARGV[7]) + 120)) end
return 0
"""

RELEASE = """
redis.call('ZREM', KEYS[1], ARGV[1])
local previous = redis.call('HGET', KEYS[3], ARGV[1])
if previous and tonumber(ARGV[2]) >= 0 then
  redis.call('INCRBY', KEYS[4], tonumber(ARGV[2]) - tonumber(previous))
  redis.call('HSET', KEYS[3], ARGV[1], ARGV[2])
end
return 1
"""


class RedisLimiter:
    def __init__(self, redis: Redis, *, concurrency: int, rpm: int, tpm: int, lease_seconds: int):
        self.redis = redis
        self.concurrency, self.rpm, self.tpm = concurrency, rpm, tpm
        self.lease_seconds = lease_seconds

    def keys(self, model: str) -> list[str]:
        # Hash tags keep a model's atomic keys together on Redis Cluster.
        prefix = "ai4edu:limit:{" + hashlib.sha256(model.encode()).hexdigest()[:16] + "}"
        return [f"{prefix}:{suffix}" for suffix in ("active", "window", "sizes", "total")]

    def acquire(self, model: str, estimated_tokens: int) -> str:
        if estimated_tokens > self.tpm:
            raise ValueError("单次请求的 Token 预留量超过配置的 TPM，请调整限额或输出长度")
        ident = uuid4().hex
        wait = self.redis.eval(
            ACQUIRE,
            4,
            *self.keys(model),
            time.time(),
            ident,
            estimated_tokens,
            self.concurrency,
            self.rpm,
            self.tpm,
            self.lease_seconds,
        )
        if wait:
            raise RateLimitWait(float(wait))
        return ident

    def release(self, model: str, ident: str, actual_tokens: int | None):
        self.redis.eval(
            RELEASE, 4, *self.keys(model), ident, -1 if actual_tokens is None else actual_tokens
        )
