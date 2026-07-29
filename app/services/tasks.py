"""Async task layer (Redis-backed).

The async endpoint stores results under ``task:result:<id>`` with TTL
``async_result_ttl_sec``. Worker state is held in Python in an asyncio
Task plus a Redis key for resilience across restarts. For the demo we
spawn tasks in-process; production would push to a dedicated queue
(Arq / RQ / Celery) — see ``docs/ARCHITECTURE.md``.
"""
from __future__ import annotations

import json
import secrets
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# In-memory registry so that within a single container lifecycle we
# can also return 'pending' for tasks whose results aren't yet in Redis.
_running_tasks: dict[str, dict[str, Any]] = {}


def _client() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def submit_task(payload: dict) -> str:
    task_id = secrets.token_urlsafe(12)
    _running_tasks[task_id] = {"state": "pending"}
    return task_id


def set_state(task_id: str, state: str) -> None:
    if task_id in _running_tasks:
        _running_tasks[task_id]["state"] = state


async def store_result(task_id: str, result: dict) -> None:
    client = _client()
    try:
        await client.setex(
            f"task:result:{task_id}",
            settings.async_result_ttl_sec,
            json.dumps({"state": "succeeded", "result": result}),
        )
    except Exception as exc:
        log.error("redis.store_failed", err=str(exc))
    finally:
        await client.aclose()
    _running_tasks.pop(task_id, None)


async def store_failure(task_id: str, error_payload: dict) -> None:
    client = _client()
    try:
        await client.setex(
            f"task:result:{task_id}",
            settings.async_result_ttl_sec,
            json.dumps({"state": "failed", "error": error_payload}),
        )
    except Exception as exc:
        log.error("redis.store_failed", err=str(exc))
    finally:
        await client.aclose()
    _running_tasks.pop(task_id, None)


async def fetch_status(task_id: str) -> dict | None:
    client = _client()
    try:
        raw = await client.get(f"task:result:{task_id}")
        if raw:
            return json.loads(raw)
    except Exception as exc:
        log.warning("redis.fetch_failed", err=str(exc))
        return None
    finally:
        await client.aclose()
    if task_id in _running_tasks:
        return {"state": _running_tasks[task_id]["state"]}
    return None
