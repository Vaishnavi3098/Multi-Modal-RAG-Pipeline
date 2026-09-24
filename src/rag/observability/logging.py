"""Structured logging and simple metrics collection."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


class Metrics:
    """Lightweight in-process metrics (replace with Prometheus/OTel in prod)."""

    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timings: Dict[str, list] = {}

    def incr(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def observe(self, name: str, value_ms: float) -> None:
        self.timings.setdefault(name, []).append(value_ms)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"counters": dict(self.counters)}
        for k, vals in self.timings.items():
            if not vals:
                continue
            out[k] = {
                "count": len(vals),
                "avg_ms": sum(vals) / len(vals),
                "p95_ms": sorted(vals)[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0],
            }
        return out


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics


@contextmanager
def timed(metric_name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        _metrics.observe(metric_name, (time.perf_counter() - start) * 1000)
