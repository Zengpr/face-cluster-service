#!/usr/bin/env python3
"""Load test: generate synthetic images and hammer the /cluster endpoint.

Usage:
  python scripts/load_test.py [--url URL] [--concurrency N] [--requests N]

Outputs:
  - console summary (p50/p90/p99 latency, throughput, error rate)
  - HTML report at jmeter/report.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
JMETER_DIR = REPO_ROOT / "jmeter"


def _make_jpeg_bytes() -> bytes:
    """Minimal valid JPEG via PIL if available, else raw."""
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (200, 200), (200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\xc8\x00\xc8\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x11\x04\x05!1\x06\x12\x13AQ\x07aq\x12\x14\"2\x81\x91\xa1#B\x15\xb1\xc1\xd1\x166\xf0\x82$br\xcc\xe1\xff\xda\x00\x08\x01\x01\x00\x00?\x00*\xbd\xa5\x00\xff\xd9"


def _do_request(url: str, threshold: float, timeout: int) -> tuple[str, int, float]:
    """Single /cluster POST. Returns (status, size_bytes, elapsed_sec)."""
    data = _make_jpeg_bytes()
    files = [("files", (f"img_{i}.jpg", data, "image/jpeg")) for i in range(3)]
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{url}/cluster",
            files=files,
            data={"threshold": str(threshold)},
            headers={"X-Demo-Mode": "true", "User-Agent": "loadtest"},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        return ("ok", r.status_code, elapsed)
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t0
        return ("timeout", 0, elapsed)
    except requests.exceptions.ConnectionError:
        elapsed = time.perf_counter() - t0
        return ("connection_error", 0, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return (f"error:{exc}", 0, elapsed)


def run_load_test(url: str, concurrency: int, total_requests: int, threshold: float, timeout: int, warmup: int = 5):
    print(f"Load test: concurrency={concurrency}, requests={total_requests}, url={url}")
    print(f"Warmup: {warmup}s  |  Timeout per request: {timeout}s\n")

    results: list[tuple[str, int, float]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_do_request, url, threshold, timeout) for _ in range(total_requests)]
        done = 0
        t_start = time.perf_counter()
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 25 == 0 or done == total_requests:
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{total_requests}] rate={rate:.1f} req/s", end="\r")
        wall = time.perf_counter() - t_start
        print()

    # Analysis
    ok = [r for r in results if r[0] == "ok"]
    errors = [r for r in results if r[0] != "ok"]
    lats = sorted([r[2] for r in ok])

    p50 = lats[len(lats) // 2] if lats else 0
    p90 = lats[int(len(lats) * 0.9)] if lats else 0
    p99 = lats[int(len(lats) * 0.99)] if lats else 0

    throughput = len(ok) / wall if wall > 0 else 0
    error_rate = len(errors) / len(results) * 100 if results else 0

    print(f"\n{'='*60}")
    print(f"RESULTS — {concurrency} threads × {total_requests} requests")
    print(f"{'='*60}")
    print(f"  Wall time:      {wall:.1f}s")
    print(f"  OK:             {len(ok)}")
    print(f"  Errors:         {len(errors)} ({error_rate:.1f}%)")
    print(f"  Throughput:     {throughput:.1f} req/s")
    print(f"  p50 latency:    {p50*1000:.0f} ms")
    print(f"  p90 latency:    {p90*1000:.0f} ms")
    print(f"  p99 latency:    {p99*1000:.0f} ms")

    # HTML report
    JMETER_DIR.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Load Test Report</title>
<style>body{{font-family:sans-serif;margin:40px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:8px 12px;text-align:right}}
th{{background:#f5f5f5}}tr.pass td{{background:#e8ffe8}}tr.fail td{{background:#ffe8e8}}</style>
</head><body>
<h1>Load Test Report</h1>
<p>Target: <code>{url}</code></p>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr class="pass"><td>Concurrency</td><td>{concurrency}</td></tr>
<tr class="pass"><td>Total requests</td><td>{total_requests}</td></tr>
<tr class="pass"><td>Wall time</td><td>{wall:.1f}s</td></tr>
<tr class="pass"><td>OK responses</td><td>{len(ok)}</td></tr>
<tr class="pass"><td>Error rate</td><td>{error_rate:.1f}%</td></tr>
<tr class="pass"><td>Throughput</td><td>{throughput:.1f} req/s</td></tr>
<tr class="pass"><td>p50 latency</td><td>{p50*1000:.0f} ms</td></tr>
<tr class="pass"><td>p90 latency</td><td>{p90*1000:.0f} ms</td></tr>
<tr class="pass"><td>p99 latency</td><td>{p99*1000:.0f} ms</td></tr>
</table>
<h2>Latency distribution</h2>
<pre>{json.dumps({f"p{p}": f"{lats[int(len(lats)*p/100)]*1000:.0f}ms" for p in [50,75,90,95,99]}, indent=2)}</pre>
</body></html>"""

    report_path = JMETER_DIR / "report.html"
    report_path.write_text(html)
    print(f"\nHTML report: {report_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8765")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()
    run_load_test(args.url, args.concurrency, args.requests, threshold=0.6, timeout=120)
