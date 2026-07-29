"""HTTP smoke test against a running face-cluster container.

Usage:
    BASE=http://localhost:8000 python tests/test_integration_live.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import httpx


BASE = os.environ.get("BASE", "http://localhost:8000").rstrip("/")
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", "tests/data/images")


def main() -> int:
    failures: list[str] = []
    print(f"[smoke] hitting {BASE}")

    r = httpx.get(f"{BASE}/health", timeout=10.0)
    assert r.status_code == 200, r.text
    print(f"  /health -> {r.json()}")

    r = httpx.post(f"{BASE}/cluster", files=[], timeout=30.0)
    status = "OK" if r.status_code == 400 else "FAIL"
    print(f"  /cluster (no files) -> {r.status_code} [{status}]")
    if r.status_code != 400:
        failures.append("cluster-no-files should be 400")

    files = []
    p = pathlib.Path(SAMPLE_DIR)
    if not p.exists():
        print(f"  [skip] sample dir {p} missing")
    else:
        for img in sorted(p.glob("*.png"))[:6]:
            files.append(("files", (img.name, img.read_bytes(), "image/png")))
        r = httpx.post(f"{BASE}/cluster", files=files, timeout=180.0)
        print(f"  /cluster with {len(files)} files -> {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            print(f"    n_images={body['n_images']} "
                  f"n_clusters={body['n_clusters']} "
                  f"silhouette={body['silhouette']:.3f}")
        elif r.status_code == 422:
            print("    [expected] NO_FACE (synthetic placeholder images)")
        else:
            failures.append(f"cluster-with-files unexpected {r.status_code}: {r.text}")

    if failures:
        print("\n[smoke] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[smoke] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
