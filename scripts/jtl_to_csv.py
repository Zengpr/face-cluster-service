"""Read a JMeter .jtl file and emit a compact summary block."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jtl", help="path to a *-results jtl file")
    args = p.parse_args()

    df = pd.read_csv(args.jtl)
    if "elapsed" not in df.columns:
        raise SystemExit(f"unexpected jtl header: {df.columns.tolist()}")

    df["elapsed_ms"] = df["elapsed"]

    summary = df.groupby([df.get("label", "all")]).agg(
        n=("elapsed_ms", "size"),
        ok_pct=("success", "mean"),
        p50=("elapsed_ms", lambda s: float(s.quantile(0.50))),
        p95=("elapsed_ms", lambda s: float(s.quantile(0.95))),
        p99=("elapsed_ms", lambda s: float(s.quantile(0.99))),
        avg=("elapsed_ms", "mean"),
        max=("elapsed_ms", "max"),
    )
    summary["ok_pct"] *= 100.0

    total_seconds = (df["timeStamp"].iloc[-1] - df["timeStamp"].iloc[0]) / 1000.0
    total_count = int(df.shape[0])
    if total_seconds > 0:
        print(f"throughput ≈ {total_count / total_seconds:.2f} req/s "
              f"({total_count} reqs / {total_seconds:.1f}s)")
    print(summary.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
