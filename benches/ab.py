#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run the criterion benchmarks in two git refs and export the dated record.

The criterion suites (`benches/parse.rs`, `benches/serialize.rs`) are the
single measurement definition; this script only orchestrates them: create a
detached worktree per ref under a temp dir, run `cargo bench` there filtered
to this crate's benchmarks, diff the two `target/criterion` trees, and write
`benches/results/{date}_{slug}_ab.json`.

Sides are measured sequentially (base first) with each side's own target
dir; the suite's warmup/measurement windows stand in for drift-cancelling
alternation. Defaults are a quick configuration (20 samples, 1 s warmup,
2 s measurement, ~2 minutes per side including the build); pass larger
values for a firmer record. Manual `cargo bench` runs keep criterion's
defaults.

The suite itself skips `json5` on canada.json (quadratic on long float
arrays, one-shot measured via examples/one_shot.rs instead); the A/B only
runs this crate's benchmarks anyway.

Metadata is limited to what is collectable automatically (date, cpu, rustc,
os, commits, corpus sizes); the human-written fields of older records (kind,
purpose, method, note) are omitted.

Usage: uv run benches/ab.py <base_ref> [head_ref] [--slug S]
       [--sample-size N] [--warm-up-time S] [--measurement-time S]

`head_ref` defaults to HEAD; both sides are measured from committed state,
so commit before running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benches" / "results"
DATA = REPO / "benches" / "data"
AB_TARGETS = REPO / "target" / "ab"

CORPORA = ["twitter.json", "citm_catalog.json", "canada.json", "json5.json5"]
OP_ORDER = ["parse", "parse/slice", "parse/typed", "serialize", "serialize/typed"]
OURS = "simd-json5"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def environment() -> dict:
    cpu = platform.processor() or "unknown"
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    rustc = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        # local date: records are named by the user-facing day, unlike report.py's UTC
        "date": dt.datetime.now().astimezone().strftime("%Y-%m-%d"),
        "cpu": cpu,
        "rustc": rustc,
        "os": f"{platform.system()} {platform.release()}",
    }


def bench_side(side: str, ref: str, tmp: Path, args: argparse.Namespace) -> None:
    """Worktree at `ref`, criterion run there into target/ab/{side}."""
    print(f"benching {side} ({ref})...", flush=True)
    wt = tmp / side
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(wt), ref],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    target = AB_TARGETS / side
    # wipe fully: a cached bench binary embeds its (long-gone) worktree path
    # via CARGO_MANIFEST_DIR, so cross-run incremental reuse is unsound
    shutil.rmtree(target, ignore_errors=True)
    env = os.environ | {"CARGO_TARGET_DIR": str(target)}
    subprocess.run(
        [
            "cargo", "bench", "--bench", "parse", "--bench", "serialize", "--",
            "--sample-size", str(args.sample_size),
            "--warm-up-time", str(args.warm_up_time),
            "--measurement-time", str(args.measurement_time),
            OURS,
        ],
        cwd=wt,
        env=env,
        check=True,
    )


def collect_side(side: str) -> dict[tuple[str, str], dict]:
    """Median/std-dev/sample count per (group, corpus) from criterion output.

    Criterion flattens group/function names into directory names with `_`
    (e.g. group `parse` + corpus `twitter.json` -> `parse_twitter.json`,
    function `twitter.json/simd-json5` -> `twitter.json_simd-json5`).
    """
    acc: dict[tuple[str, str], dict] = {}
    for est in sorted((AB_TARGETS / side / "criterion").glob("*/*/new/estimates.json")):
        group_dir, fn_dir = est.relative_to(AB_TARGETS / side / "criterion").parts[:2]
        if fn_dir == OURS:
            op, corpus = group_dir.split("_", 1)
        elif fn_dir.endswith(f"_{OURS}"):
            op = group_dir.replace("_", "/", 1)
            corpus = fn_dir.removesuffix(f"_{OURS}")
        else:
            continue
        estimates = json.loads(est.read_text())
        sample = json.loads((est.parent / "sample.json").read_text())
        times = sample["times"] if isinstance(sample, dict) else sample
        iters = sample["iters"] if isinstance(sample, dict) else [1.0] * len(times)
        acc[(op, corpus)] = {
            "median_ns": estimates["median"]["point_estimate"],
            # times are per-sample totals; per-iteration time is times/iters
            "min_ns": min(t / n for t, n in zip(times, iters)),
            "samples": len(times),
        }
    return acc


def median(vals: list[int]) -> int:
    return sorted(vals)[len(vals) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base", help="git ref to measure as the baseline")
    parser.add_argument(
        "head", nargs="?", default="HEAD", help="git ref to measure as the head (default HEAD)"
    )
    parser.add_argument("--slug", help="record filename slug: {date}_{slug}_ab.json")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--warm-up-time", type=float, default=1.0)
    parser.add_argument("--measurement-time", type=float, default=2.0)
    args = parser.parse_args()

    if git("status", "--porcelain"):
        print(
            "note: working tree is dirty; both sides are measured from committed state",
            file=sys.stderr,
        )

    tmp = Path(tempfile.mkdtemp(prefix="simd-json5-ab-"))
    try:
        for side, ref in (("base", args.base), ("head", args.head)):
            bench_side(side, ref, tmp, args)

        sides = {side: collect_side(side) for side in ("base", "head")}
        if not sides["base"]:
            sys.exit(f"no {OURS} estimates under {AB_TARGETS / 'base' / 'criterion'}")
        if set(sides["base"]) != set(sides["head"]):
            sys.exit("base and head produced different (op, corpus) sets; bench suites likely differ")
        unknown = {op for op, _ in sides["base"] if op not in OP_ORDER}
        if unknown:
            sys.exit(f"criterion reports groups outside {OP_ORDER}: {sorted(unknown)}")
        keys = sorted(
            sides["base"], key=lambda k: (OP_ORDER.index(k[0]), CORPORA.index(k[1]))
        )

        sizes = {p.name: p.stat().st_size for p in DATA.iterdir() if p.suffix in (".json", ".json5")}
        results = []
        for key in keys:
            base, head = sides["base"][key], sides["head"][key]
            base_med, head_med = round(base["median_ns"]), round(head["median_ns"])
            op, corpus = key
            results.append(
                {
                    "op": op,
                    "corpus": corpus,
                    "base_median_ns": base_med,
                    "head_median_ns": head_med,
                    "median_delta_pct": round((head_med - base_med) / base_med * 100, 2),
                    "base_min_ns": round(base["min_ns"]),
                    "head_min_ns": round(head["min_ns"]),
                    "base_median_mb_s": round(sizes[corpus] * 1e3 / base_med, 1),
                    "head_median_mb_s": round(sizes[corpus] * 1e3 / head_med, 1),
                    "samples": head["samples"],
                }
            )

        record = environment() | {
            "base_commit": git("rev-parse", args.base),
            "head_commit": git("rev-parse", args.head),
            "corpus_bytes": sizes,
            "results": results,
        }

        RESULTS.mkdir(parents=True, exist_ok=True)
        slug = f"_{args.slug}" if args.slug else ""
        out = RESULTS / f"{record['date']}{slug}_ab.json"
        if out.exists():
            sys.exit(f"{out} already exists; pass --slug to pick another name")
        out.write_text(json.dumps(record, indent=2) + "\n")
    finally:
        shutil.rmtree(tmp)
        subprocess.run(
            ["git", "worktree", "prune"], cwd=REPO, capture_output=True, check=False
        )

    for r in results:
        print(
            f"{r['op']:16} {r['corpus']:18} "
            f"{r['base_median_ns']:>10} -> {r['head_median_ns']:>10} ns  "
            f"{r['median_delta_pct']:+.2f}%"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
