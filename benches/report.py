#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Collect criterion results into benches/results/ and render SVG charts.

Reads target/criterion/**/new/estimates.json written by
`cargo bench --bench parse --bench serialize`, folds in the single-sample
entries from benches/results/one_shot.jsonl, and writes a dated JSON record
plus parse.svg / serialize.svg throughput charts for the README.

Usage: uv run benches/report.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

REPO = Path(__file__).resolve().parents[1]
CRITERION = REPO / "target" / "criterion"
DATA = REPO / "benches" / "data"
RESULTS = REPO / "benches" / "results"
ONE_SHOT = RESULTS / "one_shot.jsonl"

CORPORA = ["twitter.json", "citm_catalog.json", "canada.json", "json5.json5"]
IMPLS = ["simd-json5", "json5", "serde-json5", "serde_json", "simd-json", "sonic-rs"]
OURS = "simd-json5"
JSON5_IMPLS = {"json5", "serde-json5"}

COLOR_OURS = "#d1495b"
COLOR_JSON5 = "#3d6fb4"
COLOR_JSON = "#8a8f98"


def color(impl: str) -> str:
    if impl == OURS:
        return COLOR_OURS
    return COLOR_JSON5 if impl in JSON5_IMPLS else COLOR_JSON


def collect_criterion() -> list[dict]:
    """Latest estimates for every comparison benchmark.

    Criterion flattens group/function names into directory names with `_`
    (e.g. group `parse` + corpus `twitter.json` -> `parse_twitter.json`).
    """
    out = []
    for est in sorted(CRITERION.glob("*/*/new/estimates.json")):
        group_dir, impl = est.relative_to(CRITERION).parts[:2]
        bench, _, corpus = group_dir.partition("_")
        if bench not in ("parse", "serialize") or corpus not in CORPORA or impl not in IMPLS:
            continue
        estimates = json.loads(est.read_text())
        sample = json.loads((est.parent / "sample.json").read_text())
        times = sample["times"] if isinstance(sample, dict) else sample
        out.append({
            "benchmark": bench,
            "corpus": corpus,
            "impl": impl,
            "mean_ns": estimates["mean"]["point_estimate"],
            "std_dev_ns": estimates["std_dev"]["point_estimate"],
            "samples": len(times),
        })
    return out


def collect_one_shot() -> list[dict]:
    if not ONE_SHOT.is_file():
        return []
    return [json.loads(line) for line in ONE_SHOT.read_text().splitlines() if line.strip()]


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
        "date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "cpu": cpu,
        "rustc": rustc,
        "os": f"{platform.system()} {platform.release()}",
    }


def fmt_size(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def fmt_mb(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def plot(record: dict, bench: str, path: Path) -> None:
    cells = [
        (
            corpus,
            sorted(
                (
                    r
                    for r in record["results"]
                    if r["benchmark"] == bench and r["corpus"] == corpus
                ),
                key=lambda r: r["mb_s"],
                reverse=True,
            ),
        )
        for corpus in CORPORA
    ]
    cells = [(corpus, rows) for corpus, rows in cells if rows]
    if not cells:
        return

    nrows = math.ceil(len(cells) / 2)
    height = 3.2 * nrows + 0.9
    fig, axes = plt.subplots(nrows, 2, figsize=(10, height), squeeze=False)
    fig.subplots_adjust(
        left=0.09, right=0.985, top=1 - 0.55 / height, bottom=0.85 / height,
        hspace=0.55, wspace=0.75,
    )

    for ax, (corpus, rows) in zip(axes.flat, cells):
        names = [r["impl"] + ("*" if r.get("one_shot") else "") for r in rows]
        values = [r["mb_s"] for r in rows]
        colors = [color(r["impl"]) for r in rows]
        y = range(len(rows))

        ax.barh(y, values, color=colors, height=0.62)
        ax.set_yticks(y, names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xscale("log")
        # rows are sorted fastest-first; leave ~0.8 decades on the right for
        # the value labels and half a decade on the left for breathing room
        ax.set_xlim(values[-1] / 3, values[0] * 8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", which="major", alpha=0.35)
        ax.grid(axis="x", which="minor", alpha=0.12)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        for yi, v in zip(y, values):
            ax.text(v * 1.25, yi, fmt_mb(v), va="center", fontsize=8, color="#333")
        ax.set_title(f"{corpus} ({fmt_size(record['corpus_bytes'][corpus])})", fontsize=11)

    for ax in axes.flat[len(cells):]:
        ax.axis("off")

    handles = [
        Patch(color=COLOR_OURS, label="simd-json5 (this crate)"),
        Patch(color=COLOR_JSON5, label="other JSON5 implementations"),
        Patch(color=COLOR_JSON, label="JSON implementations"),
    ]
    fig.suptitle(f"{bench} throughput (MB/s, log scale, higher is better)", fontsize=13)
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.055),
        ncol=3, fontsize=9, frameon=False,
    )
    fig.text(
        0.5, 0.004,
        f"* single sample, measured outside criterion   |   {record['cpu']}   |   "
        f"{record['rustc'].split(' (')[0]}   |   {record['date']}",
        ha="center", fontsize=7.5, color="#555",
    )
    fig.savefig(path)
    plt.close(fig)


def summarize(record: dict) -> None:
    for bench in ("parse", "serialize"):
        for corpus in CORPORA:
            rows = {
                r["impl"]: r["mb_s"]
                for r in record["results"]
                if r["benchmark"] == bench and r["corpus"] == corpus
            }
            if not rows:
                continue
            ours = rows[OURS]
            cells = "  ".join(f"{k}={v:8.2f}" for k, v in rows.items())
            print(f"{bench:9} {corpus:18} {cells}")
            ratios = "  ".join(f"{k}={ours / v:8.1f}x" for k, v in rows.items() if k != OURS)
            print(f"{'':28} vs {OURS}: {ratios}")


def main() -> None:
    if not CRITERION.is_dir():
        sys.exit("target/criterion not found; run `cargo bench --bench parse --bench serialize` first")
    results = collect_criterion() + collect_one_shot()
    if not results:
        sys.exit("no comparison estimates found; run the parse and serialize benches first")

    sizes = {p.name: p.stat().st_size for p in DATA.iterdir() if p.suffix in (".json", ".json5")}
    record = environment() | {"corpus_bytes": sizes, "results": results}
    for r in results:
        r["mb_s"] = sizes[r["corpus"]] * 1e3 / r["mean_ns"]
    order = {
        (bench, corpus): i
        for i, (bench, corpus) in enumerate(
            (bench, corpus) for bench in ("parse", "serialize") for corpus in CORPORA
        )
    }
    record["results"] = sorted(
        results, key=lambda r: (order[(r["benchmark"], r["corpus"])], -r["mb_s"])
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{record['date']}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    for bench in ("parse", "serialize"):
        if any(r["benchmark"] == bench for r in results):
            plot(record, bench, RESULTS / f"{bench}.svg")

    summarize(record)
    print(f"\nwrote {out} plus parse.svg / serialize.svg in {RESULTS}")


if __name__ == "__main__":
    main()
