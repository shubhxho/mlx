# Copyright © 2026 Apple Inc.
"""Turn cpu_ops_bench.py JSON results into a markdown table and a chart.

Usage:
    python benchmarks/python/cpu_ops_report.py BEFORE.json AFTER.json \
        [--one-thread AFTER_1T.json] [--chart out.png]
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--one-thread", default="")
    p.add_argument("--chart", default="")
    args = p.parse_args()
    before = json.load(open(args.before))
    after = json.load(open(args.after))
    one = json.load(open(args.one_thread)) if args.one_thread else None

    hdr = "| Op | Before (ms) | After (ms) | Speedup |"
    sep = "|---|---:|---:|---:|"
    if one:
        hdr += " After, 1 thread (ms) | Speedup, 1 thread |"
        sep += "---:|---:|"
    print(hdr)
    print(sep)
    names, speedups = [], []
    for name, b in before.items():
        if name not in after:
            continue
        a = after[name]
        s = b["median_ms"] / a["median_ms"]
        row = (
            f"| {name} | {b['median_ms']:.3f} [{b['min_ms']:.3f}..{b['max_ms']:.3f}]"
            f" | {a['median_ms']:.3f} [{a['min_ms']:.3f}..{a['max_ms']:.3f}]"
            f" | {s:.2f}x |"
        )
        if one:
            o = one[name]
            row += f" {o['median_ms']:.3f} | {b['median_ms'] / o['median_ms']:.2f}x |"
        print(row)
        names.append(name)
        speedups.append(s)

    if args.chart:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = sorted(range(len(names)), key=lambda i: speedups[i])
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(names) + 1.2))
        ys = range(len(order))
        vals = [speedups[i] for i in order]
        colors = ["#2f7ed8" if v >= 1.1 else "#999999" for v in vals]
        ax.barh(list(ys), vals, color=colors)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([names[i] for i in order], fontsize=8)
        ax.set_xscale("log")
        ax.axvline(1.0, color="black", linewidth=0.8)
        for y, v in zip(ys, vals):
            ax.text(v * 1.05, y, f"{v:.1f}x", va="center", fontsize=7)
        ax.set_xlabel("Speedup vs upstream (median time, log scale)")
        ax.set_title("MLX CPU backend, perf/cpu vs upstream (8 vCPU Xeon, Linux)")
        ax.set_xlim(0.5, max(vals) * 2)
        fig.tight_layout()
        fig.savefig(args.chart, dpi=150)


if __name__ == "__main__":
    main()
