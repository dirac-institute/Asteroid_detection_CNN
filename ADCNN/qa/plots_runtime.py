#!/usr/bin/env python3
"""Render the ``run_night`` runtime report as a per-stage wall-time bar chart (acceptance E).

Reads a ``runtime_report.json`` (written by :mod:`ADCNN.pipelines.run_night`) and emits a figure
showing wall time per stage plus the derived per-visit / per-detector-pass / per-night summary.

Usage:
    PYTHONPATH=. python -m ADCNN.qa.plots_runtime <runtime_report.json> [out.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_runtime(report, out_png):
    """`report` is the dict from runtime_report.json (or a path to it)."""
    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text())
    stages = report.get("stages", [])
    names = [s["stage"] for s in stages]
    secs = [s["seconds"] for s in stages]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.barh(range(len(names)), secs, color="#1f77b4")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("wall time [s]")
    for i, v in enumerate(secs):
        ax.text(v, i, f" {v:.1f}s", va="center", fontsize=8)
    summ = (f"night={report.get('night')}  tracts={report.get('tracts')}  "
            f"visits={report.get('n_visits')}  detector-passes={report.get('n_detector_passes')}\n"
            f"per-visit {report.get('per_visit_seconds')}s · "
            f"per-detector-pass {report.get('per_detector_pass_seconds')}s · "
            f"night {report.get('per_night_seconds')}s"
            + ("   [DRY-RUN: stage times ~0]" if report.get("dry_run") else ""))
    ax.set_title("run_night wall time per stage\n" + summ, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=140)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="runtime_report.json")
    ap.add_argument("out", nargs="?", default=None, help="output PNG (default: alongside the report)")
    a = ap.parse_args()
    out = a.out or str(Path(a.report).with_name("runtime.png"))
    plot_runtime(a.report, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
