#!/usr/bin/env python3
"""Draw the "same reports, different delivery" chart for one model.

    .venv/bin/python scripts/05_delivery_figure.py --out results/figures/delivery_llama31_8b.png \
        --model-label "Llama 3.1 8B" --run test_llama31_8b --run test_staged_llama31_8b
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
spec = importlib.util.spec_from_file_location("analyze", REPO_ROOT / "scripts" / "03_analyze_results.py")
an = importlib.util.module_from_spec(spec)
sys.modules["analyze"] = an
spec.loader.exec_module(an)

from src.metrics import estimate_rate, submitted  # noqa: E402
from src.schemas import AUTHORIZED, TEST, UNAUTHORIZED  # noqa: E402

ROWS = (
    ("B_local_success", "One message, PASS reports"),
    ("S_staged", "Second message, PASS reports"),
    ("S_neutral", "Second message, neutral reports"),
    ("S_placebo", "Asked twice, no reports"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-label", default="")
    args = parser.parse_args()

    records = []
    for name in args.run:
        path = REPO_ROOT / "data" / "runs" / name / "records.jsonl"
        records += [json.loads(line) for line in path.read_text().splitlines() if line]

    fig, ax = an._new_figure(len(ROWS) * 2, width=8.0)
    n = len(ROWS)
    for i, (condition, label) in enumerate(ROWS):
        base = n - 1 - i
        for state, colour, dy in ((AUTHORIZED, an.ORANGE, 0.17), (UNAUTHORIZED, an.BLUE, -0.17)):
            subset = an.select(records, TEST, condition, state)
            if not subset:
                continue
            est = estimate_rate(subset, submitted)
            y = base + dy
            low, high = est.ci_low, est.ci_high
            if high > low:
                ax.plot([low, high], [y, y], color=colour, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter(est.rate, y, s=64, color=colour, edgecolor=an.SURFACE, linewidth=1.5, zorder=3)
            ax.text(min(max(high, est.rate) + 0.02, 1.02) if est.rate < 0.9 else 1.04, y,
                    f"{est.rate * 100:.0f}%", va="center", ha="left", color=an.INK, fontsize=9.5)
    ax.set_yticks(range(n), [label for _, label in reversed(ROWS)], color=an.INK)
    ax.set_xlim(-0.03, 1.2)
    ticks = np.arange(0, 1.01, 0.25)
    ax.set_xticks(ticks, [f"{int(t * 100)}%" for t in ticks])
    ax.set_xlabel("Requests submitted", color=an.INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("How the reports reached Agent D", color=an.INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(-0.6, n - 0.4)
    handles = [
        Line2D([], [], color=an.ORANGE, marker="o", linewidth=2, label="Authorized cases"),
        Line2D([], [], color=an.BLUE, marker="o", linewidth=2, label="Unauthorized cases"),
    ]
    title = "Same reports, different delivery"
    subtitle = f"{args.model_label + ': ' if args.model_label else ''}submission rate on held-out pairs. Rows 2 and 3 contain the same reports as row 1."
    an._save(fig, args.out, title, subtitle, handles)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
