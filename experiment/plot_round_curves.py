"""Plot R1–R3 figures: controllability (P@K) as bars; cost/latency as lines.

Also writes a single vertically stacked combined figure for the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROUNDS = ["R1", "R2", "R3"]
LABEL_LEX = "LexCaR"
LABEL_BASELINE = "traditional agentic retrieval"
COLOR_LEX = "#7dd3fc"  # powder blue (粉蓝)
COLOR_MCP = "#e9a8f0"  # powder pink-purple (粉紫)

# Larger + bold text for paper figures
FS_TITLE = 15
FS_LABEL = 13
FS_TICK = 12
FS_LEGEND = 11
FS_ANNOT = 11
FONT_WEIGHT = "bold"


def load_results(exp_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(exp_dir.glob("result_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No result_*.json under {exp_dir}")
    return rows


def series_controllability(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Mean P@K vs each-round Target Set (controllability). R1 = Stage2 before first shift."""
    lex_q: List[List[float]] = []
    mcp_q: List[List[float]] = []
    for r in results:
        shifts = sorted(r.get("shifts") or [], key=lambda s: s["round_id"])
        if len(shifts) < 2:
            continue
        r1 = float(shifts[0]["before_metrics"].get("P@K") or 0.0)
        lex_vals = [r1] + [float(s["lexcar_metrics"].get("P@K") or 0.0) for s in shifts]
        mcp_vals = [r1] + [float(s["mcp_metrics"].get("P@K") or 0.0) for s in shifts]
        lex_q.append((lex_vals + [lex_vals[-1]] * 3)[:3])
        mcp_q.append((mcp_vals + [mcp_vals[-1]] * 3)[:3])
    n = len(lex_q) or 1
    return {
        LABEL_LEX: [sum(row[i] for row in lex_q) / n for i in range(3)],
        LABEL_BASELINE: [sum(row[i] for row in mcp_q) / n for i in range(3)],
    }


def series_cumulative_cost(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    lex_q: List[List[float]] = []
    mcp_q: List[List[float]] = []
    for r in results:
        shifts = sorted(r.get("shifts") or [], key=lambda s: s["round_id"])
        r1 = float(r["round1"].get("cost_cny") or 0.0)
        lex_c = [r1]
        mcp_c = [r1]
        running_l, running_m = r1, r1
        for s in shifts:
            running_l += float(s["lexcar"].get("cost_cny") or 0.0)
            running_m += float(s["mcp"].get("cost_cny") or 0.0)
            lex_c.append(running_l)
            mcp_c.append(running_m)
        lex_q.append((lex_c + [lex_c[-1]] * 3)[:3])
        mcp_q.append((mcp_c + [mcp_c[-1]] * 3)[:3])
    n = len(lex_q) or 1
    return {
        LABEL_LEX: [sum(row[i] for row in lex_q) / n for i in range(3)],
        LABEL_BASELINE: [sum(row[i] for row in mcp_q) / n for i in range(3)],
    }


def series_cumulative_latency(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    lex_q: List[List[float]] = []
    mcp_q: List[List[float]] = []
    for r in results:
        shifts = sorted(r.get("shifts") or [], key=lambda s: s["round_id"])
        r1 = float(r["round1"].get("seconds") or 0.0)
        lex_c = [r1]
        mcp_c = [r1]
        running_l, running_m = r1, r1
        for s in shifts:
            running_l += float(s["lexcar"].get("seconds") or 0.0)
            running_m += float(s["mcp"].get("seconds") or 0.0)
            lex_c.append(running_l)
            mcp_c.append(running_m)
        lex_q.append((lex_c + [lex_c[-1]] * 3)[:3])
        mcp_q.append((mcp_c + [mcp_c[-1]] * 3)[:3])
    n = len(lex_q) or 1
    return {
        LABEL_LEX: [sum(row[i] for row in lex_q) / n for i in range(3)],
        LABEL_BASELINE: [sum(row[i] for row in mcp_q) / n for i in range(3)],
    }


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(FONT_WEIGHT)


def _legend(ax: plt.Axes) -> None:
    leg = ax.legend(frameon=False, loc="best", fontsize=FS_LEGEND)
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontweight(FONT_WEIGHT)


def draw_bars(
    ax: plt.Axes,
    series: Dict[str, List[float]],
    *,
    ylabel: str,
    title: str,
    yfmt: str = ".2f",
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    x = np.arange(len(ROUNDS))
    width = 0.36
    lex = series[LABEL_LEX]
    mcp = series[LABEL_BASELINE]

    bars_lex = ax.bar(
        x - width / 2,
        lex,
        width,
        label=LABEL_LEX,
        color=COLOR_LEX,
        edgecolor="white",
        linewidth=0.6,
    )
    bars_mcp = ax.bar(
        x + width / 2,
        mcp,
        width,
        label=LABEL_BASELINE,
        color=COLOR_MCP,
        edgecolor="white",
        linewidth=0.6,
    )

    for bars in (bars_lex, bars_mcp):
        for rect in bars:
            h = float(rect.get_height())
            ax.annotate(
                format(h, yfmt),
                (rect.get_x() + rect.get_width() / 2, h),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=FS_ANNOT,
                fontweight=FONT_WEIGHT,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ROUNDS)
    ymax = max(max(lex), max(mcp), 1e-9)
    ax.set_ylim(0, ymax * 1.22)
    style_axes(ax)
    if show_xlabel:
        ax.set_xlabel("Round", fontsize=FS_LABEL, fontweight=FONT_WEIGHT)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight=FONT_WEIGHT)
    ax.set_title(title, fontsize=FS_TITLE, fontweight=FONT_WEIGHT, pad=10)
    if show_legend:
        _legend(ax)


def draw_lines(
    ax: plt.Axes,
    series: Dict[str, List[float]],
    *,
    ylabel: str,
    title: str,
    yfmt: str = ".2f",
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    for name, vals, color, marker in (
        (LABEL_LEX, series[LABEL_LEX], COLOR_LEX, "o"),
        (LABEL_BASELINE, series[LABEL_BASELINE], COLOR_MCP, "s"),
    ):
        ax.plot(
            range(3),
            vals,
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=8,
            label=name,
        )
        for i, v in enumerate(vals):
            ax.annotate(
                format(v, yfmt),
                (i, v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=FS_ANNOT,
                fontweight=FONT_WEIGHT,
                color="#333333",
            )
    ax.set_xticks(range(3))
    ax.set_xticklabels(ROUNDS)
    style_axes(ax)
    if show_xlabel:
        ax.set_xlabel("Round", fontsize=FS_LABEL, fontweight=FONT_WEIGHT)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight=FONT_WEIGHT)
    ax.set_title(title, fontsize=FS_TITLE, fontweight=FONT_WEIGHT, pad=10)
    if show_legend:
        _legend(ax)


def save_fig(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_bars(
    series: Dict[str, List[float]],
    *,
    ylabel: str,
    title: str,
    out: Path,
    yfmt: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=160)
    draw_bars(ax, series, ylabel=ylabel, title=title, yfmt=yfmt or ".2f")
    save_fig(fig, out)


def plot_lines(
    series: Dict[str, List[float]],
    *,
    ylabel: str,
    title: str,
    out: Path,
    yfmt: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=160)
    draw_lines(ax, series, ylabel=ylabel, title=title, yfmt=yfmt or ".2f")
    save_fig(fig, out)


def plot_combined(
    acc: Dict[str, List[float]],
    cost: Dict[str, List[float]],
    lat: Dict[str, List[float]],
    out: Path,
) -> None:
    """One tall figure: controllability (bars) → cost (lines) → latency (lines)."""
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 11.5), dpi=160, sharex=False)

    draw_bars(
        axes[0],
        acc,
        ylabel="P@K",
        title="(a) Controllability by round (mean P@K)",
        yfmt=".2f",
        show_xlabel=False,
        show_legend=True,
    )
    draw_lines(
        axes[1],
        cost,
        ylabel="Cumulative cost (CNY)",
        title="(b) Cumulative cost across rounds",
        yfmt=".2f",
        show_xlabel=False,
        show_legend=True,
    )
    draw_lines(
        axes[2],
        lat,
        ylabel="Cumulative latency (sec)",
        title="(c) Cumulative latency across rounds",
        yfmt=".1f",
        show_xlabel=True,
        show_legend=True,
    )

    fig.subplots_adjust(hspace=0.35)
    save_fig(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help="Experiment result folder (default: latest exp_eff_ctrl_*)",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write the stacked combined figure",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "results"
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        cands = sorted(root.glob("exp_eff_ctrl_*"))
        if not cands:
            raise SystemExit(f"No exp_eff_ctrl_* under {root}")
        exp_dir = cands[-1]

    results = load_results(exp_dir)
    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = series_controllability(results)
    cost = series_cumulative_cost(results)
    lat = series_cumulative_latency(results)

    if not args.combined_only:
        plot_bars(
            acc,
            ylabel="P@K",
            title="Controllability by round (mean P@K)",
            out=out_dir / "fig_controllability_rounds.png",
            yfmt=".2f",
        )
        plot_lines(
            cost,
            ylabel="Cumulative cost (CNY)",
            title="Cumulative cost across rounds",
            out=out_dir / "fig_cost_cumulative.png",
            yfmt=".2f",
        )
        plot_lines(
            lat,
            ylabel="Cumulative latency (sec)",
            title="Cumulative latency across rounds",
            out=out_dir / "fig_latency_cumulative.png",
            yfmt=".1f",
        )

    plot_combined(acc, cost, lat, out=out_dir / "fig_combined.png")

    meta = {
        "exp_dir": str(exp_dir),
        "n": len(results),
        "controllability": acc,
        "cumulative_cost_cny": cost,
        "cumulative_latency_sec": lat,
        "figures": {
            "combined": "fig_combined.png",
            "controllability": "fig_controllability_rounds.png",
            "cost": "fig_cost_cumulative.png",
            "latency": "fig_latency_cumulative.png",
        },
        "note": (
            "Controllability = mean P@K vs the round's focus Target Set "
            "(anchored on the Round-1 pool). R1 uses Stage2 ranking vs the "
            "Round-2 Target Set (shared start). Cost/latency cumulatives = "
            "Round1 shared + each system's Round2..k costs."
        ),
    }
    (out_dir / "fig_data.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
