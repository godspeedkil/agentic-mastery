"""Static charts for the report. Each function answers one question and returns a Figure.

Static on purpose: GitHub renders notebook images and SVG files, not interactive charts.
Colors come from a validated palette: blue and orange separate providers for color-blind
readers too, and every mark is also labeled, so identity never rests on color alone.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from p01.models import MODELS

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
PROVIDER_COLORS = {"anthropic": "#2a78d6", "openai": "#eb6834"}
RUN_COLORS = ["#86b6ef", "#3987e5", "#184f95"]
"""Ordinal blue steps for successive runs, light (first) to dark (latest)."""
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PROVIDER = {model.id: model.provider for model in MODELS}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "font.size": 10,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "p01",
        }
    )


def save(fig: Figure, path: Path) -> None:
    """SVG with no timestamp and fixed internal IDs, so unchanged charts produce no diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})


def _percent_axis(axis) -> None:
    axis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def accuracy_vs_cost(summary: pd.DataFrame) -> Figure:
    """Which model gives the most accuracy per dollar? Bars = 95% interval over cases."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, row in summary.iterrows():
        color = PROVIDER_COLORS[PROVIDER[model]]
        ax.errorbar(
            row.cost_per_1k_usd,
            row.exact_rate,
            yerr=[[row.exact_rate - row.ci_low], [row.ci_high - row.exact_rate]],
            fmt="o",
            color=color,
            markersize=8,
            elinewidth=2,
            capsize=0,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=3,
        )
    for model, above in _label_sides(summary).items():
        row = summary.loc[model]
        ax.annotate(
            model,
            (row.cost_per_1k_usd, row.exact_rate),
            xytext=(0, 10 if above else -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=TEXT,
        )
    ax.set_xscale("log")
    ax.set_xlim(summary.cost_per_1k_usd.min() / 2, summary.cost_per_1k_usd.max() * 2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_ylim(summary.ci_low.min() - 0.03, 1.04)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    _percent_axis(ax.yaxis)
    ax.set_xlabel("Cost per 1,000 invoices (USD, log scale)")
    ax.set_ylabel("All six fields correct")
    ax.set_title("Accuracy vs cost per model")
    _provider_legend(ax)
    return fig


def _label_sides(summary: pd.DataFrame) -> dict[str, bool]:
    """Label above each point, except below when it would collide with a close neighbor."""
    sides: dict[str, bool] = {}
    previous = None
    for model, row in summary.sort_values("cost_per_1k_usd").iterrows():
        above = True
        if previous is not None:
            near_x = abs(np.log10(row.cost_per_1k_usd / previous.cost_per_1k_usd)) < 0.45
            near_y = abs(row.exact_rate - previous.exact_rate) < 0.03
            if near_x and near_y:
                above = not sides[previous.name]
        sides[model] = above
        previous = row
    return sides


def spec_iterations(runs: Sequence[tuple[str, pd.DataFrame]]) -> Figure:
    """How did each fix to the specification change each model's accuracy?"""
    models = [m for m in PROVIDER if all(m in summary.index for _, summary in runs)]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1.6))
    positions = range(len(models))[::-1]
    for y, model in zip(positions, models, strict=True):
        rates = [summary.loc[model, "exact_rate"] for _, summary in runs]
        ax.plot([min(rates), max(rates)], [y, y], color=GRID, linewidth=2, zorder=1)
    for (label, summary), color in zip(runs, RUN_COLORS, strict=False):
        rates = [summary.loc[model, "exact_rate"] for model in models]
        ax.scatter(rates, positions, s=70, color=color, edgecolor=SURFACE, linewidth=2, label=label)
    ax.set_yticks(list(positions), models)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    _percent_axis(ax.xaxis)
    ax.set_xlabel("All six fields correct")
    ax.set_title("Accuracy after each fix to the specification", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(runs), fontsize=9)
    return fig


def field_errors(rates: pd.DataFrame) -> Figure:
    """Which fields does each model get wrong? Darker = more often wrong; blank = never."""
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL).with_extremes(
        bad=SURFACE
    )
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rates) + 1.4))
    vmax = max(rates.to_numpy().max(), 0.05)
    values = np.ma.masked_equal(rates.to_numpy(), 0)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for (i, j), value in pd.DataFrame(rates.to_numpy()).stack().items():
        label = "–" if value == 0 else f"{value:.0%}"
        color = SURFACE if value > 0.6 * vmax else TEXT
        ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(rates.columns)), rates.columns)
    ax.set_yticks(range(len(rates.index)), rates.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Share of predictions with each field wrong")
    return fig


def latency(summary: pd.DataFrame) -> Figure:
    """How long does one extraction take? Dot = median, bar end = 95th percentile."""
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(summary) + 1.4))
    positions = range(len(summary))[::-1]
    for y, (model, row) in zip(positions, summary.iterrows(), strict=True):
        color = PROVIDER_COLORS[PROVIDER[model]]
        ax.plot([row.latency_p50_s, row.latency_p95_s], [y, y], color=color, linewidth=2)
        ax.plot(row.latency_p95_s, y, marker="|", markersize=10, color=color, mew=2)
        ax.scatter(row.latency_p50_s, y, s=70, color=color, edgecolor=SURFACE, linewidth=2)
        ax.annotate(
            f"{row.latency_p50_s:.1f}s / {row.latency_p95_s:.1f}s",
            (row.latency_p95_s, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=MUTED,
        )
    ax.set_yticks(list(positions), summary.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, summary.latency_p95_s.max() * 1.3)
    ax.set_xlabel("Seconds per extraction (median and 95th percentile)")
    ax.set_title("Latency per model")
    return fig


def _provider_legend(ax) -> None:
    for provider, color in PROVIDER_COLORS.items():
        ax.scatter([], [], s=60, color=color, label=provider)
    ax.legend(loc="lower right", fontsize=9)
