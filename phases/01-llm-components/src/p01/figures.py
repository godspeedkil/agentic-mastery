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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
from matplotlib.transforms import Bbox

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
    ax.set_xscale("log")
    ax.set_xlim(summary.cost_per_1k_usd.min() / 3, summary.cost_per_1k_usd.max() * 3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_ylim(summary.ci_low.min() - 0.03, 1.04)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    _percent_axis(ax.yaxis)
    ax.set_xlabel("Cost per 1,000 invoices (USD, log scale)")
    ax.set_ylabel("All six fields correct")
    ax.set_title("Accuracy vs cost per model")
    _provider_legend(ax)
    points = summary.sort_values("cost_per_1k_usd")[["cost_per_1k_usd", "exact_rate"]]
    _place_labels(ax, {str(model): (x, y) for model, (x, y) in points.iterrows()})
    return fig


LABEL_SPOTS = (
    ((8, 0), "left", "center"),
    ((-8, 0), "right", "center"),
    ((6, 6), "left", "bottom"),
    ((-6, 6), "right", "bottom"),
    ((6, -6), "left", "top"),
    ((-6, -6), "right", "top"),
    ((0, 9), "center", "bottom"),
    ((0, -9), "center", "top"),
)
"""Spots for a label, in points from its marker, in order of preference: beside the marker,
then diagonally off it (which clears its own vertical error bar), then straight above or
below."""


def _place_labels(ax, points: dict[str, tuple[float, float]]) -> None:
    """Label each point at the first spot whose text box hits no mark, label or legend.

    Call after scales, limits and the legend are set: screen positions depend on them.
    If every spot collides, the one with the fewest collisions wins; no label is dropped.
    """
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    boxes = [_extent(artist, renderer) for artist in [*ax.lines, *ax.collections]]
    taken = [box.padded(2) for box in boxes if np.isfinite(box.bounds).all()]
    if (legend := ax.get_legend()) is not None:
        taken.append(legend.get_window_extent(renderer))
    for name, xy in points.items():
        candidates = []
        for offset, ha, va in LABEL_SPOTS:
            label = ax.annotate(
                name,
                xy,
                xytext=offset,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=9,
                color=TEXT,
            )
            candidates.append((label, _widen(label.get_window_extent(renderer), ha)))
        chosen, box = min(candidates, key=lambda c: _collisions(c[1], frame, taken))
        for label, _ in candidates:
            if label is not chosen:
                label.remove()
        taken.append(box)


FONT_SLACK = 0.25
"""Extra width allowed per label. The SVG names fonts instead of embedding them, and the
viewer's fallback font (Verdana, Helvetica...) can be wider than the one measured here."""


def _widen(box: Bbox, ha: str) -> Bbox:
    """Grow a text box on the side facing away from its anchor, keeping the anchor fixed."""
    extra = box.width * FONT_SLACK
    grow = {"left": (0, extra), "right": (extra, 0)}.get(ha, (extra / 2, extra / 2))
    return Bbox.from_extents(box.x0 - grow[0], box.y0, box.x1 + grow[1], box.y1)


def _collisions(box: Bbox, frame: Bbox, taken: list[Bbox]) -> int:
    """How badly a label box fits: leaving the plot counts as one collision, like a hit."""
    outside = not (frame.contains(box.x0, box.y0) and frame.contains(box.x1, box.y1))
    return int(outside) + sum(box.overlaps(other) for other in taken)


def _extent(artist, renderer) -> Bbox:
    """Screen box of a mark. Line collections (error bars) report an empty window extent,
    so their box is built from their line segments instead."""
    if isinstance(artist, LineCollection) and artist.get_paths():
        transform = artist.get_transform()
        return Bbox.union([path.get_extents(transform) for path in artist.get_paths()])
    return artist.get_window_extent(renderer)


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
