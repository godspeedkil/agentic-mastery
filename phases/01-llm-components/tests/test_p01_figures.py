import matplotlib

matplotlib.use("Agg")  # draw without a display, as in CI

import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from p01 import figures

MODELS = ["claude-haiku-5-5", "gpt-6-luna"]
SUMMARY = pd.DataFrame(
    {
        "exact_rate": [1.0, 0.98],
        "ci_low": [0.886, 0.83],
        "ci_high": [1.0, 0.99],
        "cost_per_1k_usd": [0.30, 0.32],
        "latency_p50_s": [1.5, 1.1],
        "latency_p95_s": [2.3, 2.0],
    },
    index=pd.Index(MODELS, name="model"),
)
FIELD_ERRORS = pd.DataFrame(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]],
    index=MODELS,
    columns=["vendor", "invoice_number", "total", "currency", "due_date", "category"],
)


@pytest.mark.parametrize(
    "draw",
    [
        lambda: figures.accuracy_vs_cost(SUMMARY),
        lambda: figures.spec_iterations([("1", SUMMARY), ("2", SUMMARY)]),
        lambda: figures.field_errors(FIELD_ERRORS),
        lambda: figures.latency(SUMMARY),
    ],
)
def test_each_chart_draws(draw):
    figures.apply_style()
    assert isinstance(draw(), Figure)


def test_labels_of_close_points_do_not_overlap():
    figures.apply_style()
    fig = figures.accuracy_vs_cost(SUMMARY)
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    first, second = (text.get_window_extent(renderer) for text in ax.texts)
    assert {text.get_text() for text in ax.texts} == set(MODELS)
    assert not first.overlaps(second)


def test_widen_keeps_the_anchored_edge():
    box = Bbox.from_extents(10, 0, 50, 10)
    assert figures._widen(box, "left").x0 == 10
    assert figures._widen(box, "right").x1 == 50
    assert figures._widen(box, "left").width == pytest.approx(40 * (1 + figures.FONT_SLACK))


def test_saved_svg_is_reproducible(tmp_path):
    figures.apply_style()
    paths = [tmp_path / "a.svg", tmp_path / "b.svg"]
    for path in paths:
        figures.save(figures.latency(SUMMARY), path)
    first, second = (path.read_bytes() for path in paths)
    assert first == second
    assert b"<dc:date>" not in first
