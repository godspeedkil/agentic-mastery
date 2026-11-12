"""Turn a benchmark run's files into tables for the report.

A run is 30 cases x 3 repeats per model, and repeats of one case are strongly correlated,
so uncertainty is estimated by resampling cases, never individual predictions.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from p01.models import MODELS
from p01.schema import Case
from p01.scoring import FIELDS

MODEL_ORDER = [model.id for model in MODELS]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """One row per (model, case, repeat): correctness per field, overall, and any error."""
    rows = []
    for p in read_jsonl(run_dir / "predictions.jsonl"):
        row = {key: p[key] for key in ("run_id", "provider", "model", "case_id", "repeat")}
        row |= {f"{field}_ok": p["scores"][field] for field in FIELDS}
        row["exact"] = all(p["scores"].values())
        row["error"] = p["error"]
        row["invoice"] = p["invoice"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_calls(run_dir: Path) -> pd.DataFrame:
    """One row per API call, with usage flattened and labels turned back into columns."""
    rows = []
    for r in read_jsonl(run_dir / "calls.jsonl"):
        usage = r["usage"]
        rows.append(
            {
                "run_id": r["run_id"],
                "model": r["model"],
                "case_id": r["labels"]["case"],
                "repeat": int(r["labels"]["repeat"]),
                "attempt": r["attempt"],
                "input_tokens": usage["total_input_tokens"],
                "cache_read_tokens": usage["cache_read_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "cost_usd": r["cost_usd"],
                "latency_ms": r["latency_ms"],
                "stop_reason": r["stop_reason"],
                "error": r["error"],
            }
        )
    return pd.DataFrame(rows)


def per_prediction(run_dir: Path) -> pd.DataFrame:
    """Predictions joined with the summed cost, time and tokens of every call behind each."""
    keys = ["model", "case_id", "repeat"]
    calls = load_calls(run_dir)
    totals = calls.groupby(keys).agg(
        calls=("attempt", "size"),
        cost_usd=("cost_usd", lambda s: s.sum(min_count=len(s))),
        latency_ms=("latency_ms", "sum"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        reasoning_tokens=("reasoning_tokens", "sum"),
    )
    return load_predictions(run_dir).merge(totals, on=keys, how="left", validate="one_to_one")


def bootstrap_ci(
    per_case: pd.Series, n_boot: int = 10_000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval for the mean of per-case values, resampling cases with replacement."""
    values = per_case.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: accuracy with a case-level interval, consistency, cost, latency and tokens."""
    rows = []
    for model, group in df.groupby("model"):
        per_case = group.groupby("case_id")["exact"]
        low, high = bootstrap_ci(per_case.mean())
        rows.append(
            {
                "model": model,
                "exact_rate": group["exact"].mean(),
                "ci_low": low,
                "ci_high": high,
                "all_repeats_correct": per_case.all().mean(),
                "any_repeat_correct": per_case.any().mean(),
                "cost_per_1k_usd": group["cost_usd"].mean() * 1000,
                "latency_p50_s": group["latency_ms"].quantile(0.50) / 1000,
                "latency_p95_s": group["latency_ms"].quantile(0.95) / 1000,
                "output_tokens": group["output_tokens"].mean(),
                "reasoning_tokens": group["reasoning_tokens"].mean(),
                "calls_per_prediction": group["calls"].mean(),
            }
        )
    return order_models(pd.DataFrame(rows).set_index("model"))


def field_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of predictions with each field wrong, model x field."""
    rates = 1 - df.groupby("model")[[f"{field}_ok" for field in FIELDS]].mean()
    rates.columns = list(FIELDS)
    return order_models(rates)


def tag_error_rates(df: pd.DataFrame, cases: Sequence[Case]) -> pd.DataFrame:
    """Share of predictions not fully correct, tag x model. A case counts under each tag."""
    tags = pd.DataFrame([(c.id, tag) for c in cases for tag in c.tags], columns=["case_id", "tag"])
    tagged = df.merge(tags, on="case_id")
    rates = 1 - tagged.pivot_table(index="tag", columns="model", values="exact", aggfunc="mean")
    return rates[[m for m in MODEL_ORDER if m in rates.columns]]


def failures(df: pd.DataFrame, cases: Sequence[Case]) -> pd.DataFrame:
    """Every wrong field: what was expected and what the model answered."""
    expected = {c.id: c.expected.model_dump(mode="json") for c in cases}
    rows = []
    for row in df.itertuples():
        for field in FIELDS:
            if not getattr(row, f"{field}_ok"):
                rows.append(
                    {
                        "case_id": row.case_id,
                        "field": field,
                        "model": row.model,
                        "repeat": row.repeat,
                        "expected": expected[row.case_id][field],
                        "got": (row.invoice or {}).get(field),
                        "error": row.error,
                    }
                )
    columns = ["case_id", "field", "model", "repeat", "expected", "got", "error"]
    return pd.DataFrame(rows, columns=columns).sort_values(["case_id", "field", "model", "repeat"])


def order_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in benchmark order (provider, then tier) rather than alphabetical."""
    return frame.loc[[m for m in MODEL_ORDER if m in frame.index]]
