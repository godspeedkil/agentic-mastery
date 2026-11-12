import pytest

from core.llm import CallRecord, Usage
from p01.analysis import (
    failures,
    field_error_rates,
    per_prediction,
    summarize,
    tag_error_rates,
    wilson_interval,
)
from p01.run import Prediction
from p01.schema import load_cases
from p01.scoring import score

CASES = load_cases()[:2]
GOOD, FLAKY = "claude-haiku-5-5", "gpt-6-luna"


def write_run(run_dir):
    """GOOD is always right. FLAKY gets case 0's total wrong in repeat 1 only.

    Every prediction costs one call of $0.001 and 1 s, except FLAKY's mistake, which
    needed a repair: two calls.
    """
    run_dir.mkdir()
    predictions, calls = [], []
    for model in (GOOD, FLAKY):
        for case in CASES:
            for repeat in (1, 2, 3):
                invoice = case.expected
                wrong = model == FLAKY and case is CASES[0] and repeat == 1
                if wrong:
                    invoice = invoice.model_copy(update={"total": invoice.total + 1})
                provider = "anthropic" if model == GOOD else "openai"
                predictions.append(
                    Prediction(
                        run_id="r",
                        case_id=case.id,
                        provider=provider,
                        model=model,
                        repeat=repeat,
                        invoice=invoice,
                        scores=score(case.expected, invoice),
                    )
                )
                for attempt in range(1, 3 if wrong else 2):
                    calls.append(
                        CallRecord(
                            run_id="r",
                            labels={"case": case.id, "repeat": str(repeat)},
                            provider=provider,
                            model=model,
                            attempt=attempt,
                            usage=Usage(input_tokens=500, output_tokens=200, reasoning_tokens=150),
                            cost_usd=0.001,
                            latency_ms=1000.0,
                            stop_reason="end_turn",
                        )
                    )
    lines = lambda items: "".join(item.model_dump_json() + "\n" for item in items)  # noqa: E731
    (run_dir / "predictions.jsonl").write_text(lines(predictions))
    (run_dir / "calls.jsonl").write_text(lines(calls))
    return run_dir


@pytest.fixture
def frame(tmp_path):
    return per_prediction(write_run(tmp_path / "run"))


def test_calls_are_summed_per_prediction(frame):
    assert len(frame) == 2 * len(CASES) * 3
    mistake = frame[(frame.model == FLAKY) & (frame.case_id == CASES[0].id) & (frame.repeat == 1)]
    assert mistake["calls"].item() == 2
    assert mistake["cost_usd"].item() == pytest.approx(0.002)
    assert frame["calls"].sum() == len(frame) + 1


def test_summary_rates_and_consistency(frame):
    summary = summarize(frame)
    assert list(summary.index) == [GOOD, FLAKY]
    assert summary.loc[GOOD, "exact_rate"] == 1.0
    assert summary.loc[FLAKY, "exact_rate"] == pytest.approx(5 / 6)
    # FLAKY: case 0 right in some repeats but not all; case 1 right in all.
    assert summary.loc[FLAKY, "all_repeats_correct"] == 0.5
    assert summary.loc[FLAKY, "any_repeat_correct"] == 1.0
    assert summary.loc[GOOD, "cost_per_1k_usd"] == pytest.approx(1.0)
    assert summary.loc[GOOD, "latency_p50_s"] == pytest.approx(1.0)


def test_interval_contains_the_estimate(frame):
    row = summarize(frame).loc[FLAKY]
    assert row.ci_low <= row.exact_rate <= row.ci_high


def test_perfect_score_still_has_a_lower_bound():
    low, high = wilson_interval(1.0, n=30)
    assert high == 1.0
    assert 0.85 < low < 0.92  # zero failures in 30 cases does not prove 100%


def test_interval_narrows_with_more_cases():
    small = wilson_interval(0.9, n=30)
    large = wilson_interval(0.9, n=300)
    assert (large[1] - large[0]) < (small[1] - small[0])
    assert small[0] < 0.9 < small[1]


def test_interval_uses_cases_not_predictions(frame):
    row = summarize(frame).loc[FLAKY]
    assert (row.ci_low, row.ci_high) == wilson_interval(row.exact_rate, n=len(CASES))


def test_field_error_rates_point_at_the_wrong_field(frame):
    rates = field_error_rates(frame)
    assert rates.loc[FLAKY, "total"] == pytest.approx(1 / 6)
    assert rates.drop(columns="total").to_numpy().sum() == 0


def test_tag_error_rates_cover_each_tag_of_the_failing_case(frame):
    rates = tag_error_rates(frame, CASES)
    for tag in CASES[0].tags:
        assert rates.loc[tag, FLAKY] > 0
    assert (rates[GOOD] == 0).all()


def test_failures_list_expected_and_answered_values(frame):
    table = failures(frame, CASES)
    assert len(table) == 1
    row = table.iloc[0]
    # Brackets, not attributes: row.repeat would be the Series.repeat method.
    assert (row["case_id"], row["field"], row["model"], row["repeat"]) == (
        CASES[0].id,
        "total",
        FLAKY,
        1,
    )
    assert row["got"] == pytest.approx(row["expected"] + 1)
