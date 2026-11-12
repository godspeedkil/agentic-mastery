import json

from core.llm import (
    Ledger,
    LLMClient,
    Request,
    Response,
    RetryPolicy,
    TextBlock,
    Usage,
    read_records,
)
from core.llm.errors import RequestRejectedError
from p01.models import Model
from p01.run import Prediction, build_request, run_benchmark
from p01.schema import load_cases

CASES = load_cases()[:2]
MODEL = Model("anthropic", "claude-haiku-5-5")


class EchoProvider:
    """Answers each case with its expected invoice, except cases listed as failing."""

    name = "anthropic"

    def __init__(self, failing: set[str] = frozenset()) -> None:
        self.by_text = {case.text: case for case in CASES}
        self.failing = failing

    def complete(self, request: Request) -> Response:
        document = request.messages[0].content[0].text
        case = next(c for text, c in self.by_text.items() if text in document)
        if case.id in self.failing:
            raise RequestRejectedError("unknown model", provider="anthropic", status_code=404)
        answer = case.expected.model_dump_json()
        return Response(
            model=request.model,
            content=[TextBlock(text=answer)],
            stop_reason="end_turn",
            usage=Usage(input_tokens=500, output_tokens=100),
        )


def job_key(prediction: Prediction) -> tuple[str, int]:
    return prediction.case_id, prediction.repeat


def run(tmp_path, provider, repeats=2):
    client = LLMClient(
        {"anthropic": provider},
        ledger=Ledger(tmp_path / "calls.jsonl"),
        run_id="test-run",
        retry=RetryPolicy(max_attempts=1),
    )
    path = tmp_path / "predictions.jsonl"
    predictions = run_benchmark(client, "test-run", [MODEL], CASES, repeats, path, workers=4)
    return predictions, path


def test_request_wraps_the_document_and_sets_the_prompt():
    request = build_request(CASES[0], MODEL)
    text = request.messages[0].content[0].text
    assert text.startswith("<document>\n") and text.endswith("\n</document>")
    assert CASES[0].text in text
    assert request.system and request.model == MODEL.id


def test_every_job_produces_a_scored_prediction_on_disk(tmp_path):
    predictions, path = run(tmp_path, EchoProvider())

    assert len(predictions) == len(CASES) * 2
    assert all(all(p.scores.values()) for p in predictions)
    on_disk = [Prediction.model_validate_json(line) for line in path.read_text().splitlines()]
    assert sorted(on_disk, key=job_key) == sorted(predictions, key=job_key)


def test_calls_are_labelled_with_case_and_repeat(tmp_path):
    run(tmp_path, EchoProvider())
    records = read_records(tmp_path / "calls.jsonl")
    labels = {(record.labels["case"], record.labels["repeat"]) for record in records}
    assert labels == {(case.id, str(r)) for case in CASES for r in (1, 2)}


def test_provider_errors_become_failed_predictions_not_crashes(tmp_path):
    predictions, _ = run(tmp_path, EchoProvider(failing={CASES[0].id}), repeats=1)

    failed = next(p for p in predictions if p.case_id == CASES[0].id)
    assert failed.invoice is None
    assert failed.error and failed.error.startswith("RequestRejectedError")
    assert not any(failed.scores.values())
    ok = next(p for p in predictions if p.case_id == CASES[1].id)
    assert ok.error is None and all(ok.scores.values())


def test_predictions_file_is_valid_jsonl(tmp_path):
    _, path = run(tmp_path, EchoProvider(), repeats=1)
    for line in path.read_text().splitlines():
        assert json.loads(line)["run_id"] == "test-run"
