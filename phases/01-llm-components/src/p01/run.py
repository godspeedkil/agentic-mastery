"""Run every benchmarked model on every case, several times, recording calls and scores.

uv run poe bench                      # full run: all models, all cases, 3 repeats
uv run poe bench --limit 2 --repeats 1 --models claude-haiku-5-5   # cheap smoke test
"""

import argparse
import subprocess
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from core.llm import Ledger, LLMClient, Message, ProviderAdapter, Request
from core.llm.errors import LLMError, StructuredOutputError
from p01.models import MODELS, Model, load_prices
from p01.schema import Case, Invoice, load_cases
from p01.scoring import score

RESULTS = Path(__file__).parents[2] / "results" / "runs"

SYSTEM_PROMPT = (
    "You extract structured data from invoices, receipts and billing messages for an "
    "accounts-payable system. Follow each field's description in the schema exactly. "
    "When a document mentions several amounts or references, choose the one the field "
    "describes. Never guess: use null where the schema allows it and the document does "
    "not say."
)

MAX_TOKENS = 8192
"""Generous, because current models think before answering and thinking counts as output."""


class Prediction(BaseModel):
    """The outcome of one model on one case in one repeat. Costs live in the call ledger."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    case_id: str
    provider: str
    model: str
    repeat: int
    invoice: Invoice | None
    scores: dict[str, bool]
    error: str | None = None


def build_request(case: Case, model: Model) -> Request:
    document = f"<document>\n{case.text}\n</document>"
    return Request(
        model=model.id,
        system=SYSTEM_PROMPT,
        messages=[Message.user(document)],
        max_tokens=MAX_TOKENS,
    )


def run_one(client: LLMClient, run_id: str, model: Model, case: Case, repeat: int) -> Prediction:
    labels = {"case": case.id, "repeat": str(repeat)}
    invoice, error = None, None
    try:
        parsed = client.parse(model.provider, build_request(case, model), Invoice, labels=labels)
        invoice = parsed.value
    except (LLMError, StructuredOutputError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return Prediction(
        run_id=run_id,
        case_id=case.id,
        provider=model.provider,
        model=model.id,
        repeat=repeat,
        invoice=invoice,
        scores=score(case.expected, invoice),
        error=error,
    )


def run_benchmark(
    client: LLMClient,
    run_id: str,
    models: Sequence[Model],
    cases: Sequence[Case],
    repeats: int,
    predictions_path: Path,
    workers: int = 8,
    on_done: Callable[[Prediction], None] | None = None,
) -> list[Prediction]:
    """Run all (model, case, repeat) jobs concurrently, appending each prediction as it lands."""
    jobs = [(model, case, r) for model in models for case in cases for r in range(1, repeats + 1)]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results: list[Prediction] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, client, run_id, *job) for job in jobs]
        for future in as_completed(futures):
            prediction = future.result()
            with lock, predictions_path.open("a", encoding="utf-8") as f:
                f.write(prediction.model_dump_json() + "\n")
            results.append(prediction)
            if on_done is not None:
                on_done(prediction)
    return results


def make_providers(models: Sequence[Model]) -> dict[str, ProviderAdapter]:
    """Only build the providers the selected models need, so one API key is enough."""
    needed = {model.provider for model in models}
    providers: dict[str, ProviderAdapter] = {}
    if "anthropic" in needed:
        from core.llm.anthropic_provider import AnthropicProvider

        providers["anthropic"] = AnthropicProvider()
    if "openai" in needed:
        from core.llm.openai_provider import OpenAIProvider

        providers["openai"] = OpenAIProvider()
    return providers


def git_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.stdout.strip()


class RunInfo(BaseModel):
    """What was run, written next to the results so a run explains itself."""

    run_id: str
    started_at: datetime
    git_commit: str | None
    models: list[str]
    cases: int
    repeats: int
    system_prompt: str
    max_tokens: int


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 invoice-extraction benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N cases (smoke tests)")
    parser.add_argument("--models", nargs="+", choices=[m.id for m in MODELS], help="default: all")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    from rich.console import Console
    from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn
    from rich.table import Table

    args = parse_args(argv)
    models = [m for m in MODELS if args.models is None or m.id in args.models]
    cases = load_cases()[: args.limit]
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / run_id
    out.mkdir(parents=True)

    info = RunInfo(
        run_id=run_id,
        started_at=started_at,
        git_commit=git_commit(),
        models=[m.id for m in models],
        cases=len(cases),
        repeats=args.repeats,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    )
    (out / "run.json").write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")

    client = LLMClient(
        make_providers(models),
        prices=load_prices(),
        ledger=Ledger(out / "calls.jsonl"),
        run_id=run_id,
    )
    console = Console()
    total = len(models) * len(cases) * args.repeats
    columns = (*Progress.get_default_columns()[:2], MofNCompleteColumn(), TimeElapsedColumn())
    with Progress(*columns, console=console) as progress:
        task = progress.add_task(f"run {run_id}", total=total)
        predictions = run_benchmark(
            client,
            run_id,
            models,
            cases,
            args.repeats,
            out / "predictions.jsonl",
            workers=args.workers,
            on_done=lambda _: progress.advance(task),
        )

    table = Table(title=f"Exact match (all six fields) — {out}")
    table.add_column("model")
    table.add_column("exact", justify="right")
    table.add_column("errors", justify="right")
    for model in models:
        mine = [p for p in predictions if p.model == model.id]
        exact = sum(all(p.scores.values()) for p in mine)
        errors = sum(p.error is not None for p in mine)
        table.add_row(model.id, f"{exact}/{len(mine)}", str(errors))
    console.print(table)


if __name__ == "__main__":
    main()
