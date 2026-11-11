"""What the benchmark asks models to extract, and the labeled cases it is scored against."""

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Currency = Literal["USD", "EUR", "GBP", "JPY", "CAD", "other"]
Category = Literal["software", "hardware", "services", "travel", "office_supplies", "other"]

DEFAULT_CASES = Path(__file__).parents[2] / "data" / "cases.jsonl"


class Invoice(BaseModel):
    """Fields to extract from an invoice, receipt or billing message."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    vendor: str
    """The business or person issuing the document, exactly as written."""
    invoice_number: str | None
    """Invoice, receipt, folio or booking reference as written; null if there is none."""
    total: float
    """Total of all charges including tax and fees, after discounts and credits, before
    subtracting payments or deposits already made."""
    currency: Currency
    """ISO code of the currency the total is stated in; "other" if none of the listed ones."""
    due_date: date | None
    """Date payment is due. Compute it from relative terms such as "Net 30" using the
    document date. Null if already paid, due on receipt, or not stated."""
    category: Category
    """What is being paid for. office_supplies covers consumables and printing; services
    covers work done by people; other covers memberships, dues and anything else."""


class Case(BaseModel):
    """One labeled document."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    expected: Invoice
    tags: list[str]
    """What makes the case hard, for slicing results; "easy" when nothing does."""


def load_cases(path: Path = DEFAULT_CASES) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case.model_validate(json.loads(line)) for line in lines if line.strip()]
