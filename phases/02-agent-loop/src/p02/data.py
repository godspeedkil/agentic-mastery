"""The approval lab's records, its labeled cases, and how reference numbers are compared."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Currency = Literal["USD", "EUR", "GBP", "JPY", "CAD"]
Decision = Literal["approve", "flag"]
Reason = Literal[
    "unknown_vendor",
    "vendor_blocked",
    "missing_po",
    "po_not_found",
    "po_wrong_vendor",
    "po_closed",
    "currency_mismatch",
    "over_po_amount",
    "duplicate_invoice",
]

DATA = Path(__file__).parents[2] / "data"
TOLERANCE = 0.02
"""An invoice may exceed its PO amount by up to 2% and still be approved."""


class Vendor(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    status: Literal["active", "blocked"]
    currency: Currency


class PurchaseOrder(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: str
    vendor_id: str
    status: Literal["open", "closed"]
    currency: Currency
    amount: float
    description: str


class PaidInvoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    vendor_id: str
    invoice_number: str
    total: float
    currency: Currency
    paid_on: str


class Records(BaseModel):
    """Everything the agent's tools can look up."""

    model_config = ConfigDict(frozen=True)

    vendors: list[Vendor]
    purchase_orders: list[PurchaseOrder]
    paid_invoices: list[PaidInvoice]


class Facts(BaseModel):
    """What the invoice actually says, resolved against the records: the labels' ground truth."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    vendor_id: str | None
    """The vendor who sent it; None when it isn't in the records."""
    po_number: str | None
    """The PO it cites, normalized; None when it cites none."""
    invoice_number: str
    total: float
    currency: Currency


class Expected(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Decision
    reasons: list[Reason]
    """Every check that fails, in policy order; empty when approved."""


class Case(BaseModel):
    """One invoice to decide on."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    id: str
    text: str
    facts: Facts
    expected: Expected
    tags: list[str]
    """What makes the case hard, for slicing results; "clean" when nothing does."""


def load_records(data: Path = DATA) -> Records:
    return Records(
        vendors=json.loads((data / "vendors.json").read_text(encoding="utf-8")),
        purchase_orders=json.loads((data / "purchase_orders.json").read_text(encoding="utf-8")),
        paid_invoices=json.loads((data / "paid_invoices.json").read_text(encoding="utf-8")),
    )


def load_cases(path: Path = DATA / "cases.jsonl") -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case.model_validate(json.loads(line)) for line in lines if line.strip()]


def normalize_po(text: str) -> str:
    """'po 1042', 'PO1042', and '1042' all become 'PO-1042'. Other characters are kept."""
    compact = re.sub(r"[\s_-]", "", text.strip().upper())
    return "PO-" + compact.removeprefix("PO")


def normalize_invoice_number(text: str) -> str:
    """Letters and digits only, uppercase, without an 'INV' prefix or leading zeros."""
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return compact.removeprefix("INV").lstrip("0")
