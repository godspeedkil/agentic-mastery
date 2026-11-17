"""The approval agent's tools: look up vendors, purchase orders, and paid invoices."""

import json
import re
from difflib import SequenceMatcher

from pydantic import BaseModel, ConfigDict

from core.agent import Tool
from core.llm import ToolSpec
from p02.data import (
    Decision,
    Reason,
    Records,
    normalize_invoice_number,
    normalize_po,
)

MAX_VENDOR_MATCHES = 3
MAX_INVOICES = 5
MIN_SIMILARITY = 0.5
"""Vendor names scoring below this are not offered as matches."""

LEGAL_SUFFIXES = {"ltd", "inc", "llc", "co", "corp", "gmbh", "kk", "plc", "sa"}
"""Dropped before comparing names, so "Acme Ltd" and "ACME" compare as equal."""


class Arguments(BaseModel):
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class FindVendor(Arguments):
    name: str
    """The vendor's name as written on the invoice. Case, punctuation, and legal suffixes
    such as Ltd or Inc don't matter."""


class GetPurchaseOrder(Arguments):
    po_number: str
    """As written on the invoice: PO-1042, po 1042, and 1042 all work."""


class FindPaidInvoices(Arguments):
    vendor_id: str
    """The vendor's ID from find_vendor, such as V-101."""
    invoice_number: str | None
    """Optional. Matches regardless of punctuation, an INV prefix, or leading zeros."""


class SubmitDecision(Arguments):
    decision: Decision
    """approve only if every policy check passes; otherwise flag."""
    reasons: list[Reason]
    """Every policy check that fails; empty when approving."""
    vendor_id: str | None
    """The vendor's ID, or null if the vendor isn't in the records."""
    po_number: str | None
    """The PO the invoice cites, as found in the records, or null if it cites none."""
    note: str
    """One sentence explaining the decision."""


def name_key(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower().replace(".", "")).split()
    return " ".join(word for word in words if word not in LEGAL_SUFFIXES)


def make_tools(records: Records) -> list[Tool]:
    """The three lookup tools, bound to one set of records."""
    vendors = {vendor.id: vendor for vendor in records.vendors}
    pos = {po.number: po for po in records.purchase_orders}

    def find_vendor(args: FindVendor) -> str:
        key = name_key(args.name)
        scored = sorted(
            ((SequenceMatcher(None, key, name_key(v.name)).ratio(), v) for v in vendors.values()),
            key=lambda pair: pair[0],
            reverse=True,
        )
        matches = [
            {
                "vendor_id": vendor.id,
                "name": vendor.name,
                "status": vendor.status,
                "currency": vendor.currency,
                "similarity": round(score, 2),
            }
            for score, vendor in scored[:MAX_VENDOR_MATCHES]
            if score >= MIN_SIMILARITY
        ]
        if not matches:
            return json.dumps({"matches": [], "note": f"No vendor resembles {args.name!r}."})
        return json.dumps({"matches": matches})

    def get_purchase_order(args: GetPurchaseOrder) -> str:
        number = normalize_po(args.po_number)
        if not re.fullmatch(r"PO-\d+", number):
            raise ValueError(
                f"{args.po_number!r} is not a PO number: they are PO- followed by digits only, "
                "such as PO-1001. Check the invoice for letters misread as digits."
            )
        po = pos.get(number)
        if po is None:
            return json.dumps({"found": False, "note": f"No purchase order {number} exists."})
        return json.dumps(
            {
                "found": True,
                "po_number": po.number,
                "vendor_id": po.vendor_id,
                "vendor_name": vendors[po.vendor_id].name,
                "status": po.status,
                "currency": po.currency,
                "amount": po.amount,
                "description": po.description,
            }
        )

    def find_paid_invoices(args: FindPaidInvoices) -> str:
        if args.vendor_id not in vendors:
            raise ValueError(f"Unknown vendor_id {args.vendor_id!r}. Get one from find_vendor.")
        paid = [i for i in records.paid_invoices if i.vendor_id == args.vendor_id]
        if args.invoice_number is not None:
            wanted = normalize_invoice_number(args.invoice_number)
            paid = [i for i in paid if normalize_invoice_number(i.invoice_number) == wanted]
        result: dict = {"invoices": [i.model_dump() for i in paid[:MAX_INVOICES]]}
        if len(paid) > MAX_INVOICES:
            result["note"] = (
                f"Showing {MAX_INVOICES} of {len(paid)}. Pass invoice_number to narrow."
            )
        return json.dumps(result)

    return [
        Tool(
            ToolSpec(
                name="find_vendor",
                description=(
                    "Look up a vendor by name. Returns up to 3 closest matches with their ID, "
                    "status (active or blocked), and currency, each with a similarity score "
                    "from 0 to 1. Similar names can belong to different companies: accept a "
                    "match only if it is clearly the vendor on the invoice."
                ),
                input_type=FindVendor,
            ),
            find_vendor,
        ),
        Tool(
            ToolSpec(
                name="get_purchase_order",
                description=(
                    "Get a purchase order by number: its vendor, status (open or closed), "
                    'currency, and approved amount. Returns {"found": false} if no such PO '
                    "exists."
                ),
                input_type=GetPurchaseOrder,
            ),
            get_purchase_order,
        ),
        Tool(
            ToolSpec(
                name="find_paid_invoices",
                description=(
                    "List invoices from a vendor that were already paid, to check for "
                    "duplicates. Pass the invoice number to check one invoice directly."
                ),
                input_type=FindPaidInvoices,
            ),
            find_paid_invoices,
        ),
    ]


SUBMIT_DECISION = ToolSpec(
    name="submit_decision",
    description=(
        "Record your decision on the invoice and end the review. Call it exactly once, after "
        "checking the vendor, the purchase order, and earlier payments."
    ),
    input_type=SubmitDecision,
)
