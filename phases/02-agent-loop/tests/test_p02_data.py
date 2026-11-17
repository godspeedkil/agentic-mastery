from collections import Counter

import pytest

from p02.data import (
    TOLERANCE,
    Facts,
    Reason,
    Records,
    load_cases,
    load_records,
    normalize_invoice_number,
    normalize_po,
)

KNOWN_TAGS = {
    "clean",
    "non_usd",
    "non_english",
    "locale_formats",
    "dollar_variant",
    "vendor_variant",
    "similar_vendor",
    "unknown_vendor",
    "blocked",
    "ocr_po",
    "noisy_text",
    "po_format",
    "informal",
    "distractor_reference",
    "near_duplicate",
    "missing_po",
    "po_not_found",
    "po_wrong_vendor",
    "po_closed",
    "currency",
    "tolerance_inside",
    "tolerance_boundary",
    "tolerance_outside",
    "over_amount",
    "duplicate",
    "invoice_format",
    "multiple",
}


def policy(facts: Facts, records: Records) -> list[Reason]:
    """The approval policy, written independently of the labels to check them."""
    vendors = {v.id: v for v in records.vendors}
    pos = {p.number: p for p in records.purchase_orders}
    paid = {
        (i.vendor_id, normalize_invoice_number(i.invoice_number)) for i in records.paid_invoices
    }
    vendor = vendors.get(facts.vendor_id) if facts.vendor_id else None
    reasons: list[Reason] = []
    if vendor is None:
        reasons.append("unknown_vendor")
    elif vendor.status == "blocked":
        reasons.append("vendor_blocked")
    if facts.po_number is None:
        reasons.append("missing_po")
    elif (po := pos.get(facts.po_number)) is None:
        reasons.append("po_not_found")
    else:
        if vendor is not None and po.vendor_id != vendor.id:
            reasons.append("po_wrong_vendor")
        if po.status == "closed":
            reasons.append("po_closed")
        if facts.currency != po.currency:
            reasons.append("currency_mismatch")
        elif facts.total > po.amount * (1 + TOLERANCE) + 1e-9:
            reasons.append("over_po_amount")
    if vendor is not None and (vendor.id, normalize_invoice_number(facts.invoice_number)) in paid:
        reasons.append("duplicate_invoice")
    return reasons


def test_every_label_follows_the_policy():
    records = load_records()
    for case in load_cases():
        reasons = policy(case.facts, records)
        assert case.expected.reasons == reasons, case.id
        assert case.expected.decision == ("flag" if reasons else "approve"), case.id


def test_dataset_size_ids_and_tags():
    cases = load_cases()
    assert len(cases) == 50
    assert len({case.id for case in cases}) == len(cases)
    for case in cases:
        assert case.tags and set(case.tags) <= KNOWN_TAGS, case.id


def test_approvals_are_a_third_so_flagging_everything_scores_poorly():
    decisions = Counter(case.expected.decision for case in load_cases())
    assert decisions["approve"] / sum(decisions.values()) >= 1 / 3


def test_every_reason_appears_at_least_twice():
    counts = Counter(r for case in load_cases() for r in case.expected.reasons)
    assert set(counts) == set(Reason.__args__)
    assert min(counts.values()) >= 2, counts


def test_facts_point_at_real_records():
    records = load_records()
    vendor_ids = {v.id for v in records.vendors}
    for case in load_cases():
        assert case.facts.vendor_id is None or case.facts.vendor_id in vendor_ids, case.id
        if "noisy_text" not in case.tags:
            assert case.facts.invoice_number in case.text, case.id


def test_record_keys_are_unique():
    records = load_records()
    assert len({v.id for v in records.vendors}) == len(records.vendors)
    assert len({p.number for p in records.purchase_orders}) == len(records.purchase_orders)


@pytest.mark.parametrize("text", ["PO-1042", "po-1042", "po 1042", "PO1042", "1042", " PO_1042 "])
def test_po_numbers_normalize_to_one_form(text):
    assert normalize_po(text) == "PO-1042"


def test_po_normalization_keeps_ocr_letters():
    assert normalize_po("PO-1O42") == "PO-1O42"


@pytest.mark.parametrize("text", ["INV-0042", "inv 42", "42", "0042"])
def test_invoice_numbers_normalize_to_one_form(text):
    assert normalize_invoice_number(text) == "42"
