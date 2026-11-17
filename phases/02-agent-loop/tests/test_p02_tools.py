import json

import pytest
from pydantic import ValidationError

from core.agent import run_tool
from core.llm import ToolCall
from p02.data import Records, load_records
from p02.tools import (
    MAX_INVOICES,
    SUBMIT_DECISION,
    FindPaidInvoices,
    FindVendor,
    GetPurchaseOrder,
    SubmitDecision,
    make_tools,
    name_key,
)

TOOLS = {tool.spec.name: tool for tool in make_tools(load_records())}


def run(tool_name: str, **arguments) -> dict:
    tool = TOOLS[tool_name]
    return json.loads(tool.run(tool.spec.input_type(**arguments)))


def test_legal_suffixes_and_punctuation_are_ignored():
    assert name_key("ACME LTD.") == name_key("Acme") == "acme"
    assert name_key("Sakura Components K.K.") == "sakura components"


def test_find_vendor_matches_a_variant_spelling():
    [match] = run("find_vendor", name="GRANITE FACILITIES MGMT")["matches"]
    assert match["vendor_id"] == "V-112"
    assert match["name"] == "Granite Facilities Management"
    assert set(match) == {"vendor_id", "name", "status", "currency", "similarity"}


def test_find_vendor_shows_close_names_with_their_scores():
    matches = run("find_vendor", name="Acme Labs")["matches"]
    assert [m["vendor_id"] for m in matches] == ["V-101", "V-102"]
    assert all(m["similarity"] < 0.7 for m in matches)


def test_find_vendor_says_when_nothing_resembles_the_name():
    result = run("find_vendor", name="Zenith Media Group")
    assert result["matches"] == []
    assert "No vendor resembles" in result["note"]


def test_get_purchase_order_accepts_common_formats_and_names_the_vendor():
    for text in ("PO-1003", "po 1003", "1003"):
        po = run("get_purchase_order", po_number=text)
        assert po["found"] and po["po_number"] == "PO-1003"
        assert po["vendor_name"] == "Acme Logistics Inc."


def test_a_missing_po_is_an_answer_not_an_error():
    assert run("get_purchase_order", po_number="PO-1099") == {
        "found": False,
        "note": "No purchase order PO-1099 exists.",
    }


def test_an_ocr_letter_in_a_po_number_is_an_error_that_says_what_to_check():
    call = ToolCall(id="c1", name="get_purchase_order", arguments='{"po_number": "PO-1O42"}')
    result = run_tool(call, TOOLS)
    assert result.is_error
    assert "PO- followed by digits only" in result.content
    assert "letters misread as digits" in result.content


def test_find_paid_invoices_matches_reformatted_numbers():
    [paid] = run("find_paid_invoices", vendor_id="V-101", invoice_number="42")["invoices"]
    assert paid["invoice_number"] == "INV-0042"


def test_find_paid_invoices_lists_a_vendors_payments():
    invoices = run("find_paid_invoices", vendor_id="V-104", invoice_number=None)["invoices"]
    assert {i["invoice_number"] for i in invoices} == {"55790", "55871"}


def test_find_paid_invoices_caps_long_lists_and_says_so():
    many = Records(
        vendors=load_records().vendors,
        purchase_orders=[],
        paid_invoices=[
            {
                "vendor_id": "V-101",
                "invoice_number": f"N-{i}",
                "total": 1.0,
                "currency": "USD",
                "paid_on": "2026-01-01",
            }
            for i in range(8)
        ],
    )
    tool = {t.spec.name: t for t in make_tools(many)}["find_paid_invoices"]
    result = json.loads(tool.run(FindPaidInvoices(vendor_id="V-101", invoice_number=None)))
    assert len(result["invoices"]) == MAX_INVOICES
    assert result["note"] == "Showing 5 of 8. Pass invoice_number to narrow."


def test_an_unknown_vendor_id_is_an_error_that_points_to_find_vendor():
    call = ToolCall(
        id="c1",
        name="find_paid_invoices",
        arguments='{"vendor_id": "Acme", "invoice_number": null}',
    )
    result = run_tool(call, TOOLS)
    assert result.is_error
    assert "Get one from find_vendor" in result.content


def test_every_argument_is_described_for_the_model():
    for model in (FindVendor, GetPurchaseOrder, FindPaidInvoices, SubmitDecision):
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name}"


def test_submit_decision_rejects_reasons_outside_the_policy():
    with pytest.raises(ValidationError):
        SUBMIT_DECISION.input_type.model_validate_json(
            '{"decision": "flag", "reasons": ["looks_odd"], "vendor_id": null,'
            ' "po_number": null, "note": "x"}'
        )
