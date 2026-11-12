from datetime import date

import pytest

from p01.schema import Invoice
from p01.scoring import FIELDS, normalize_name, normalize_reference, score

EXPECTED = Invoice(
    vendor="Desk & Drawer Co.",
    invoice_number="2026-0098",
    total=147.00,
    currency="USD",
    due_date=date(2026, 9, 12),
    category="office_supplies",
)


def predicted(**changes) -> Invoice:
    return EXPECTED.model_copy(update=changes)


def test_identical_prediction_is_fully_correct():
    assert score(EXPECTED, EXPECTED) == dict.fromkeys(FIELDS, True)


def test_missing_prediction_is_fully_wrong():
    assert score(EXPECTED, None) == dict.fromkeys(FIELDS, False)


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        ("Desk & Drawer Co.", "desk & drawer co", True),
        ("Hollis & Grey Ltd", "Hollis & Grey Ltd.", True),
        ("Maple Ridge IT Services Inc.", "Maple  Ridge IT Services, Inc", True),
        ("サクラ電機株式会社", "サクラ電機株式会社", True),
        ("ＡＢＣ Corp", "ABC Corp", True),
        ("Desk & Drawer Co.", "Desk and Drawer Co", False),
        ("Northwind Components", "Northwind", False),
    ],
)
def test_vendor_names_ignore_case_punctuation_and_spacing(a, b, match):
    assert (normalize_name(a) == normalize_name(b)) is match


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        ("A-1042", "#A-1042", True),
        ("HG/0457", "hg/0457", True),
        ("2026-0098", "2026 - 0098", True),
        ("4471-B", "4471B", False),
        ("HG/0457", "HG-0457", False),
        (None, None, True),
        (None, "", False),
    ],
)
def test_references_keep_their_separators(a, b, match):
    assert (normalize_reference(a) == normalize_reference(b)) is match


@pytest.mark.parametrize(("total", "correct"), [(147.004, True), (147.0, True), (147.01, False)])
def test_totals_must_match_to_the_cent(total, correct):
    assert score(EXPECTED, predicted(total=total))["total"] is correct


def test_null_due_date_only_matches_null():
    no_date = EXPECTED.model_copy(update={"due_date": None})
    assert score(no_date, predicted(due_date=None))["due_date"] is True
    assert score(no_date, EXPECTED)["due_date"] is False
    assert score(EXPECTED, predicted(due_date=None))["due_date"] is False


def test_each_field_is_scored_independently():
    result = score(EXPECTED, predicted(total=7700.0, category="hardware"))
    assert [field for field, correct in result.items() if not correct] == ["total", "category"]
