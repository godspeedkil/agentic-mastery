"""Field-by-field comparison of a predicted Invoice with the expected one."""

import unicodedata

from p01.schema import Invoice

FIELDS = ("vendor", "invoice_number", "total", "currency", "due_date", "category")

CENT = 0.005
"""Totals within half a cent count as equal: the same amount once rounded to cents."""


def normalize_name(text: str) -> str:
    """Compare names on letters and digits only, ignoring case, punctuation and spacing.

    "Desk & Drawer Co." and "desk & drawer co" match; "Desk and Drawer Co" does not.
    NFKC folds full-width and other compatibility characters into their plain forms.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    kept = "".join(char if char.isalnum() else " " for char in folded)
    return " ".join(kept.split())


def normalize_reference(text: str | None) -> str | None:
    """References keep their hyphens and slashes; a leading "#" and spaces are ignored."""
    if text is None:
        return None
    folded = unicodedata.normalize("NFKC", text).casefold().strip().lstrip("#")
    return "".join(folded.split())


def score(expected: Invoice, predicted: Invoice | None) -> dict[str, bool]:
    """Whether each field is correct. A missing prediction gets every field wrong."""
    if predicted is None:
        return dict.fromkeys(FIELDS, False)
    return {
        "vendor": normalize_name(predicted.vendor) == normalize_name(expected.vendor),
        "invoice_number": (
            normalize_reference(predicted.invoice_number)
            == normalize_reference(expected.invoice_number)
        ),
        "total": abs(predicted.total - expected.total) < CENT,
        "currency": predicted.currency == expected.currency,
        "due_date": predicted.due_date == expected.due_date,
        "category": predicted.category == expected.category,
    }
