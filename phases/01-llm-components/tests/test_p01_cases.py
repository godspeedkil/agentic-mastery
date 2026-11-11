from collections import Counter

from p01.schema import load_cases

KNOWN_TAGS = {
    "easy",
    "distractor_amounts",
    "distractor_documents",
    "relative_due_date",
    "no_due_date",
    "no_invoice_number",
    "non_english",
    "non_usd",
    "locale_formats",
    "noisy_text",
    "informal",
    "ambiguous_category",
    "requires_arithmetic",
}


def test_dataset_loads_and_ids_are_unique():
    cases = load_cases()
    assert len(cases) == 30
    assert len({case.id for case in cases}) == len(cases)


def test_every_tag_is_known():
    for case in load_cases():
        assert case.tags, case.id
        assert set(case.tags) <= KNOWN_TAGS, case.id


def test_hard_cases_are_well_represented():
    counts = Counter(tag for case in load_cases() for tag in case.tags)
    for tag in ("distractor_amounts", "relative_due_date", "no_due_date", "non_english"):
        assert counts[tag] >= 4, tag
