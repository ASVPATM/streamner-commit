from __future__ import annotations

import json
import re
from pathlib import Path

FIXTURE_PATH = Path(__file__).parents[1] / "data" / "fixtures" / "streaming_cases.json"

MODEL_CARD_LABELS = (
    "person",
    "email address",
    "phone number",
    "street address",
    "credit card number",
    "passport number",
)

REQUIRED_CATEGORIES = {
    "multi_word_person",
    "email_address",
    "phone_number",
    "street_address",
    "credit_card_like",
    "passport_like",
    "two_entities",
    "negative_no_pii",
    "entity_at_beginning",
    "entity_at_end",
}


def load_fixture_data() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    parity = len(digits) % 2
    total = 0
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return bool(digits) and total % 10 == 0


def test_fixture_document_is_explicitly_synthetic_debug_data() -> None:
    payload = load_fixture_data()

    assert payload["schema_version"] == 1
    assert payload["fictional"] is True
    assert payload["annotations_are_benchmark_gold"] is False
    assert "not a benchmark" in str(payload["purpose"]).lower()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_fixture_uses_exact_model_card_label_vocabulary() -> None:
    payload = load_fixture_data()

    assert payload["labels"] == list(MODEL_CARD_LABELS)
    cases = payload["cases"]
    annotation_labels = {
        annotation["label"] for case in cases for annotation in case["debug_annotations"]
    }
    assert annotation_labels == set(MODEL_CARD_LABELS)


def test_case_ids_and_required_categories_are_complete() -> None:
    cases = load_fixture_data()["cases"]
    case_ids = [case["id"] for case in cases]
    categories = {category for case in cases for category in case["categories"]}

    assert len(cases) >= 10
    assert len(case_ids) == len(set(case_ids))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", case_id) for case_id in case_ids)
    assert REQUIRED_CATEGORIES <= categories


def test_debug_annotation_offsets_and_text_are_exact() -> None:
    cases = load_fixture_data()["cases"]

    for case in cases:
        text = case["text"]
        assert case["fictional"] is True
        assert isinstance(text, str) and text
        for annotation in case["debug_annotations"]:
            start = annotation["start"]
            end = annotation["end"]
            assert 0 <= start < end <= len(text)
            assert text[start:end] == annotation["text"]
            assert annotation["label"] in MODEL_CARD_LABELS


def test_category_specific_debug_cases_have_required_shapes() -> None:
    cases = load_fixture_data()["cases"]
    by_category = {
        category: case
        for case in cases
        for category in case["categories"]
        if category in REQUIRED_CATEGORIES
    }

    person = by_category["multi_word_person"]["debug_annotations"]
    assert len(person) == 1
    assert person[0]["label"] == "person"
    assert " " in person[0]["text"]

    assert len(by_category["two_entities"]["debug_annotations"]) >= 2
    assert by_category["negative_no_pii"]["debug_annotations"] == []
    assert any(
        annotation["start"] == 0
        for annotation in by_category["entity_at_beginning"]["debug_annotations"]
    )
    ending_case = by_category["entity_at_end"]
    assert any(
        annotation["end"] == len(ending_case["text"])
        for annotation in ending_case["debug_annotations"]
    )


def test_values_use_reserved_or_obviously_synthetic_forms() -> None:
    cases = load_fixture_data()["cases"]
    annotations = [annotation for case in cases for annotation in case["debug_annotations"]]

    for annotation in annotations:
        label = annotation["label"]
        value = annotation["text"]
        if label == "person":
            assert value == "Mira Solenne"
        elif label == "email address":
            assert value.endswith("@example.test")
        elif label == "phone number":
            assert re.fullmatch(r"\+1 \(\d{3}\) 555-01\d{2}", value)
        elif label == "street address":
            assert "Imaginary Lane" in value and value.endswith("ZZ 00000")
        elif label == "credit card number":
            assert not luhn_valid(value)
        elif label == "passport number":
            assert re.fullmatch(r"ZZ0{7}", value)
