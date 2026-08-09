import pytest

import classify_fresh_food_relevance as relevance
from classify_fresh_food_relevance import (
    DEFAULT_DEEPSEEK_MODEL,
    build_prompt,
    classify_row,
    explicit_non_fresh_exclusion_reason,
)
from src.harmonize.customer_rules import (
    apply_customer_category_overrides,
    apply_customer_category_overrides_to_row,
    customer_exclusion_reason,
)


def test_customer_category_overrides_assign_requested_categories():
    assert apply_customer_category_overrides("sonstiges", product_name="Erbsen tiefgefroren") == "tk"
    assert apply_customer_category_overrides("fisch", product_name="Friesenkrone Heringssalat") == "mopro"
    assert apply_customer_category_overrides("fleisch", product_name="Delikatess Bacon") == "wurst"
    assert apply_customer_category_overrides("sonstiges", product_name="Rostbratwürste") == "wurst"


def test_customer_category_overrides_move_glass_and_can_to_other():
    assert apply_customer_category_overrides("wurst", product_name="Hot Dog Würstchen", unit="glas") == "sonstiges"
    assert apply_customer_category_overrides("mopro", product_name="Kondensmilch Dose") == "sonstiges"


def test_customer_exclusions_block_wine_in_other_and_blocked_brands():
    wine_row = apply_customer_category_overrides_to_row(
        {"category": "sonstiges", "product_name": "Wein trocken"}
    )
    assert customer_exclusion_reason(wine_row) == "Wine"

    assert customer_exclusion_reason({"category": "tk", "product_name": "Iglo Rahmspinat"}) == "Blocked brand"
    assert customer_exclusion_reason({"category": "mopro", "brand": "ja!", "product_name": "Milch"}) == "Blocked brand"


def test_classify_row_sends_customer_exclusion_candidate_to_pro(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "Nein|Blocked brand"}}]}

    monkeypatch.setattr(relevance, "call_deepseek", fake_call)
    index, label, reason = classify_row(
        0,
        {"category": "tk", "product_name": "Iglo Gemüse tiefgefroren"},
        api_key="unused",
        model=DEFAULT_DEEPSEEK_MODEL,
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=1,
    )

    assert index == 0
    assert label == "Nein"
    assert reason == "Blocked brand"
    assert len(calls) == 1
    assert calls[0]["model"] == DEFAULT_DEEPSEEK_MODEL


def test_non_fresh_exclusions_cover_customer_feedback_examples():
    examples = [
        ({"category": "fisch", "product_name": "Delikatess-Sahne-Heringsfilets"}, "Prepared Fish"),
        ({"category": "fleisch", "product_name": "Marinierte Schweinesteaks"}, "Prepared"),
        ({"category": "obst_gemuese", "product_name": "Pizzasauce"}, "Sauce"),
        ({"category": "obst_gemuese", "product_name": "Guacamole"}, "Dip"),
        ({"category": "obst_gemuese", "product_name": "Gewürzgurken"}, "Preserved"),
        ({"category": "sonstiges", "product_name": "Toilettenpapier"}, "Non Food"),
    ]

    for row, reason in examples:
        assert explicit_non_fresh_exclusion_reason(row) == reason


def test_classify_row_sends_non_fresh_candidate_to_pro(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "Nein|Prepared Fish"}}]}

    monkeypatch.setattr(relevance, "call_deepseek", fake_call)
    index, label, reason = classify_row(
        0,
        {"category": "fisch", "product_name": "Delikatess-Sahne-Heringsfilets"},
        api_key="unused",
        model=DEFAULT_DEEPSEEK_MODEL,
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=1,
    )

    assert index == 0
    assert label == "Nein"
    assert reason == "Prepared Fish"
    assert len(calls) == 1


def test_relevance_call_locks_pro_and_enables_thinking(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Ja|Fresh Product"}}]}

    class FakeSession:
        def post(self, url, *, headers, json, timeout):
            captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setattr(relevance, "get_session", lambda: FakeSession())
    relevance.call_deepseek(
        api_key="secret",
        model=DEFAULT_DEEPSEEK_MODEL,
        base_url="https://example.invalid",
        timeout_seconds=17,
        prompt="test",
    )

    assert captured["json"]["model"] == DEFAULT_DEEPSEEK_MODEL
    assert captured["json"]["thinking"] == {"type": "enabled"}

    with pytest.raises(RuntimeError, match="locked"):
        relevance.call_deepseek(
            api_key="secret",
            model="deepseek-v4-flash",
            base_url="https://example.invalid",
            timeout_seconds=17,
            prompt="test",
        )


def test_relevance_prompt_mentions_customer_feedback_negative_examples():
    prompt = build_prompt({"category": "obst_gemuese", "product_name": "Guacamole"})

    assert "Delikatess-Sahne-Heringsfilets => Nein|Prepared Fish" in prompt
    assert "Pizzasauce => Nein|Sauce" in prompt
    assert "Guacamole => Nein|Dip" in prompt
    assert "Gewürzgurken => Nein|Preserved" in prompt
    assert "Toilettenpapier => Nein|Non Food" in prompt
