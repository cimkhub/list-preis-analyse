import classify_fresh_food_relevance as relevance
from classify_fresh_food_relevance import build_prompt, classify_row, explicit_non_fresh_exclusion_reason
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


def test_classify_row_applies_customer_exclusion_without_api_call():
    index, label, reason = classify_row(
        0,
        {"category": "tk", "product_name": "Iglo Gemüse tiefgefroren"},
        api_key="unused",
        model="unused",
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=1,
    )

    assert index == 0
    assert label == "Nein"
    assert reason == "Blocked brand"


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


def test_classify_row_applies_non_fresh_exclusions_before_core_category():
    index, label, reason = classify_row(
        0,
        {"category": "fisch", "product_name": "Delikatess-Sahne-Heringsfilets"},
        api_key="unused",
        model="unused",
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=1,
    )

    assert index == 0
    assert label == "Nein"
    assert reason == "Prepared Fish"


def test_classify_row_defaults_to_yes_when_deepseek_output_is_truncated(monkeypatch):
    monkeypatch.setattr(
        relevance,
        "call_deepseek",
        lambda **_kwargs: {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "The output budget ended before the final answer.",
                    },
                }
            ]
        },
    )

    index, label, reason = classify_row(
        150,
        {"category": "sonstiges", "product_name": "Paprikastreifen- oder Würfel-Mix"},
        api_key="unused",
        model="deepseek-v4-flash",
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=3,
    )

    assert index == 150
    assert label == "Ja"
    assert reason == "LLM Fallback"


def test_relevance_prompt_mentions_customer_feedback_negative_examples():
    prompt = build_prompt({"category": "obst_gemuese", "product_name": "Guacamole"})

    assert "Delikatess-Sahne-Heringsfilets => Nein|Prepared Fish" in prompt
    assert "Pizzasauce => Nein|Sauce" in prompt
    assert "Guacamole => Nein|Dip" in prompt
    assert "Gewürzgurken => Nein|Preserved" in prompt
    assert "Toilettenpapier => Nein|Non Food" in prompt
