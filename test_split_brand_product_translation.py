import split_brand_product as sbp

from split_brand_product import (
    CACHE_PROMPT_VERSION,
    DEFAULT_DEEPSEEK_MODEL,
    build_prompt,
    cache_key,
    normalize_split,
)
from match_competitor_products import display_product_name, product_name_from_offer_cells


def test_existing_split_call_returns_concise_german_product_name():
    prompt = build_prompt(
        {
            "product_name": "Pork Back",
            "description": "boneless",
            "category": "fleisch",
            "supplier": "metro",
        }
    )

    assert "Return product as one concise final German product name." in prompt
    assert "translate only the product words into standard German" in prompt
    assert '"product":"Schweinerücken"' in prompt
    assert "Do not return language labels, translation flags, explanations, or alternatives." in prompt
    assert "detected_language" not in prompt
    assert "translation_reason" not in prompt


def test_translated_product_is_used_as_final_product_value():
    result = normalize_split(
        {"product_name": "Pork Back"},
        {"brand": "", "product": "Schweinerücken", "confidence": 95},
        "deepseek",
    )

    assert result["product"] == "Schweinerücken"
    assert result["source"] == "deepseek"


def test_translation_prompt_has_versioned_cache_and_flash_model():
    row = {
        "product_name": "Pork Back",
        "description": "",
        "category": "fleisch",
        "supplier": "metro",
    }

    assert cache_key(row).startswith(CACHE_PROMPT_VERSION.casefold() + "||")
    assert DEFAULT_DEEPSEEK_MODEL == "deepseek-v4-flash"


def test_translation_call_disables_thinking_and_uses_flash(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"brand":"","product":"Schweinerücken","confidence":95}'
                        }
                    }
                ]
            }

    class FakeSession:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr(sbp, "get_session", lambda: FakeSession())

    result = sbp.call_deepseek(
        "test-key",
        DEFAULT_DEEPSEEK_MODEL,
        sbp.DEFAULT_DEEPSEEK_BASE_URL,
        10,
        "translate",
    )

    assert captured["json"]["model"] == "deepseek-v4-flash"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert result["product"] == "Schweinerücken"


def test_translated_product_flows_into_final_product_column():
    product = {"product_name": "Pork Back", "product": "Schweinerücken"}

    assert display_product_name(product) == "Schweinerücken"
    assert product_name_from_offer_cells({"Metro": "Produkt: Schweinerücken"}) == "Schweinerücken"
