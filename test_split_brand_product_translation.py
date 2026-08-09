import csv
import json

import pytest

import classify_fresh_food_relevance as relevance
import split_brand_product as sbp
from match_competitor_products import display_product_name, product_name_from_offer_cells
from split_brand_product import (
    CACHE_PROMPT_VERSION,
    DEFAULT_DEEPSEEK_MODEL,
    SemanticProtectionError,
    TranslationFormatError,
    build_prompt,
    cache_key,
    extract_certifications,
    normalize_split,
    verified_local_brand,
)


def test_prompt_requests_only_one_plain_text_final_name():
    prompt = build_prompt(
        {
            "product_name": "Pork Back",
            "description": "boneless",
            "category": "fleisch",
        }
    )

    assert "Return exactly one final product name in German as plain text." in prompt
    assert "Do not return JSON" in prompt
    assert "Pork Back -> Schweinerücken" in prompt
    assert "Return strict JSON" not in prompt
    assert '"brand"' not in prompt
    assert '"confidence"' not in prompt
    assert '"reason"' not in prompt


def test_translated_plain_text_is_used_without_llm_brand_or_confidence():
    result = normalize_split(
        {"product_name": "Pork Back", "relevance_brand": "ARO"},
        "Schweinerücken",
        "deepseek",
    )

    assert result == {
        "brand": "ARO",
        "product_name_de": "Schweinerücken",
        "product": "Schweinerücken",
        "certifications": "[]",
        "confidence": "",
        "source": "deepseek",
    }


def test_normalize_split_remains_tolerant_of_legacy_local_mapping_shape():
    result = normalize_split(
        {"product_name": "Pork Back"},
        {"brand": "Invented", "product": "Schweinerücken", "confidence": 95},
        "legacy-test",
    )

    assert result["product_name_de"] == "Schweinerücken"
    assert result["brand"] == ""
    assert result["confidence"] == ""


def test_translation_cache_is_versioned_hashed_and_shared_across_sources():
    base = {
        "product_name": "Pork Back 8/12",
        "description": "boneless",
        "category": "fleisch",
        "origin": "EU",
        "calibre": "8/12",
    }
    source_a = {**base, "supplier": "metro", "source_file": "a.pdf", "source_page": "1"}
    source_b = {**base, "supplier": "selgros", "source_file": "b.pdf", "source_page": "9"}

    key = cache_key(source_a)
    assert key == cache_key(source_b)
    assert key.startswith(CACHE_PROMPT_VERSION + ":")
    assert len(key.removeprefix(CACHE_PROMPT_VERSION + ":")) == 64
    assert cache_key({**base, "origin": "DE"}) != key
    assert cache_key({**base, "calibre": "16/20"}) != key
    assert DEFAULT_DEEPSEEK_MODEL == "deepseek-v4-flash"


def test_translation_call_uses_flash_disables_thinking_and_returns_plain_text(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Schweinerücken"}}]}

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

    assert result == "Schweinerücken"
    assert captured["json"]["model"] == "deepseek-v4-flash"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["max_tokens"] == sbp.DEFAULT_MAX_TOKENS
    assert captured["json"]["max_tokens"] <= 80
    assert "plain text" in captured["json"]["messages"][0]["content"]
    assert all("response_format" not in key for key in captured["json"])


def test_translation_call_rejects_any_other_model_before_network(monkeypatch):
    monkeypatch.setattr(
        sbp,
        "get_session",
        lambda: pytest.fail("network must not be reached for an unsupported model"),
    )

    with pytest.raises(RuntimeError, match="deepseek-v4-flash"):
        sbp.call_deepseek("key", "deepseek-v4-pro", "https://example.invalid", 1, "x")


@pytest.mark.parametrize(
    "bad_response",
    [
        '{"product":"Schweinerücken"}',
        '"Schweinerücken"',
        "```\nSchweinerücken\n```",
        "Produktname: Schweinerücken\nGrund: Übersetzung",
    ],
)
def test_plain_text_contract_rejects_json_quotes_markdown_and_multiline(bad_response):
    with pytest.raises(TranslationFormatError):
        sbp.normalize_plain_text_response(bad_response)


def test_verified_brand_prefers_relevance_then_requires_source_evidence_in_field():
    assert verified_local_brand(
        {
            "product_name": "Metro Chef Pommes",
            "relevance_brand": "ARO",
            "source_brand": "Metro Chef",
            "brand_evidence": "Metro Chef",
            "brand_evidence_source": "product_name",
        }
    ) == "ARO"
    assert verified_local_brand(
        {
            "product_name": "Metro Chef Pommes",
            "source_brand": "Metro Chef",
            "brand_evidence": "METRO Chef",
            "brand_evidence_source": "product_name",
        }
    ) == "Metro Chef"
    assert verified_local_brand(
        {
            "product_name": "Pommes",
            "source_brand": "Metro Chef",
            "brand_evidence": "Metro Chef",
            "brand_evidence_source": "product_name",
        }
    ) == ""
    assert verified_local_brand(
        {
            "product_name": "Metro Chef Pommes",
            "source_brand": "Aviko",
            "brand_evidence": "Metro Chef",
            "brand_evidence_source": "product_name",
        }
    ) == ""
    assert verified_local_brand(
        {
            "description": "Premium von Milram",
            "source_brand": "Milram",
            "brand_evidence": "Milram",
            "brand_evidence_source": "description",
        }
    ) == "Milram"
    assert verified_local_brand(
        {
            "source_brand": "Aviko",
            "brand_evidence": "Aviko",
            "brand_evidence_source": "image",
        }
    ) == "Aviko"


def test_certifications_are_local_json_evidence_and_not_brands():
    row = {
        "product_name": "Asc QS Bio Lachs",
        "description": "MSC zertifiziert",
        "source_brand": "ASC",
        "brand_evidence": "ASC",
        "brand_evidence_source": "image",
    }

    assert extract_certifications(row) == ["ASC", "MSC", "QS", "BIO"]
    assert sbp.certifications_json(row) == '["ASC","MSC","QS","BIO"]'
    assert verified_local_brand(row) == ""


def test_all_protected_tokens_phrases_numbers_and_percentages_survive():
    row = {
        "product_name": (
            "Asc Msc Qs Bio Aro Eu Bbq Tk Duroc Black Angus Free-Range "
            "Beef 8/12 80-110 1.2 kg 48%"
        )
    }
    translated = (
        "ASC MSC QS BIO ARO EU BBQ TK Duroc Black Angus Free-Range "
        "Rindfleisch 8/12 80–110 1,2 kg 48 %"
    )

    result = normalize_split(row, translated, "deepseek")

    assert result["product_name_de"] == translated
    assert json.loads(result["certifications"]) == ["ASC", "MSC", "QS", "BIO"]


@pytest.mark.parametrize("incorrect", ["Asc Lachs", "Msc Lachs", "Aro Lachs", "Eu Lachs", "Bbq Lachs", "Tk Lachs"])
def test_protected_acronyms_must_remain_exact_uppercase(incorrect):
    original = incorrect.split()[0].upper() + " Salmon"
    with pytest.raises(SemanticProtectionError):
        normalize_split({"product_name": original}, incorrect, "deepseek")


def test_red_snapper_can_never_become_rotbarsch():
    with pytest.raises(SemanticProtectionError):
        normalize_split(
            {"product_name": "ASC Red Snapper Filet 8/12"},
            "ASC Rotbarsch Filet 8/12",
            "deepseek",
        )


@pytest.mark.parametrize("incorrect", ["Forelle Rot", "Forelle", "Forellenfilet"])
def test_lachsforelle_can_never_be_generalized(incorrect):
    with pytest.raises(SemanticProtectionError):
        normalize_split({"product_name": "Lachsforelle"}, incorrect, "deepseek")


def test_semantic_violation_falls_back_immediately_without_retry(monkeypatch):
    calls = []

    def wrong_translation(*args, **kwargs):
        calls.append(1)
        return "Rotbarsch Filet"

    monkeypatch.setattr(sbp, "call_deepseek", wrong_translation)
    monkeypatch.setattr(sbp.time, "sleep", lambda _seconds: pytest.fail("must not retry"))

    _, result = sbp.split_row(
        0,
        {"product_name": "Red Snapper Filet"},
        "key",
        DEFAULT_DEEPSEEK_MODEL,
        sbp.DEFAULT_DEEPSEEK_BASE_URL,
        10,
        3,
    )

    assert len(calls) == 1
    assert result["product_name_de"] == "Red Snapper Filet"
    assert result["source"] == "fallback"


def test_transient_call_error_uses_only_existing_retry_loop(monkeypatch):
    calls = []

    def flaky_translation(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return "Schweinerücken"

    monkeypatch.setattr(sbp, "call_deepseek", flaky_translation)
    monkeypatch.setattr(sbp.time, "sleep", lambda _seconds: None)

    _, result = sbp.split_row(
        0,
        {"product_name": "Pork Back"},
        "key",
        DEFAULT_DEEPSEEK_MODEL,
        sbp.DEFAULT_DEEPSEEK_BASE_URL,
        10,
        3,
    )

    assert len(calls) == 2
    assert result["product_name_de"] == "Schweinerücken"


def test_same_source_content_uses_one_call_and_keeps_row_local_brands(tmp_path, monkeypatch):
    input_path = tmp_path / "products.csv"
    output_path = tmp_path / "translated.csv"
    fieldnames = [
        "supplier",
        "product_name",
        "description",
        "category",
        "origin",
        "calibre",
        "Relevant",
        "Relevant Time",
        "relevance_brand",
    ]
    with input_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "supplier": "metro",
                "product_name": "Pork Back",
                "description": "boneless",
                "category": "fleisch",
                "origin": "EU",
                "calibre": "",
                "Relevant": "Ja",
                "Relevant Time": "Ja",
                "relevance_brand": "ARO",
            }
        )
        writer.writerow(
            {
                "supplier": "selgros",
                "product_name": "Pork Back",
                "description": "boneless",
                "category": "fleisch",
                "origin": "EU",
                "calibre": "",
                "Relevant": "Ja",
                "Relevant Time": "Ja",
                "relevance_brand": "Milram",
            }
        )

    calls = []
    cache_items = []

    def fake_split_row(index, row, *args):
        calls.append(index)
        return index, normalize_split(row, "Schweinerücken", "deepseek")

    monkeypatch.setattr(sbp, "read_cache", lambda: {})
    monkeypatch.setattr(sbp, "append_cache", cache_items.append)
    monkeypatch.setattr(sbp, "get_env_any", lambda *names: "test-key")
    monkeypatch.setattr(sbp, "split_row", fake_split_row)

    sbp.run_brand_product_split(input_path, output_path, workers=1)

    with output_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(calls) == 1
    assert cache_items[0]["product_name_de"] == "Schweinerücken"
    assert cache_items[0]["certifications"] == "[]"
    assert [row["product_name_de"] for row in rows] == ["Schweinerücken", "Schweinerücken"]
    assert [row["brand"] for row in rows] == ["ARO", "Milram"]


def test_offline_run_keeps_product_name_byte_identical_and_adds_fields(tmp_path, monkeypatch):
    input_path = tmp_path / "products.csv"
    output_path = tmp_path / "translated.csv"
    original_name = "aSC  Pork BACK 8/12 48%"
    with input_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "product_name",
                "description",
                "category",
                "Relevant",
                "Relevant Time",
                "relevance_brand",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product_name": original_name,
                "description": "",
                "category": "fleisch",
                "Relevant": "Ja",
                "Relevant Time": "Ja",
                "relevance_brand": "ARO",
            }
        )

    monkeypatch.setattr(sbp, "read_cache", lambda: {})
    sbp.run_brand_product_split(input_path, output_path, skip_llm=True)

    with output_path.open("r", newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["product_name"] == original_name
    assert row["product_name_original"] == original_name
    assert row["product_name_de"] == original_name
    assert row["product"] == original_name
    assert row["brand"] == "ARO"
    assert row["certifications"] == '["ASC"]'
    assert row["brand_product_confidence"] == ""


def test_relevance_save_rows_no_longer_normalizes_product_name(tmp_path):
    output_path = tmp_path / "relevant.csv"
    original_name = "mSC  Pork BACK 8/12"

    relevance.save_rows(
        output_path,
        ["product_name", "valid_from", "valid_to"],
        [{"product_name": original_name, "valid_from": "", "valid_to": ""}],
        [("Ja", "Core")],
    )

    with output_path.open("r", newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["product_name"] == original_name


def test_translated_product_flows_into_final_product_column():
    product = {
        "product_name": "Pork Back",
        "product_name_original": "Pork Back",
        "product_name_de": "Schweinerücken",
        "product": "Schweinerücken",
    }

    assert display_product_name(product) == "Schweinerücken"
    assert product_name_from_offer_cells({"Metro": "Produkt: Schweinerücken"}) == "Schweinerücken"
