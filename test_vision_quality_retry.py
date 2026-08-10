from types import SimpleNamespace

import src.extract.vision as vision


def complete_item(name="Tomaten"):
    return {
        "product_name": name,
        "description": "Klasse I",
        "category": "obst_gemuese",
        "unit": "kg",
        "quantity": 1,
        "price": 2.99,
        "price_is_net": False,
        "price_gross": None,
        "price_tiers": None,
        "confidence": 0.95,
    }


def test_page_quality_detects_missing_price_quantity_and_unit():
    issues = vision.page_extraction_quality_issues(
        [
            {"product_name": "Ohne Preis", "price": None, "quantity": 1, "unit": "kg"},
            {"product_name": "Ohne Menge", "price": 4.99, "quantity": None, "unit": "packung"},
            {"product_name": "Ohne Einheit", "price": 3.49, "quantity": 1, "unit": None},
        ]
    )

    assert issues == [
        "item_1:missing_price",
        "item_2:missing_quantity",
        "item_3:missing_unit",
    ]
    assert vision.page_extraction_quality_issues([complete_item()]) == []
    assert vision.page_extraction_quality_issues(
        [{"product_name": "Rinderhackfleisch", "price": 12.49, "quantity": None, "unit": "kg"}]
    ) == []
    assert vision.page_extraction_quality_issues(
        [
            {
                "product_name": "Hähnchenfleisch",
                "description": "ca. 2,5 kg Packung",
                "price": 6.29,
                "quantity": None,
                "unit": "kg",
            }
        ]
    ) == ["item_1:missing_quantity"]


def test_incomplete_page_is_replaced_atomically_by_one_gemini_36_call(monkeypatch):
    primary = [
        {"product_name": "Alt A", "price": None, "quantity": 1, "unit": "kg"},
        complete_item("Alt B"),
    ]
    retry = [complete_item("Neu A"), complete_item("Neu B"), complete_item("Neu C")]
    calls = []

    monkeypatch.setattr(vision, "extract_products_from_image", lambda *args, **kwargs: primary)

    def fake_analyze(**kwargs):
        calls.append(kwargs)
        return retry

    monkeypatch.setattr(vision, "analyze_image_json", fake_analyze)

    result = vision.extract_products_from_image_with_quality_retry(
        "page-7.png",
        "metro",
        model_name="gemini-2.5-flash",
        max_retries=3,
        temperature=0.1,
        page_number=7,
        source_file="offer.pdf",
    )

    assert result is retry
    assert result[0]["product_name"] == "Neu A"
    assert all(item["product_name"] != "Alt B" for item in result)
    assert len(calls) == 1
    assert calls[0]["model_name"] == "gemini-3.6-flash"
    assert calls[0]["max_retries"] == 1
    assert calls[0]["include_temperature"] is False
    assert calls[0]["operation"] == "product_extraction_quality_retry"


def test_complete_page_does_not_call_quality_retry(monkeypatch):
    primary = [complete_item()]
    monkeypatch.setattr(vision, "extract_products_from_image", lambda *args, **kwargs: primary)

    def unexpected_retry(**kwargs):
        raise AssertionError("Gemini 3.6 retry must not run for a complete page")

    monkeypatch.setattr(vision, "analyze_image_json", unexpected_retry)
    assert vision.extract_products_from_image_with_quality_retry("page.png", "edeka") is primary


def test_failed_or_empty_retry_keeps_primary_page(monkeypatch):
    primary = [{"product_name": "Alt", "price": 4.99, "quantity": None, "unit": "kg"}]
    monkeypatch.setattr(vision, "extract_products_from_image", lambda *args, **kwargs: primary)

    monkeypatch.setattr(vision, "analyze_image_json", lambda **kwargs: None)
    assert vision.extract_products_from_image_with_quality_retry("page.png", "selgros") is primary

    monkeypatch.setattr(vision, "analyze_image_json", lambda **kwargs: [])
    assert vision.extract_products_from_image_with_quality_retry("page.png", "selgros") is primary


def test_gemini_36_call_omits_deprecated_temperature(tmp_path, monkeypatch):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"not-a-real-png-needed-by-fake-client")
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="[]", usage_metadata=None, model_version="gemini-3.6-flash")

    monkeypatch.setattr(vision, "_client", SimpleNamespace(models=FakeModels()))

    result = vision.analyze_image_json(
        str(image_path),
        prompt="same schema",
        system_prompt="json only",
        model_name="gemini-3.6-flash",
        max_retries=1,
        temperature=0.1,
        include_temperature=False,
    )

    assert result == []
    config = captured["config"].model_dump(exclude_none=True)
    assert "temperature" not in config
    assert config["response_mime_type"] == "application/json"
