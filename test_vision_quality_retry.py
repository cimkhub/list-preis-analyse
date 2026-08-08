from src.extract import vision


def _valid_item(name: str = "Rinderfilet", price: float = 19.99, confidence: float = 0.95):
    return {
        "product_name": name,
        "description": None,
        "category": "fleisch",
        "unit": "kg",
        "quantity": 1,
        "price": price,
        "confidence": confidence,
    }


def test_quality_check_flags_missing_price_garbled_text_and_low_confidence():
    issues = vision.extraction_quality_issues(
        [
            {
                "product_name": "@@@###",
                "description": "ok",
                "price": None,
                "confidence": 0.4,
            }
        ]
    )

    assert any("Name unleserlich" in issue for issue in issues)
    assert any("Preis fehlt/ungültig" in issue for issue in issues)
    assert any("geringe Sicherheit" in issue for issue in issues)


def test_unclear_page_is_retried_exactly_once_with_gemini_36_flash(monkeypatch):
    calls = []

    def fake_extract(image_path, supplier, **kwargs):
        calls.append({"image_path": image_path, "supplier": supplier, **kwargs})
        if kwargs.get("operation") == "product_extraction_quality_retry":
            return [_valid_item("Rinderfilet", 21.99)]
        return [{"product_name": "Rinderfilet", "price": None, "confidence": 0.5}]

    monkeypatch.setattr(vision, "extract_products_from_image", fake_extract)

    products = vision.extract_products_from_pdf_images(
        ["page-1.png"],
        supplier="metro",
        source_file="offer.pdf",
        model_name="gemini-2.5-flash",
        max_concurrent_requests=1,
    )

    assert len(calls) == 2
    retry_call = calls[1]
    assert retry_call["model_name"] == "gemini-3.6-flash"
    assert retry_call["max_retries"] == 1
    assert retry_call["operation"] == "product_extraction_quality_retry"
    assert products[0].product_name == "Rinderfilet"
    assert products[0].price == 21.99


def test_clear_page_does_not_use_quality_retry(monkeypatch):
    calls = []

    def fake_extract(image_path, supplier, **kwargs):
        calls.append(kwargs)
        return [_valid_item()]

    monkeypatch.setattr(vision, "extract_products_from_image", fake_extract)

    products = vision.extract_products_from_pdf_images(
        ["page-1.png"],
        supplier="selgros",
        model_name="gemini-2.5-flash",
        max_concurrent_requests=1,
    )

    assert len(calls) == 1
    assert len(products) == 1


def test_worse_retry_keeps_original_page_result(monkeypatch):
    calls = []

    def fake_extract(image_path, supplier, **kwargs):
        calls.append(kwargs)
        if kwargs.get("operation") == "product_extraction_quality_retry":
            return [{"product_name": "@@@###", "price": None, "confidence": 0.2}]
        return [_valid_item("Rinderhüfte", 18.49, confidence=0.6)]

    monkeypatch.setattr(vision, "extract_products_from_image", fake_extract)

    products = vision.extract_products_from_pdf_images(
        ["page-1.png"],
        supplier="handelshof",
        model_name="gemini-2.5-flash",
        max_concurrent_requests=1,
    )

    assert len(calls) == 2
    assert products[0].product_name == "Rinderhüfte"
    assert products[0].price == 18.49


def test_gemini_36_retry_omits_deprecated_temperature():
    retry_config = vision._generation_config("gemini-3.6-flash", "system", 0.1)
    legacy_config = vision._generation_config("gemini-2.5-flash", "system", 0.1)

    assert retry_config.temperature is None
    assert legacy_config.temperature == 0.1


def test_empty_retry_never_replaces_a_nonempty_original_result():
    incomplete_original = [{"product_name": "Rinderfilet", "price": None, "confidence": 0.5}]

    assert vision._page_quality_rank(incomplete_original) > vision._page_quality_rank([])
