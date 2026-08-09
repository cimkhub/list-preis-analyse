import csv

from main import load_products_from_parsed_csv
from src.extract import vision
from src.extract.prompts import get_extraction_prompt
from src.extract.text_extract import (
    _build_metro_product,
    classify_category,
    classify_product_family,
    classify_temperature_state,
)
from src.extract.web_extract import parse_json_ld_offers
from src.models import RawProduct
from src.report.parsed_csv import PARSED_CSV_FIELDS, save_parsed_csv


IDENTITY_FIELDS = {
    "product_family",
    "temperature_state",
    "processing_state",
    "calibre",
    "source_brand",
    "brand_evidence",
    "brand_evidence_source",
}


def test_raw_product_uses_safe_identity_defaults_for_legacy_data():
    product = RawProduct(
        supplier="metro",
        product_name="Rinder-Roastbeef",
        category="fleisch",
        price=19.99,
    )

    assert product.product_family == "unknown"
    assert product.temperature_state == "unknown"
    assert product.processing_state == "unknown"
    assert product.calibre is None
    assert product.source_brand is None
    assert product.brand_evidence is None
    assert product.brand_evidence_source == "unknown"


def test_raw_product_normalizes_invalid_or_empty_identity_values_safely():
    product = RawProduct(
        supplier="selgros",
        product_name="Mango",
        category="obst_gemuese",
        price=2.49,
        product_family="not-a-family",
        temperature_state="",
        processing_state=None,
        calibre="  8/12  ",
        source_brand="   ",
        brand_evidence=" ASC Logo ",
        brand_evidence_source="not-a-source",
    )

    assert product.product_family == "unknown"
    assert product.temperature_state == "unknown"
    assert product.processing_state == "unknown"
    assert product.calibre == "8/12"
    assert product.source_brand is None
    assert product.brand_evidence == "ASC Logo"
    assert product.brand_evidence_source == "unknown"


def test_extraction_prompt_contains_identity_contract_and_key_safeguards():
    prompt = get_extraction_prompt("metro")

    for field in IDENTITY_FIELDS:
        assert f'"{field}"' in prompt
    assert 'Tiefgekühlter Fisch bleibt product_family "fisch"' in prompt
    assert '"erntefrisch" allein ist KEIN Hinweis auf "frozen"' in prompt
    assert "Händler-/Supplier-Logo" in prompt


def test_vision_mapping_preserves_extracted_identity_evidence():
    products = vision._raw_items_to_products(
        [
            {
                "product_name": "Black Tiger Garnelen 8/12",
                "description": "roh, mit Schale, gefroren",
                "category": "tk",
                "product_family": "fisch",
                "temperature_state": "frozen",
                "processing_state": "raw_plain",
                "calibre": "8/12",
                "source_brand": "ASC",
                "brand_evidence": "ASC",
                "brand_evidence_source": "image",
                "unit": "kg",
                "quantity": 1,
                "price": 19.99,
                "confidence": 0.94,
            }
        ],
        supplier="selgros",
        source_file="angebot.pdf",
        source_page=4,
    )

    assert len(products) == 1
    product = products[0]
    assert product.category == "tk"
    assert product.product_family == "fisch"
    assert product.temperature_state == "frozen"
    assert product.processing_state == "raw_plain"
    assert product.calibre == "8/12"
    assert product.source_brand == "ASC"
    assert product.brand_evidence == "ASC"
    assert product.brand_evidence_source == "image"


def test_vision_mapping_does_not_override_extracted_category():
    products = vision._raw_items_to_products(
        [
            {
                "product_name": "Friesenkrone Heringssalat",
                "description": "tiefgefroren, im Glas",
                "category": "fisch",
                "product_family": "fisch",
                "temperature_state": "frozen",
                "unit": "glas",
                "price": 3.99,
            }
        ],
        supplier="selgros",
    )

    assert products[0].category == "fisch"
    assert products[0].product_family == "fisch"
    assert products[0].temperature_state == "frozen"


def test_text_fallback_classifies_family_independently_from_temperature():
    assert classify_product_family("Rinderfilet tiefgefroren") == "fleisch"
    assert classify_product_family("Lachsfilet tiefgefroren") == "fisch"
    assert classify_product_family("Filet tiefgefroren") == "unknown"
    assert classify_category("Filet tiefgefroren") == "sonstiges"
    assert classify_product_family("Broccoli tiefgefroren") == "obst_gemuese"
    assert classify_product_family("Himbeeren erntefrisch") == "obst_gemuese"
    assert classify_product_family("Brechbohnen") == "obst_gemuese"
    assert classify_product_family("Friesenkrone Heringssalat") == "fisch"

    assert classify_temperature_state("Broccoli tiefgefroren") == "frozen"
    assert classify_temperature_state("Broccoli Tiefkühl") == "frozen"
    assert classify_temperature_state("Lachsfilet frozen") == "frozen"
    assert classify_temperature_state("Himbeeren erntefrisch") == "unknown"
    assert classify_temperature_state("Frische Forelle") == "fresh"


def test_metro_text_fallback_populates_family_and_temperature_separately():
    product = _build_metro_product(
        "Broccoli",
        ["tiefgefroren"],
        [{"min_qty": 1, "price_net": 3.99, "price_gross": 4.27}],
        2,
        "angebot.pdf",
        None,
        None,
    )

    assert product is not None
    assert product.category == "obst_gemuese"
    assert product.product_family == "obst_gemuese"
    assert product.temperature_state == "frozen"


def test_web_extraction_populates_family_and_explicit_temperature():
    products = parse_json_ld_offers(
        [
            {
                "@type": "Product",
                "name": "Himbeeren",
                "description": "erntefrisch",
                "offers": {"price": "2.49"},
            },
            {
                "@type": "Product",
                "name": "Lachsfilet",
                "description": "tiefgefroren",
                "offers": {"price": "12.99"},
            },
        ],
        supplier="handelshof",
    )

    assert products[0].category == "obst_gemuese"
    assert products[0].product_family == "obst_gemuese"
    assert products[0].temperature_state == "unknown"
    assert products[1].category == "fisch"
    assert products[1].product_family == "fisch"
    assert products[1].temperature_state == "frozen"


def test_load_products_accepts_legacy_csv_without_identity_columns(tmp_path):
    legacy_fields = [field for field in PARSED_CSV_FIELDS if field not in IDENTITY_FIELDS]
    csv_path = tmp_path / "legacy.csv"
    row = {field: "" for field in legacy_fields}
    row.update(
        {
            "supplier": "handelshof",
            "product_name": "Forelle",
            "category": "fisch",
            "unit": "kg",
            "price": "12.99",
            "price_is_net": "False",
            "extraction_confidence": "0.91",
        }
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow(row)

    products = load_products_from_parsed_csv(csv_path)

    assert len(products) == 1
    assert products[0].product_name == "Forelle"
    assert products[0].product_family == "unknown"
    assert products[0].temperature_state == "unknown"
    assert products[0].processing_state == "unknown"


def test_identity_fields_survive_parsed_csv_roundtrip(tmp_path):
    product = RawProduct(
        supplier="metro",
        product_name="Rinder-Roastbeef",
        description="gefroren, aus Argentinien",
        category="tk",
        product_family="fleisch",
        temperature_state="frozen",
        processing_state="raw_cut",
        calibre="ca. 3,5 kg",
        source_brand="Metro Chef",
        brand_evidence="METRO Chef",
        brand_evidence_source="image",
        unit="stueck",
        quantity=3.5,
        price=17.99,
    )

    save_parsed_csv([product], "metro", 32, 2026, str(tmp_path))
    csv_path = tmp_path / "KW32_2026" / "metro.csv"
    loaded = load_products_from_parsed_csv(csv_path)

    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.product_family == "fleisch"
    assert restored.temperature_state == "frozen"
    assert restored.processing_state == "raw_cut"
    assert restored.calibre == "ca. 3,5 kg"
    assert restored.source_brand == "Metro Chef"
    assert restored.brand_evidence == "METRO Chef"
    assert restored.brand_evidence_source == "image"


def test_csv_serialization_does_not_rewrite_category_from_product_text(tmp_path):
    product = RawProduct(
        supplier="selgros",
        product_name="Friesenkrone Heringssalat",
        description="tiefgefroren in der Dose",
        category="fisch",
        product_family="fisch",
        temperature_state="frozen",
        unit="dose",
        price=3.99,
    )

    save_parsed_csv([product], "selgros", 32, 2026, str(tmp_path))
    restored = load_products_from_parsed_csv(
        tmp_path / "KW32_2026" / "selgros.csv"
    )[0]

    assert restored.category == "fisch"
    assert restored.product_family == "fisch"
    assert restored.temperature_state == "frozen"
