import csv
import json

import pytest

from main import load_products_from_parsed_csv
from match_competitor_products import amount_label
from src.extract import vision
from src.harmonize.product_identity import (
    build_offer_records,
    make_offer_id,
    offer_sort_key,
)
from src.models import RawProduct
from src.report import parsed_csv


PACKAGING_FIELDS = {
    "price_basis",
    "package_count",
    "package_size_value",
    "package_size_unit",
    "total_content_value",
    "total_content_unit",
    "packaging_type",
    "packaging_raw",
    "certifications",
}

SOURCE_ITEM_FIELDS = {
    "source_document_sha256",
    "source_item_id",
    "source_item_index",
}

QUALITY_PROVENANCE_FIELDS = {
    "primary_extraction_model",
    "selected_extraction_model",
    "quality_retry_attempted",
    "quality_retry_status",
    "quality_retry_model",
    "quality_retry_issues",
    "extraction_schema_version",
}


def _product(**overrides) -> RawProduct:
    values = {
        "supplier": "edeka",
        "product_name": "TK Broccoli",
        "description": "2500 g Beutel",
        "category": "tk",
        "unit": "beutel",
        "quantity": 2500,
        "price": 3.49,
    }
    values.update(overrides)
    return RawProduct(**values)


def _structured_offer(**overrides) -> dict:
    values = {
        "supplier": "selgros",
        "supplier_norm": "Selgros",
        "location": "Braunschweig",
        "price": "5.99",
        "price_per_kg": "",
        "price_is_net": "False",
        "price_gross": "",
        "price_tiers": "",
        "price_basis": "per_package",
        "unit": "beutel",
        "quantity": "1",
        "package_count": "1",
        "package_size_value": "2500",
        "package_size_unit": "g",
        "total_content_value": "2500",
        "total_content_unit": "g",
        "packaging_type": "bag",
        "packaging_raw": "2500 g Beutel",
        "calibre": "",
        "valid_from": "2026-08-10",
        "valid_to": "2026-08-15",
        "calendar_week": "32",
        "year": "2026",
    }
    values.update(overrides)
    return values


def _require_callable(module, name: str):
    function = getattr(module, name, None)
    assert callable(function), f"P6 requires callable {module.__name__}.{name}"
    return function


def test_raw_product_exposes_safe_structured_packaging_defaults():
    first = _product()
    second = _product(product_name="TK Erbsen")

    assert PACKAGING_FIELDS <= set(RawProduct.model_fields)
    assert SOURCE_ITEM_FIELDS <= set(RawProduct.model_fields)
    assert first.price_basis == "unknown"
    assert first.package_count is None
    assert first.package_size_value is None
    assert first.package_size_unit == "unknown"
    assert first.total_content_value is None
    assert first.total_content_unit == "unknown"
    assert first.packaging_type == "unknown"
    assert first.packaging_raw is None
    assert first.certifications == []
    assert first.source_document_sha256 is None
    assert first.source_item_id is None
    assert first.source_item_index is None

    # The list default must not be shared between model instances.
    first.certifications.append("ASC")
    assert second.certifications == []


def test_raw_product_keeps_only_canonical_packaging_enums():
    valid = _product(
        price_basis="per_package",
        package_count=1,
        package_size_value=2500,
        package_size_unit="g",
        total_content_value=2500,
        total_content_unit="g",
        packaging_type="bag",
        packaging_raw="2500 g Beutel",
        certifications=["ASC", "MSC"],
    )

    assert valid.price_basis == "per_package"
    assert valid.package_count == 1
    assert valid.package_size_value == 2500
    assert valid.package_size_unit == "g"
    assert valid.total_content_value == 2500
    assert valid.total_content_unit == "g"
    assert valid.packaging_type == "bag"
    assert valid.packaging_raw == "2500 g Beutel"
    assert valid.certifications == ["ASC", "MSC"]

    invalid = _product(
        price_basis="weekly",
        package_size_unit="tonne",
        total_content_unit="pallet",
        packaging_type="shipping_container",
    )
    assert invalid.price_basis == "unknown"
    assert invalid.package_size_unit == "unknown"
    assert invalid.total_content_unit == "unknown"
    assert invalid.packaging_type == "unknown"


@pytest.mark.parametrize(
    ("raw_text", "size", "packaging_type"),
    [
        ("2500 g Beutel", 2500, "bag"),
        ("1000 g Packung", 1000, "pack"),
        ("800 g Beutel", 800, "bag"),
    ],
)
def test_vision_mapping_separates_content_from_packaging_count(
    raw_text,
    size,
    packaging_type,
):
    products = vision._raw_items_to_products(
        [
            {
                "product_name": "TK Gemüse",
                "description": raw_text,
                "category": "tk",
                "unit": "beutel" if packaging_type == "bag" else "packung",
                "quantity": size,
                "price": 3.49,
                "price_basis": "per_package",
                "package_count": 1,
                "package_size_value": size,
                "package_size_unit": "g",
                "total_content_value": size,
                "total_content_unit": "g",
                "packaging_type": packaging_type,
                "packaging_raw": raw_text,
                "certifications": ["BIO"],
            }
        ],
        supplier="edeka",
        source_file="angebot.pdf",
        source_page=7,
        source_document_sha256="a" * 64,
    )

    assert len(products) == 1
    product = products[0]
    assert product.package_count == 1
    assert product.package_size_value == size
    assert product.package_size_unit == "g"
    assert product.total_content_value == size
    assert product.total_content_unit == "g"
    assert product.packaging_type == packaging_type
    assert product.packaging_raw == raw_text
    assert product.certifications == ["BIO"]


def test_multipack_can_store_count_individual_size_and_total_content():
    products = vision._raw_items_to_products(
        [
            {
                "product_name": "Burger Patties",
                "description": "10 Stück à 80 g, Gesamt 800 g pro Packung",
                "category": "tk",
                "price": 8.99,
                "price_basis": "per_package",
                "package_count": 10,
                "package_size_value": 80,
                "package_size_unit": "g",
                "total_content_value": 800,
                "total_content_unit": "g",
                "packaging_type": "pack",
                "packaging_raw": "10 Stück à 80 g, Gesamt 800 g pro Packung",
            }
        ],
        supplier="metro",
        source_page=3,
        source_document_sha256="b" * 64,
    )

    product = products[0]
    assert product.package_count == 10
    assert product.package_size_value == 80
    assert product.package_size_unit == "g"
    assert product.total_content_value == 800
    assert product.total_content_unit == "g"
    assert product.packaging_type == "pack"


def test_parsed_csv_fields_are_additive_and_include_p6_contract():
    fields = parsed_csv.PARSED_CSV_FIELDS
    legacy_last_field = fields.index("brand_evidence_source")

    assert PACKAGING_FIELDS <= set(fields)
    assert SOURCE_ITEM_FIELDS <= set(fields)
    assert QUALITY_PROVENANCE_FIELDS <= set(fields)
    assert all(fields.index(field) > legacy_last_field for field in PACKAGING_FIELDS)
    assert all(fields.index(field) > legacy_last_field for field in SOURCE_ITEM_FIELDS)
    assert all(
        fields.index(field) > legacy_last_field for field in QUALITY_PROVENANCE_FIELDS
    )


def test_structured_packaging_survives_serializer_roundtrip():
    serialize = _require_callable(parsed_csv, "serialize_product_row")
    deserialize = _require_callable(parsed_csv, "deserialize_product_row")
    product = _product(
        price_basis="per_package",
        package_count=10,
        package_size_value=80,
        package_size_unit="g",
        total_content_value=800,
        total_content_unit="g",
        packaging_type="pack",
        packaging_raw="10 Stück à 80 g, Gesamt 800 g pro Packung",
        certifications=["ASC", "MSC"],
        source_document_sha256="c" * 64,
        source_item_id="src_item_test",
        source_item_index=4,
        primary_extraction_model="gemini-2.5-flash",
        selected_extraction_model="gemini-3.6-flash",
        quality_retry_attempted=True,
        quality_retry_status="selected",
        quality_retry_model="gemini-3.6-flash",
        quality_retry_issues=["Produkt 1: Preis fehlt/ungültig"],
        extraction_schema_version=2,
    )

    row = serialize(product)
    assert json.loads(row["certifications"]) == ["ASC", "MSC"]
    assert json.loads(row["quality_retry_issues"]) == [
        "Produkt 1: Preis fehlt/ungültig"
    ]

    restored = deserialize(row)
    assert isinstance(restored, RawProduct)
    for field in PACKAGING_FIELDS | SOURCE_ITEM_FIELDS | QUALITY_PROVENANCE_FIELDS:
        assert getattr(restored, field) == getattr(product, field)


def test_serializer_does_not_double_encode_existing_json_strings():
    product = _product(
        price_tiers=[{"min_qty": 2, "price": 3.19}],
        certifications=["BIO"],
        quality_retry_issues=["Preis unklar"],
    )
    first = parsed_csv.serialize_product_row(product)
    second = parsed_csv.serialize_product_row(first)

    assert json.loads(second["price_tiers"]) == [{"min_qty": 2, "price": 3.19}]
    assert json.loads(second["certifications"]) == ["BIO"]
    assert json.loads(second["quality_retry_issues"]) == ["Preis unklar"]
    assert parsed_csv.deserialize_product_row(second).price_is_net is False


def test_deserializer_tolerates_invalid_optional_numeric_values():
    product = parsed_csv.deserialize_product_row(
        {
            "supplier": "edeka",
            "product_name": "TK Gemüse",
            "category": "tk",
            "price": "nicht-lesbar",
            "quantity": "?",
            "package_count": "1x",
            "package_size_value": "abc",
            "source_page": "Seite sieben",
        }
    )

    assert product.price == 0
    assert product.quantity is None
    assert product.package_count is None
    assert product.package_size_value is None
    assert product.source_page is None


def test_legacy_csv_row_deserializes_with_safe_p6_defaults():
    deserialize = _require_callable(parsed_csv, "deserialize_product_row")
    product = deserialize(
        {
            "supplier": "handelshof",
            "product_name": "Forelle",
            "category": "fisch",
            "unit": "kg",
            "quantity": "1",
            "price": "12.99",
            "price_is_net": "False",
            "extraction_confidence": "0.91",
        }
    )

    assert isinstance(product, RawProduct)
    assert product.product_name == "Forelle"
    assert product.price_basis == "unknown"
    assert product.package_count is None
    assert product.package_size_value is None
    assert product.package_size_unit == "unknown"
    assert product.total_content_value is None
    assert product.total_content_unit == "unknown"
    assert product.packaging_type == "unknown"
    assert product.packaging_raw is None
    assert product.certifications == []
    assert product.source_document_sha256 is None
    assert product.source_item_id is None
    assert product.source_item_index is None
    assert product.primary_extraction_model is None
    assert product.selected_extraction_model is None
    assert product.quality_retry_attempted is False
    assert product.quality_retry_status == "not_needed"
    assert product.quality_retry_model is None
    assert product.quality_retry_issues == []
    assert product.extraction_schema_version == 1


def test_structured_packaging_survives_csv_file_roundtrip(tmp_path):
    product = _product(
        price_basis="per_package",
        package_count=1,
        package_size_value=2500,
        package_size_unit="g",
        total_content_value=2500,
        total_content_unit="g",
        packaging_type="bag",
        packaging_raw="2500 g Beutel",
        certifications=["BIO"],
        source_document_sha256="d" * 64,
        source_item_id="src_item_csv",
        source_item_index=2,
        primary_extraction_model="gemini-2.5-flash",
        selected_extraction_model="gemini-3.6-flash",
        quality_retry_attempted=True,
        quality_retry_status="selected",
        quality_retry_model="gemini-3.6-flash",
        quality_retry_issues=["Produkt 1: Verpackungsdaten widersprüchlich"],
        extraction_schema_version=2,
    )

    parsed_csv.save_parsed_csv([product], "edeka", 32, 2026, str(tmp_path))
    path = tmp_path / "KW32_2026" / "edeka.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        persisted = next(csv.DictReader(handle))
    restored = load_products_from_parsed_csv(path)[0]

    assert json.loads(persisted["certifications"]) == ["BIO"]
    assert json.loads(persisted["quality_retry_issues"]) == [
        "Produkt 1: Verpackungsdaten widersprüchlich"
    ]
    for field in PACKAGING_FIELDS | SOURCE_ITEM_FIELDS | QUALITY_PROVENANCE_FIELDS:
        assert getattr(restored, field) == getattr(product, field)


@pytest.mark.parametrize(
    ("product", "packaging_word", "content_labels", "forbidden"),
    [
        (
            {
                "quantity": "2500",
                "unit": "beutel",
                "package_count": 1,
                "package_size_value": 2500,
                "package_size_unit": "g",
                "total_content_value": 2500,
                "total_content_unit": "g",
                "packaging_type": "bag",
            },
            "beutel",
            {"2500 g", "2,5 kg"},
            "2500 beutel",
        ),
        (
            {
                "quantity": "1000",
                "unit": "packung",
                "package_count": 1,
                "package_size_value": 1000,
                "package_size_unit": "g",
                "total_content_value": 1000,
                "total_content_unit": "g",
                "packaging_type": "pack",
            },
            "packung",
            {"1000 g", "1 kg"},
            "1000 packung",
        ),
        (
            {
                "quantity": "800",
                "unit": "beutel",
                "package_count": 1,
                "package_size_value": 800,
                "package_size_unit": "g",
                "total_content_value": 800,
                "total_content_unit": "g",
                "packaging_type": "bag",
            },
            "beutel",
            {"800 g", "0,8 kg"},
            "800 beutel",
        ),
    ],
)
def test_amount_label_prefers_structured_content_over_legacy_quantity(
    product,
    packaging_word,
    content_labels,
    forbidden,
):
    label = amount_label(product).casefold()

    assert packaging_word in label
    assert any(content.casefold() in label for content in content_labels)
    assert forbidden not in label


def test_amount_label_renders_multipack_without_losing_total_content():
    label = amount_label(
        {
            "quantity": "1",
            "unit": "packung",
            "package_count": 10,
            "package_size_value": 80,
            "package_size_unit": "g",
            "total_content_value": 800,
            "total_content_unit": "g",
            "packaging_type": "pack",
        }
    )

    assert "10 × 80 g" in label
    assert "800 g" in label
    assert "10 Packung" not in label


def test_amount_label_keeps_legacy_fallback_without_structured_fields():
    assert amount_label({"quantity": "2.5", "unit": "kg"}) == "2,5 kg"


def test_offer_id_distinguishes_every_structured_packaging_dimension():
    base = _structured_offer()
    variants = [
        base,
        {**base, "package_count": "2"},
        {**base, "package_size_value": "1000"},
        {**base, "package_size_unit": "kg"},
        {**base, "total_content_value": "1000"},
        {**base, "total_content_unit": "kg"},
        {**base, "packaging_type": "basket"},
        {**base, "price_basis": "per_kg"},
    ]

    offer_ids = {make_offer_id("pv_tk_gemuese", offer) for offer in variants}
    assert len(offer_ids) == len(variants)


def test_three_kg_crate_and_one_kg_basket_remain_distinct_offers():
    crate = _structured_offer(
        price="28.99",
        package_count="1",
        package_size_value="3",
        package_size_unit="kg",
        total_content_value="3",
        total_content_unit="kg",
        packaging_type="crate",
        packaging_raw="3-kg-Kiste",
    )
    basket = _structured_offer(
        price="11.99",
        package_count="1",
        package_size_value="1",
        package_size_unit="kg",
        total_content_value="1",
        total_content_unit="kg",
        packaging_type="basket",
        packaging_raw="1-kg-Korb",
    )

    assert make_offer_id("pv_pfifferlinge", crate) != make_offer_id(
        "pv_pfifferlinge", basket
    )


def test_structured_offer_id_ignores_conflicting_legacy_amount_aliases():
    legacy_bug_encoding = _structured_offer(unit="beutel", quantity="2500")
    corrected_legacy_alias = _structured_offer(unit="packung", quantity="1")

    assert make_offer_id(
        "pv_tk_gemuese", legacy_bug_encoding
    ) == make_offer_id("pv_tk_gemuese", corrected_legacy_alias)

    no_structured_data = {
        **_structured_offer(),
        "price_basis": "",
        "package_count": "",
        "package_size_value": "",
        "package_size_unit": "",
        "total_content_value": "",
        "total_content_unit": "",
        "packaging_type": "",
        "packaging_raw": "",
    }
    assert make_offer_id(
        "pv_tk_gemuese", {**no_structured_data, "unit": "beutel", "quantity": "2500"}
    ) != make_offer_id(
        "pv_tk_gemuese", {**no_structured_data, "unit": "packung", "quantity": "1"}
    )


def test_offer_sort_prefers_structured_total_over_conflicting_legacy_quantity():
    one_kg = _structured_offer(
        price="5.99",
        quantity="999",
        package_size_value="1",
        package_size_unit="kg",
        total_content_value="1",
        total_content_unit="kg",
        packaging_type="basket",
        packaging_raw="1-kg-Korb",
    )
    three_kg = _structured_offer(
        price="5.99",
        quantity="1",
        package_size_value="3",
        package_size_unit="kg",
        total_content_value="3",
        total_content_unit="kg",
        packaging_type="crate",
        packaging_raw="3-kg-Kiste",
    )
    records = build_offer_records("pv_pfifferlinge", [three_kg, one_kg])
    records_by_total = {
        record.product["total_content_value"]: record for record in records
    }

    assert offer_sort_key(records_by_total["1"]) < offer_sort_key(records_by_total["3"])


def test_source_item_id_is_canonical_and_sensitive_to_source_identity():
    make_source_item_id = _require_callable(vision, "make_source_item_id")
    raw_item = {
        "product_name": "TK Broccoli",
        "price": 3.49,
        "package_count": 1,
        "package_size_value": 2500,
        "package_size_unit": "g",
        "packaging_type": "bag",
    }
    reordered_item = dict(reversed(list(raw_item.items())))

    first = make_source_item_id("a" * 64, 7, raw_item)
    assert first == make_source_item_id("a" * 64, 7, reordered_item)
    assert first != make_source_item_id(
        "a" * 64,
        7,
        raw_item,
        occurrence_index=1,
    )
    assert first != make_source_item_id("b" * 64, 7, raw_item)
    assert first != make_source_item_id("a" * 64, 8, raw_item)
    assert first != make_source_item_id(
        "a" * 64,
        7,
        {**raw_item, "total_content_value": 1000},
    )
    assert first == make_source_item_id(
        "a" * 64,
        7,
        {**raw_item, "price": "3.490", "confidence": 0.2},
    )


def test_source_item_id_normalizes_optional_nulls_and_price_tier_order():
    make_source_item_id = vision.make_source_item_id
    first_item = {
        "product_name": "Rinderfilet",
        "price": 19.99,
        "description": None,
        "price_tiers": [
            {"min_qty": 5, "price": 18.99},
            {"min_qty": 1, "price": 19.99},
        ],
    }
    second_item = {
        "product_name": "Rinderfilet",
        "price": "19.990",
        "price_tiers": [
            {"min_qty": "1.0", "price": "19.990", "max_qty": None},
            {"min_qty": "5.0", "price": "18.990"},
        ],
    }

    assert make_source_item_id("a" * 64, 1, first_item) == make_source_item_id(
        "a" * 64, 1, second_item
    )


def test_mapping_does_not_create_source_item_id_without_document_hash():
    product = vision._raw_items_to_products(
        [{"product_name": "Rinderfilet", "category": "fleisch", "price": 19.99}],
        supplier="metro",
        source_file="missing-offer.pdf",
        source_page=1,
    )[0]

    assert product.source_document_sha256 is None
    assert product.source_item_id is None


def test_source_document_hash_uses_actual_file_bytes(tmp_path):
    source = tmp_path / "offer.pdf"
    source.write_bytes(b"same source bytes")
    renamed = tmp_path / "renamed.pdf"
    renamed.write_bytes(source.read_bytes())

    assert vision.file_sha256(source) == vision.file_sha256(renamed)
    assert len(vision.file_sha256(source)) == 64


def test_source_item_id_survives_raw_item_reordering_in_page_mapping():
    items = [
        {
            "product_name": "TK Broccoli",
            "category": "tk",
            "price": 3.49,
            "price_basis": "per_package",
            "package_count": 1,
            "package_size_value": 2500,
            "package_size_unit": "g",
            "total_content_value": 2500,
            "total_content_unit": "g",
            "packaging_type": "bag",
        },
        {
            "product_name": "TK Erbsen",
            "category": "tk",
            "price": 2.99,
            "price_basis": "per_package",
            "package_count": 1,
            "package_size_value": 1000,
            "package_size_unit": "g",
            "total_content_value": 1000,
            "total_content_unit": "g",
            "packaging_type": "pack",
        },
    ]
    kwargs = {
        "supplier": "edeka",
        "source_file": "local-name-can-change.pdf",
        "source_page": 7,
        "source_document_sha256": "e" * 64,
    }

    first = vision._raw_items_to_products(items, **kwargs)
    second = vision._raw_items_to_products(
        list(reversed(items)),
        **{**kwargs, "source_file": "renamed-copy.pdf"},
    )
    first_ids = {product.product_name: product.source_item_id for product in first}
    second_ids = {product.product_name: product.source_item_id for product in second}

    assert first_ids == second_ids
    assert all(first_ids.values())
    assert len(set(first_ids.values())) == len(items)


def test_duplicate_identical_page_items_receive_distinct_source_item_ids():
    duplicate = {
        "product_name": "TK Broccoli",
        "category": "tk",
        "price": 3.49,
        "price_basis": "per_package",
        "package_count": 1,
        "package_size_value": 2500,
        "package_size_unit": "g",
        "total_content_value": 2500,
        "total_content_unit": "g",
        "packaging_type": "bag",
        "packaging_raw": "2500 g Beutel",
    }

    products = vision._raw_items_to_products(
        [dict(duplicate), dict(duplicate)],
        supplier="edeka",
        source_page=7,
        source_document_sha256="f" * 64,
    )
    repeated = vision._raw_items_to_products(
        [dict(duplicate), dict(duplicate)],
        supplier="edeka",
        source_page=7,
        source_document_sha256="f" * 64,
    )

    assert len(products) == 2
    assert len({product.source_item_id for product in products}) == 2
    assert len({product.source_item_index for product in products}) == 2
    assert {product.source_item_id for product in products} == {
        product.source_item_id for product in repeated
    }
