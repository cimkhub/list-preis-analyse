import csv
import json
import logging
from pathlib import Path

from src.models import RawProduct
from src.utils.logging_setup import log_event, log_stage

PARSED_CSV_FIELDS = [
    "supplier",
    "location",
    "product_name",
    "description",
    "category",
    "origin",
    "unit",
    "quantity",
    "price",
    "price_per_kg",
    "price_is_net",
    "price_gross",
    "price_tiers",
    "valid_from",
    "valid_to",
    "calendar_week",
    "year",
    "source_file",
    "source_title",
    "source_tab",
    "source_page",
    "extraction_confidence",
    "product_family",
    "temperature_state",
    "processing_state",
    "calibre",
    "source_brand",
    "brand_evidence",
    "brand_evidence_source",
    "price_basis",
    "package_count",
    "package_size_value",
    "package_size_unit",
    "total_content_value",
    "total_content_unit",
    "packaging_type",
    "packaging_raw",
    "certifications",
    "source_item_index",
    "source_item_id",
    "source_document_sha256",
    "primary_extraction_model",
    "selected_extraction_model",
    "quality_retry_attempted",
    "quality_retry_status",
    "quality_retry_model",
    "quality_retry_issues",
    "extraction_schema_version",
]

_JSON_FIELDS = ("price_tiers", "certifications", "quality_retry_issues")
_FLOAT_FIELDS = (
    "price",
    "price_per_kg",
    "price_gross",
    "quantity",
    "package_size_value",
    "total_content_value",
    "extraction_confidence",
)
_INTEGER_FIELDS = (
    "calendar_week",
    "year",
    "source_page",
    "package_count",
    "source_item_index",
    "extraction_schema_version",
)
_BOOLEAN_FIELDS = ("price_is_net", "quality_retry_attempted")
_OPTIONAL_DATE_FIELDS = ("valid_from", "valid_to")
_EMPTY_VALUES = {"", "none", "null", "nan"}


def get_parsed_batch_dir(week: int, year: int, base_dir: str = "parsed") -> Path:
    return Path(base_dir) / f"KW{week:02d}_{year}"


def get_supplier_parsed_csv_path(
    supplier: str,
    week: int,
    year: int,
    base_dir: str = "parsed",
) -> Path:
    return get_parsed_batch_dir(week, year, base_dir) / f"{supplier}.csv"


def get_combined_parsed_csv_path(
    week: int,
    year: int,
    base_dir: str = "parsed",
) -> Path:
    return get_parsed_batch_dir(week, year, base_dir) / "all_suppliers.csv"


def find_existing_supplier_parsed_csv_path(
    supplier: str,
    week: int,
    year: int,
    base_dir: str = "parsed",
) -> Path:
    canonical_path = get_supplier_parsed_csv_path(supplier, week, year, base_dir)
    if canonical_path.exists():
        return canonical_path

    legacy_path = Path(base_dir) / supplier / f"KW{week:02d}_{year}.csv"
    return legacy_path


def find_existing_combined_parsed_csv_path(
    week: int,
    year: int,
    base_dir: str = "parsed",
) -> Path:
    canonical_path = get_combined_parsed_csv_path(week, year, base_dir)
    if canonical_path.exists():
        return canonical_path

    legacy_path = Path(base_dir) / f"KW{week:02d}_{year}_all_suppliers.csv"
    return legacy_path


def serialize_product_row(product: RawProduct | dict) -> dict[str, object]:
    """Return one CSV-safe row while keeping the historical column order stable."""
    row = product.model_dump(mode="json") if isinstance(product, RawProduct) else dict(product)
    for field in _JSON_FIELDS:
        value = row.get(field)
        if field == "price_tiers" and value is None:
            row[field] = ""
        else:
            default = None if field == "price_tiers" else []
            parsed = _parse_json_field(value, default)
            row[field] = (
                ""
                if field == "price_tiers" and parsed is None
                else json.dumps(parsed or [], ensure_ascii=False)
            )
    return {field: row.get(field, "") for field in PARSED_CSV_FIELDS}


def _is_empty(value) -> bool:
    return value is None or str(value).strip().casefold() in _EMPTY_VALUES


def _parse_json_field(value, default):
    if _is_empty(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return parsed


def _parse_bool(value, default: bool = False) -> bool:
    if _is_empty(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "ja", "y"}


def deserialize_product_row(row: dict[str, object]) -> RawProduct:
    """Load current or historical parsed-CSV rows into ``RawProduct``."""
    values = {
        key: value
        for key, value in dict(row).items()
        if key in RawProduct.model_fields
    }

    values["price_tiers"] = _parse_json_field(values.get("price_tiers"), None)
    values["certifications"] = _parse_json_field(values.get("certifications"), [])
    values["quality_retry_issues"] = _parse_json_field(
        values.get("quality_retry_issues"), []
    )

    for field in _FLOAT_FIELDS:
        raw_value = values.get(field)
        if _is_empty(raw_value):
            if field == "price":
                values[field] = 0.0
            elif field == "extraction_confidence":
                values[field] = 0.8
            else:
                values[field] = None
            continue
        try:
            values[field] = float(str(raw_value).replace(",", "."))
        except (TypeError, ValueError):
            if field == "price":
                values[field] = 0.0
            elif field == "extraction_confidence":
                values[field] = 0.8
            else:
                values[field] = None

    for field in _INTEGER_FIELDS:
        raw_value = values.get(field)
        if _is_empty(raw_value):
            if field == "extraction_schema_version":
                values[field] = 1
            else:
                values[field] = None
            continue
        try:
            values[field] = int(float(str(raw_value).replace(",", ".")))
        except (TypeError, ValueError):
            values[field] = 1 if field == "extraction_schema_version" else None

    for field in _BOOLEAN_FIELDS:
        values[field] = _parse_bool(values.get(field), default=False)

    for field in _OPTIONAL_DATE_FIELDS:
        if _is_empty(values.get(field)):
            values[field] = None

    if _is_empty(values.get("quality_retry_status")):
        values["quality_retry_status"] = "not_needed"

    return RawProduct(**values)


# Backward-compatible private name used by older imports/tests.
_serialize_product = serialize_product_row


def save_parsed_csv(
    products: list[RawProduct],
    supplier: str,
    week: int,
    year: int,
    base_dir: str = "parsed",
):
    filepath = get_supplier_parsed_csv_path(supplier, week, year, base_dir)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("birkenhof")

    with log_stage(
        logger,
        "Save parsed CSV",
        event="save_parsed_csv",
        supplier=supplier,
        output_path=str(filepath),
        product_count=len(products),
    ):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PARSED_CSV_FIELDS)
            writer.writeheader()
            for product in products:
                writer.writerow(serialize_product_row(product))

    log_event(
        logger,
        f"Saved {len(products)} products to {filepath}",
        event="save_parsed_csv",
        status="ok",
        supplier=supplier,
        output_path=str(filepath),
        product_count=len(products),
    )


def save_combined_csv(
    products: list[RawProduct],
    week: int,
    year: int,
    base_dir: str = "parsed",
):
    filepath = get_combined_parsed_csv_path(week, year, base_dir)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("birkenhof")

    with log_stage(
        logger,
        "Save combined parsed CSV",
        event="save_combined_csv",
        output_path=str(filepath),
        product_count=len(products),
    ):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PARSED_CSV_FIELDS)
            writer.writeheader()
            for product in products:
                writer.writerow(serialize_product_row(product))

    log_event(
        logger,
        f"Saved {len(products)} products to {filepath}",
        event="save_combined_csv",
        status="ok",
        output_path=str(filepath),
        product_count=len(products),
    )
