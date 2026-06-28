import csv
import json
import logging
from pathlib import Path

from src.harmonize.customer_rules import apply_customer_category_overrides_to_product
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
]


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


def _serialize_product(product: RawProduct) -> dict:
    product = apply_customer_category_overrides_to_product(product)
    row = product.model_dump(mode="json")
    row["price_tiers"] = (
        json.dumps(row["price_tiers"], ensure_ascii=False)
        if row.get("price_tiers") is not None
        else ""
    )
    return row


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
                writer.writerow(_serialize_product(product))

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
                writer.writerow(_serialize_product(product))

    log_event(
        logger,
        f"Saved {len(products)} products to {filepath}",
        event="save_combined_csv",
        status="ok",
        output_path=str(filepath),
        product_count=len(products),
    )
