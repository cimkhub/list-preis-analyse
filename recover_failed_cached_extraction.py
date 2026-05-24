#!/usr/bin/env python3
"""Rerun cached PDF extractions that had page-level LLM failures and merge results safely."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.extract.cached_batch import (
    dedupe_documents,
    discover_cached_targets,
    get_scraper,
    load_downloaded_documents,
)
from src.models import RawProduct
from src.report.parsed_csv import (
    PARSED_CSV_FIELDS,
    find_existing_supplier_parsed_csv_path,
    get_combined_parsed_csv_path,
    get_supplier_parsed_csv_path,
)
from src.utils.logging_setup import get_run_id, log_event, setup_logging


FAILED_PAGE_RE = re.compile(
    r"Failed to analyze images/(?P<supplier>[^/]+)/(?P<year>\d+)/KW(?P<week>\d+)/(?P<stem>.+?)/page-[^ ]+ after 3 attempts"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only the cached PDFs whose page extraction failed in a previous log, "
            "then replace just those source files inside parsed CSV outputs."
        )
    )
    parser.add_argument("--week", type=int, required=True, help="Calendar week to repair")
    parser.add_argument("--year", type=int, required=True, help="Year to repair")
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to the previous extraction .log file to inspect for failed pages",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=25,
        help="Concurrent Vision calls per repaired PDF. Default: 25",
    )
    parser.add_argument(
        "--supplier",
        nargs="+",
        choices=["metro", "edeka", "selgros", "handelshof"],
        help="Optionally restrict repair to specific suppliers",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    return parser


def parse_failed_document_keys(log_file: Path, week: int, year: int) -> list[tuple[str, int, int, str]]:
    keys: list[tuple[str, int, int, str]] = []
    seen: set[tuple[str, int, int, str]] = set()

    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FAILED_PAGE_RE.search(line)
        if not match:
            continue

        key = (
            match.group("supplier"),
            int(match.group("week")),
            int(match.group("year")),
            match.group("stem"),
        )
        if key[1] != week or key[2] != year or key in seen:
            continue

        seen.add(key)
        keys.append(key)

    return keys


def build_document_index(config, selected_suppliers: list[str], week: int, year: int):
    targets = discover_cached_targets(
        config.storage.data_dir,
        selected_suppliers,
        week=week,
        year=year,
    )

    documents = []
    for target in targets:
        supplier_config = config.suppliers[target.supplier].model_dump()
        documents.extend(load_downloaded_documents(target, supplier_config))

    index = {}
    for document in dedupe_documents(documents):
        if not document.file_path:
            continue
        key = (document.supplier, document.calendar_week, document.year, Path(document.file_path).stem)
        index[key] = document
    return index


def serialize_product(product: RawProduct) -> dict[str, str]:
    row = product.model_dump(mode="json")
    row["price_tiers"] = (
        json.dumps(row["price_tiers"], ensure_ascii=False)
        if row.get("price_tiers") is not None
        else ""
    )
    return row


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def sort_key(row: dict[str, str]):
        source_page = row.get("source_page") or ""
        try:
            page_num = int(source_page)
        except ValueError:
            page_num = 0
        return (
            row.get("supplier", ""),
            row.get("source_file", ""),
            page_num,
            row.get("product_name", ""),
            row.get("price", ""),
        )

    return sorted(rows, key=sort_key)


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARSED_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, "") for field in PARSED_CSV_FIELDS}
            writer.writerow(normalized)
    tmp_path.replace(path)


def rebuild_combined_csv(base_dir: str, week: int, year: int, suppliers: list[str]) -> tuple[Path, int]:
    combined_rows: list[dict[str, str]] = []
    for supplier in suppliers:
        supplier_path = find_existing_supplier_parsed_csv_path(supplier, week, year, base_dir)
        combined_rows.extend(load_csv_rows(supplier_path))

    combined_rows = sort_rows(combined_rows)
    combined_path = get_combined_parsed_csv_path(week, year, base_dir)
    write_csv_rows(combined_path, combined_rows)
    return combined_path, len(combined_rows)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    logger = setup_logging(config.storage.logs_dir)
    logger.info("=" * 60)

    if not config.vision_api.api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Extraction requires the Vision API.")

    if args.workers < 1:
        raise RuntimeError("--workers must be at least 1")

    log_file = Path(args.log_file).resolve()
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    selected_suppliers = [
        supplier_name
        for supplier_name, supplier_config in config.suppliers.items()
        if supplier_config.enabled and (not args.supplier or supplier_name in args.supplier)
    ]

    failed_keys = parse_failed_document_keys(log_file, args.week, args.year)
    if args.supplier:
        failed_keys = [key for key in failed_keys if key[0] in args.supplier]
    if not failed_keys:
        logger.info("No failed cached PDFs found in %s for KW%02d %d", log_file, args.week, args.year)
        return

    from src.extract.vision import configure_gemini

    configure_gemini(
        config.vision_api.api_key,
        model_name=config.vision_api.model,
        max_retries=config.vision_api.max_retries,
        temperature=config.vision_api.temperature,
        max_concurrent_requests=args.workers,
        min_request_interval_seconds=config.vision_api.min_request_interval_seconds,
    )

    document_index = build_document_index(config, selected_suppliers, args.week, args.year)
    missing_documents = [key for key in failed_keys if key not in document_index]
    if missing_documents:
        raise RuntimeError(f"Could not locate cached PDFs for: {missing_documents}")

    supplier_configs = {
        supplier_name: config.suppliers[supplier_name].model_dump()
        for supplier_name in selected_suppliers
    }
    documents = [document_index[key] for key in failed_keys]

    log_event(
        logger,
        "Cached PDF repair run started",
        event="cached_pdf_repair",
        status="start",
        run_id=get_run_id(),
        week=args.week,
        year=args.year,
        log_file=str(log_file),
        worker_count=args.workers,
        document_count=len(documents),
        suppliers=selected_suppliers,
    )

    replaced_source_files_by_group: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    new_products_by_group: dict[tuple[str, int, int], list[RawProduct]] = defaultdict(list)

    for index, document in enumerate(documents, start=1):
        if not document.file_path:
            continue

        document_name = Path(document.file_path).name
        logger.info(
            "Repairing %d/%d: %s [%s]",
            index,
            len(documents),
            document_name,
            document.supplier,
        )
        scraper = get_scraper(document.supplier, supplier_configs[document.supplier])
        products = scraper.extract_products(document)
        group_key = (document.supplier, document.calendar_week, document.year)
        replaced_source_files_by_group[group_key].add(document.file_path)
        new_products_by_group[group_key].extend(products)
        logger.info(
            "Recovered %d products from %s [%s, %s -> %s]",
            len(products),
            document_name,
            document.location,
            document.valid_from,
            document.valid_to,
        )

    for group_key, source_files in sorted(replaced_source_files_by_group.items()):
        supplier, week, year = group_key
        supplier_path = find_existing_supplier_parsed_csv_path(supplier, week, year, config.storage.parsed_dir)
        existing_rows = load_csv_rows(supplier_path)
        kept_rows = [row for row in existing_rows if row.get("source_file") not in source_files]
        new_rows = [serialize_product(product) for product in new_products_by_group[group_key]]
        merged_rows = sort_rows(kept_rows + new_rows)
        output_path = get_supplier_parsed_csv_path(supplier, week, year, config.storage.parsed_dir)
        write_csv_rows(output_path, merged_rows)
        logger.info(
            "Updated %s with %d replaced source file(s), %d kept rows, %d repaired rows",
            output_path,
            len(source_files),
            len(kept_rows),
            len(new_rows),
        )

    combined_path, combined_count = rebuild_combined_csv(
        config.storage.parsed_dir,
        args.week,
        args.year,
        selected_suppliers,
    )
    logger.info("Rebuilt combined CSV: %s (%d rows)", combined_path, combined_count)

    log_event(
        logger,
        "Cached PDF repair run completed",
        event="cached_pdf_repair",
        status="ok",
        run_id=get_run_id(),
        week=args.week,
        year=args.year,
        log_file=str(log_file),
        repaired_document_count=len(documents),
        repaired_group_count=len(replaced_source_files_by_group),
        combined_output_path=str(combined_path),
        combined_row_count=combined_count,
    )


if __name__ == "__main__":
    main()
