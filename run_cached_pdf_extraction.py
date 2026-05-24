#!/usr/bin/env python3
"""Extract products from already-downloaded PDF brochures without matching."""

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.extract.cached_batch import (
    dedupe_documents,
    discover_cached_targets,
    get_scraper,
    load_downloaded_documents,
)
from src.report.parsed_csv import (
    get_combined_parsed_csv_path,
    get_supplier_parsed_csv_path,
    save_combined_csv,
    save_parsed_csv,
)
from src.utils.logging_setup import get_run_id, log_event, setup_logging


def _extract_document(document, supplier_config: dict):
    scraper = get_scraper(document.supplier, supplier_config)
    return scraper.extract_products(document)


def main():
    parser = argparse.ArgumentParser(
        description="Extract products from already-downloaded PDF brochures only"
    )
    parser.add_argument("--week", type=int, help="Only use cached documents from this week")
    parser.add_argument("--year", type=int, help="Only use cached documents from this year")
    parser.add_argument(
        "--supplier",
        type=str,
        nargs="+",
        choices=["metro", "edeka", "selgros", "handelshof"],
        help="Process only specific suppliers",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel extraction workers. Values below 10 are raised to 10.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        help="Process only the cached PDF with this basename",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    if (args.week is None) != (args.year is None):
        raise SystemExit("Use --week and --year together, or neither.")

    config = load_config(args.config)
    logger = setup_logging(config.storage.logs_dir)
    logger.info("=" * 60)

    if not config.vision_api.api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Extraction requires the Vision API.")

    workers = max(10, args.workers)
    selected_suppliers = [
        supplier_name
        for supplier_name, supplier_config in config.suppliers.items()
        if supplier_config.enabled and (not args.supplier or supplier_name in args.supplier)
    ]

    from src.extract.vision import configure_gemini

    page_concurrency = 1
    if args.file_name:
        page_concurrency = workers

    # Normal batch mode keeps one Vision call per document worker to cap total load.
    # Single-file mode reuses the same worker budget inside that one PDF to finish it quickly.
    configure_gemini(
        config.vision_api.api_key,
        model_name=config.vision_api.model,
        max_retries=config.vision_api.max_retries,
        temperature=config.vision_api.temperature,
        max_concurrent_requests=page_concurrency,
        min_request_interval_seconds=config.vision_api.min_request_interval_seconds,
    )

    log_event(
        logger,
        "Cached PDF extraction run started",
        event="cached_pdf_extract",
        status="start",
        run_id=get_run_id(),
        suppliers=selected_suppliers,
        week=args.week,
        year=args.year,
        worker_count=workers,
        llm_concurrency_per_worker=page_concurrency,
    )

    targets = discover_cached_targets(
        config.storage.data_dir,
        selected_suppliers,
        week=args.week,
        year=args.year,
    )
    if not targets:
        logger.warning("No cached PDF targets found")
        return

    logger.info("Found %d cached supplier/week targets", len(targets))

    documents = []
    for target in targets:
        supplier_config = config.suppliers[target.supplier].model_dump()
        target_docs = load_downloaded_documents(target, supplier_config)
        logger.info(
            "Loaded %d cached PDFs for %s from %s",
            len(target_docs),
            target.supplier,
            target.data_dir,
        )
        documents.extend(target_docs)

    documents = dedupe_documents(documents)
    if args.file_name:
        documents = [
            document
            for document in documents
            if document.file_path and Path(document.file_path).name == args.file_name
        ]
        logger.info(
            "Filtered cached documents to %d file(s) matching %s",
            len(documents),
            args.file_name,
        )
    if not documents:
        logger.warning("No cached PDF documents available after deduplication")
        return

    logger.info(
        "Starting extraction for %d unique PDFs with %d parallel workers",
        len(documents),
        workers,
    )

    all_products = []
    supplier_configs = {
        supplier_name: config.suppliers[supplier_name].model_dump()
        for supplier_name in selected_suppliers
    }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _extract_document,
                document,
                supplier_configs[document.supplier],
            ): document
            for document in documents
        }

        for future in as_completed(future_map):
            document = future_map[future]
            document_name = Path(document.file_path).name if document.file_path else (document.title or "unknown")
            try:
                products = future.result()
            except Exception as exc:
                logger.error(
                    "Extraction failed for %s [%s]: %s",
                    document_name,
                    document.supplier,
                    exc,
                    exc_info=True,
                )
                continue

            all_products.extend(products)
            logger.info(
                "Extracted %d products from %s [%s, %s, %s -> %s]",
                len(products),
                document_name,
                document.supplier,
                document.location,
                document.valid_from,
                document.valid_to,
            )

    grouped_products = defaultdict(list)
    combined_products = defaultdict(list)
    for product in all_products:
        if product.calendar_week is None or product.year is None:
            logger.warning(
                "Skipping product without calendar grouping: %s [%s]",
                product.product_name,
                product.source_file,
            )
            continue
        grouped_products[(product.supplier, product.calendar_week, product.year)].append(product)
        combined_products[(product.calendar_week, product.year)].append(product)

    for (supplier, week, year), products in sorted(grouped_products.items()):
        save_parsed_csv(products, supplier, week, year, config.storage.parsed_dir)
        output_path = get_supplier_parsed_csv_path(supplier, week, year, config.storage.parsed_dir)
        logger.info(
            "Saved %d extracted products to %s",
            len(products),
            output_path,
        )

    for (week, year), products in sorted(combined_products.items()):
        save_combined_csv(products, week, year, config.storage.parsed_dir)
        output_path = get_combined_parsed_csv_path(week, year, config.storage.parsed_dir)
        logger.info(
            "Saved %d extracted products to %s",
            len(products),
            output_path,
        )

    logger.info(
        "Cached extraction complete. PDFs: %d, products: %d, supplier CSVs: %d, combined CSVs: %d",
        len(documents),
        len(all_products),
        len(grouped_products),
        len(combined_products),
    )
    log_event(
        logger,
        "Cached PDF extraction run completed",
        event="cached_pdf_extract",
        status="ok",
        run_id=get_run_id(),
        pdf_count=len(documents),
        product_count=len(all_products),
        output_file_count=len(grouped_products) + len(combined_products),
        supplier_output_file_count=len(grouped_products),
        combined_output_file_count=len(combined_products),
        worker_count=workers,
        llm_concurrency_per_worker=page_concurrency,
    )


if __name__ == "__main__":
    main()
