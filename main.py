#!/usr/bin/env python3
"""Birkenhof Pipeline - Automated food wholesaler price comparison."""

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.extract.pdf_relevance import filter_relevant_documents
from src.models import RawProduct
from src.notify.twilio_sms import build_pipeline_sms_text, send_twilio_sms
from src.utils.logging_setup import get_run_id, log_event, log_stage, setup_logging
from src.utils.week import next_week, format_week
from src.harmonize.matcher import (
    load_canonical_products, match_all_products, build_comparison
)
from src.report.parsed_csv import (
    find_existing_supplier_parsed_csv_path,
    get_combined_parsed_csv_path,
    get_supplier_parsed_csv_path,
    save_combined_csv,
    save_parsed_csv,
)
from src.report.excel_report import generate_report


def get_scraper(name: str, config: dict):
    if name == "metro":
        from src.acquire.metro import MetroScraper
        return MetroScraper(config)
    elif name == "edeka":
        from src.acquire.edeka import EdekaScraper
        return EdekaScraper(config)
    elif name == "selgros":
        from src.acquire.selgros import SelgrosScraper
        return SelgrosScraper(config)
    elif name == "handelshof":
        from src.acquire.handelshof import HandelshofScraper
        return HandelshofScraper(config)
    else:
        raise ValueError(f"Unknown supplier: {name}")

def run_pipeline(week: int | None = None, year: int | None = None,
                 suppliers: list[str] | None = None,
                 skip_scrape: bool = False,
                 use_cache: bool = False,
                 scrape_only: bool = False):
    config = load_config()
    logger = setup_logging(config.storage.logs_dir)
    logger.info("=" * 60)

    if week is None or year is None:
        week, year = next_week()
    logger.info(f"Running pipeline for {format_week(week, year)}")
    selected_suppliers = [
        supplier_name
        for supplier_name, supplier_config in config.suppliers.items()
        if supplier_config.enabled and (not suppliers or supplier_name in suppliers)
    ]

    # Configure Vision API
    if not scrape_only and config.vision_api.api_key:
        from src.extract.vision import configure_gemini
        configure_gemini(
            config.vision_api.api_key,
            model_name=config.vision_api.model,
            max_retries=config.vision_api.max_retries,
            temperature=config.vision_api.temperature,
            max_concurrent_requests=config.vision_api.max_concurrent_requests,
        )
    elif not scrape_only:
        logger.warning("No GEMINI_API_KEY set - Vision API extraction will fail")

    all_products: list[RawProduct] = []
    total_documents_found = 0
    final_status = "failed"
    final_extra = None

    log_event(
        logger,
        "Pipeline run started",
        event="pipeline_run",
        status="start",
        run_id=get_run_id(),
        week=week,
        year=year,
        suppliers=selected_suppliers,
        skip_scrape=skip_scrape,
        use_cache=use_cache,
        scrape_only=scrape_only,
    )

    should_send_completion_sms = (
        not skip_scrape
        and set(selected_suppliers) == {"metro", "edeka", "selgros", "handelshof"}
    )

    try:
        if not skip_scrape:
            # Stage 1: acquire/download all selected supplier documents first.
            acquired_documents = {}
            scrapers = {}
            logger.info("--- ACQUISITION ---")
            for supplier_name in selected_suppliers:
                supplier_config = config.suppliers[supplier_name]
                logger.info(f"--- Acquiring {supplier_name.upper()} ---")
                try:
                    scraper = get_scraper(supplier_name, supplier_config.model_dump())
                    scrapers[supplier_name] = scraper
                    with log_stage(
                        logger,
                        f"{supplier_name} acquisition",
                        event="supplier_acquire",
                        supplier=supplier_name,
                        week=week,
                        year=year,
                    ):
                        documents = scraper.get_current_offers(week, year, force=not use_cache)
                    acquired_documents[supplier_name] = documents
                    total_documents_found += len(documents)
                    logger.info(f"{supplier_name}: Found {len(documents)} documents")
                    log_event(
                        logger,
                        f"{supplier_name} acquisition completed",
                        event="supplier_acquire",
                        status="ok",
                        supplier=supplier_name,
                        document_count=len(documents),
                    )
                except Exception as e:
                    logger.error(f"{supplier_name}: Acquisition failed - {e}", exc_info=True)
                    log_event(
                        logger,
                        f"{supplier_name} acquisition failed",
                        event="supplier_acquire",
                        status="error",
                        supplier=supplier_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        exc_info=True,
                    )

            if scrape_only:
                logger.info("Scrape-only mode active, skipping relevance filter and extraction")
            else:
                # Stage 2-3: relevance filter, then extract only relevant documents.
                logger.info("--- RELEVANCE + EXTRACTION ---")
                for supplier_name, documents in acquired_documents.items():
                    logger.info(f"--- Processing {supplier_name.upper()} ---")
                    scraper = scrapers[supplier_name]

                    try:
                        if not documents:
                            logger.warning(f"{supplier_name}: No documents acquired")
                            continue

                        # The relevance filter sends the PDF title and first-page visual to Gemini,
                        # extracts visible validity dates, skips Non-Food/Outdoor material,
                        # and skips byte-identical PDFs already seen in the previous ISO week.
                        with log_stage(
                            logger,
                            f"{supplier_name} relevance filter",
                            event="document_relevance",
                            supplier=supplier_name,
                            document_count=len(documents),
                        ):
                            documents, skipped_documents = filter_relevant_documents(
                                documents,
                                supplier=supplier_name,
                                images_base_dir=config.storage.images_dir,
                            )
                        logger.info(
                            f"{supplier_name}: {len(documents)} relevant documents, "
                            f"{len(skipped_documents)} skipped after relevance check"
                        )
                        log_event(
                            logger,
                            f"{supplier_name} relevance filter completed",
                            event="document_relevance",
                            status="ok",
                            supplier=supplier_name,
                            relevant_count=len(documents),
                            skipped_count=len(skipped_documents),
                        )
                        for skipped in skipped_documents:
                            skipped_name = Path(skipped.file_path).name if skipped.file_path else (skipped.title or "unknown")
                            logger.info(
                                f"{supplier_name}: Skipping {skipped_name} "
                                f"[{skipped.relevance_label}] {skipped.relevance_reason}"
                            )

                        supplier_products = []
                        for doc in documents:
                            doc_name = Path(doc.file_path).name if doc.file_path else (doc.title or "unknown")
                            with log_stage(
                                logger,
                                f"{supplier_name} document extraction",
                                event="document_extract",
                                supplier=supplier_name,
                                document=doc_name,
                                source_file=doc.file_path,
                                title=doc.title,
                                tab=doc.tab,
                            ):
                                products = scraper.extract_products(doc)
                            supplier_products.extend(products)
                            log_event(
                                logger,
                                f"{supplier_name} document extracted",
                                event="document_extract",
                                status="ok",
                                supplier=supplier_name,
                                document=doc_name,
                                product_count=len(products),
                            )

                        if supplier_products:
                            save_parsed_csv(supplier_products, supplier_name, week, year,
                                            config.storage.parsed_dir)
                            csv_path = get_supplier_parsed_csv_path(
                                supplier_name,
                                week,
                                year,
                                config.storage.parsed_dir,
                            )
                            all_products.extend(supplier_products)
                            logger.info(
                                "%s: Extracted %d products to %s",
                                supplier_name,
                                len(supplier_products),
                                csv_path,
                            )
                        else:
                            logger.warning(f"{supplier_name}: No products extracted")

                    except Exception as e:
                        logger.error(f"{supplier_name}: Failed - {e}", exc_info=True)
                        log_event(
                            logger,
                            f"{supplier_name} supplier pipeline failed",
                            event="supplier_pipeline",
                            status="error",
                            supplier=supplier_name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            exc_info=True,
                        )
        else:
            # Load from existing parsed CSVs without scraping/relevance/extraction.
            for supplier_name in selected_suppliers:
                try:
                    csv_path = find_existing_supplier_parsed_csv_path(
                        supplier_name,
                        week,
                        year,
                        config.storage.parsed_dir,
                    )
                    if csv_path.exists():
                        with open(csv_path, encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                # Convert types
                                row["price"] = float(row["price"]) if row["price"] else 0
                                row["extraction_confidence"] = float(row.get("extraction_confidence", 0.8))
                                row["price_is_net"] = row.get("price_is_net", "False") == "True"
                                row["price_tiers"] = (
                                    json.loads(row["price_tiers"])
                                    if row.get("price_tiers") and row["price_tiers"] != "None"
                                    else None
                                )
                                for field in ["price_per_kg", "price_gross", "quantity"]:
                                    row[field] = float(row[field]) if row.get(field) and row[field] != "None" else None
                                for field in ["calendar_week", "year", "source_page"]:
                                    row[field] = int(row[field]) if row.get(field) and row[field] != "None" else None
                                for field in ["valid_from", "valid_to"]:
                                    row[field] = row[field] if row.get(field) and row[field] != "None" else None
                                all_products.append(RawProduct(**row))
                        logger.info(f"Loaded {supplier_name} from existing CSV")
                except Exception as e:
                    logger.error(f"{supplier_name}: Failed loading cache - {e}", exc_info=True)
                    log_event(
                        logger,
                        f"{supplier_name} cached load failed",
                        event="supplier_pipeline",
                        status="error",
                        supplier=supplier_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        exc_info=True,
                    )

        if scrape_only:
            logger.info("Scrape-only run complete")
            logger.info(f"  Total documents found: {total_documents_found}")
            final_status = "completed"
            final_extra = f"Scrape-only run done. Documents found: {total_documents_found}."
            log_event(
                logger,
                "Scrape-only run completed",
                event="pipeline_run",
                status="ok",
                run_id=get_run_id(),
                week=week,
                year=year,
                total_documents_found=total_documents_found,
                total_products=0,
                note=final_extra,
            )
            return

        if not all_products:
            logger.error("No products extracted from any supplier. Aborting.")
            final_extra = f"No products extracted. Documents found: {total_documents_found}."
            return

        save_combined_csv(all_products, week, year, config.storage.parsed_dir)
        combined_csv_path = get_combined_parsed_csv_path(week, year, config.storage.parsed_dir)
        logger.info(
            "Saved %d extracted products to %s",
            len(all_products),
            combined_csv_path,
        )

        # Stage 4: Product-level relevance classification
        logger.info("--- PRODUCT RELEVANCE ---")
        with log_stage(
            logger,
            "Product relevance classification",
            event="product_relevance",
            input_path=str(combined_csv_path),
            product_count=len(all_products),
        ):
            from classify_fresh_food_relevance import run_relevance_classification
            from extract_single_cached_pdf import write_xlsx
            from split_brand_product import run_brand_product_split

            relevant_csv_path, relevant_yes_count, relevant_no_count = run_relevance_classification(
                input_path=combined_csv_path,
                output_path=combined_csv_path.with_name("all_suppliers_relevant.csv"),
            )
            relevant_csv_path, split_count = run_brand_product_split(
                input_path=relevant_csv_path,
                workers=25,
            )
            relevant_xlsx_path = write_xlsx(relevant_csv_path)
        logger.info(
            "Saved product relevance outputs to %s and %s (Ja=%d, Nein=%d, brand/product split=%d)",
            relevant_csv_path,
            relevant_xlsx_path,
            relevant_yes_count,
            relevant_no_count,
            split_count,
        )

        # Stage 4b: Competitor product matching for the final LIST Excel output.
        logger.info("--- COMPETITOR PRODUCT MATCHING ---")
        matched_competitor_path = combined_csv_path.with_name(f"Artikelvergleich KW{week:02d}.xlsx")
        with log_stage(
            logger,
            "Competitor product matching",
            event="competitor_product_matching",
            input_path=str(relevant_csv_path),
            output_path=str(matched_competitor_path),
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "match_competitor_products.py"),
                    "--input",
                    str(relevant_csv_path),
                    "--output",
                    str(matched_competitor_path),
                    "--embedding-dir",
                    "embeddings/product_matching",
                    "--pair-workers",
                    "25",
                ],
                check=True,
            )
        logger.info("Saved competitor matching output to %s", matched_competitor_path)

        # Stage 5: Harmonize
        logger.info("--- HARMONIZATION ---")
        ref_path = Path(config.storage.reference_dir) / "canonical_products.csv"
        with log_stage(
            logger,
            "Harmonization",
            event="harmonize",
            product_count=len(all_products),
            canonical_path=str(ref_path),
        ):
            canonicals = load_canonical_products(str(ref_path))

            matched, unmatched = match_all_products(
                all_products, canonicals, config.pipeline.fuzzy_match_threshold
            )

            comparison = build_comparison(matched, canonicals)

        # Stage 6: Report
        logger.info("--- REPORT GENERATION ---")
        with log_stage(
            logger,
            "Excel report generation",
            event="report_generation",
            matched_count=len(matched),
            unmatched_count=len(unmatched),
            product_count=len(all_products),
        ):
            report_path = generate_report(
                comparison, unmatched, all_products, week, year,
                config.storage.reports_dir,
            )

        logger.info(f"Pipeline complete! Report: {report_path}")
        logger.info(f"  Total products extracted: {len(all_products)}")
        logger.info(f"  Matched: {len(matched)}")
        logger.info(f"  Unmatched: {len(unmatched)}")
        final_status = "completed"
        final_extra = (
            f"Done. Documents found: {total_documents_found}. "
            f"Products: {len(all_products)}. Report: {Path(report_path).name}."
        )
        log_event(
            logger,
            "Pipeline run completed",
            event="pipeline_run",
            status="ok",
            run_id=get_run_id(),
            week=week,
            year=year,
            total_documents_found=total_documents_found,
            total_products=len(all_products),
            matched_count=len(matched),
            unmatched_count=len(unmatched),
            report_path=str(report_path),
        )
    finally:
        if final_status != "completed":
            log_event(
                logger,
                "Pipeline run finished without success",
                event="pipeline_run",
                level=logging.WARNING,
                status=final_status,
                run_id=get_run_id(),
                week=week,
                year=year,
                total_documents_found=total_documents_found,
                total_products=len(all_products),
                note=final_extra or "Run ended before report generation.",
            )
        if should_send_completion_sms:
            send_twilio_sms(
                config.twilio,
                build_pipeline_sms_text(
                    stage=f"pipeline {final_status}",
                    week=week,
                    year=year,
                    suppliers=selected_suppliers,
                    extra=final_extra or "Run finished.",
                ),
                logger=logger,
            )


def main():
    parser = argparse.ArgumentParser(description="Birkenhof Price Comparison Pipeline")
    parser.add_argument("--week", type=int, help="Calendar week (default: next week)")
    parser.add_argument("--year", type=int, help="Year (default: current)")
    parser.add_argument("--supplier", type=str, nargs="+",
                        choices=["metro", "edeka", "selgros", "handelshof"],
                        help="Process only specific suppliers")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping, use existing parsed CSVs")
    parser.add_argument("--cache", action="store_true",
                        help="Use cached data if available (default: always scrape fresh)")
    parser.add_argument("--scrape-only", action="store_true",
                        help="Run only supplier scraping/downloads without relevance filtering or PDF extraction")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    run_pipeline(
        week=args.week,
        year=args.year,
        suppliers=args.supplier,
        skip_scrape=args.skip_scrape,
        use_cache=args.cache,
        scrape_only=args.scrape_only,
    )


if __name__ == "__main__":
    main()
