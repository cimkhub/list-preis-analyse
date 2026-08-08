#!/usr/bin/env python3
"""Birkenhof Pipeline - Automated food wholesaler price comparison."""

import argparse
import csv
import json
import logging
import os
import shlex
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


DEEPSEEK_MODEL_ALIASES = {
    "pro": "deepseek-v4-pro",
    "v4-pro": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
}

ADDITIONAL_ONEDRIVE_URL = (
    "https://listgs-my.sharepoint.com/:f:/g/personal/l_kornblum_list-goslar_com/"
    "IgDnGmx_nK7IRY7o9R4SHweyAcIUGjZTepVdKaP8sCrSZzY?e=UC9vdF"
)


def resolve_deepseek_model(value: str | None) -> tuple[str, str]:
    requested = (value or "pro").strip()
    key = requested.casefold()
    if key not in DEEPSEEK_MODEL_ALIASES:
        raise ValueError("This pipeline is locked to the DeepSeek Pro model (deepseek-v4-pro).")
    return "pro", DEEPSEEK_MODEL_ALIASES[key]


def run_logged_subprocess(
    command: list[str],
    *,
    logger: logging.Logger,
    label: str,
    env: dict[str, str] | None = None,
) -> None:
    logger.info("%s command: %s", label, shlex.join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("%s | %s", label, line.rstrip())
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


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


def load_products_from_parsed_csv(csv_path: Path) -> list[RawProduct]:
    products: list[RawProduct] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
            products.append(RawProduct(**row))
    return products


def run_pipeline(week: int | None = None, year: int | None = None,
                 suppliers: list[str] | None = None,
                 skip_scrape: bool = False,
                 use_cache: bool = False,
                 scrape_only: bool = False,
                 skip_market_forecast: bool = False,
                 skip_onedrive_upload: bool = False,
                 onedrive_url: str | None = None,
                 additional_onedrive_url: str | None = ADDITIONAL_ONEDRIVE_URL,
                 onedrive_login_pause: bool = False,
                 deepseek_model: str = "pro"):
    config = load_config()

    if week is None or year is None:
        week, year = next_week()

    logger = setup_logging(config.storage.logs_dir, week=week, year=year)
    logger.info("=" * 60)
    deepseek_profile, deepseek_model_id = resolve_deepseek_model(deepseek_model)
    deepseek_env = os.environ.copy()
    deepseek_env["DEEPSEEK_MODEL"] = deepseek_model_id
    deepseek_env["DEEPSEEK_SIGNAL_MODEL"] = deepseek_model_id
    logger.info(
        "DeepSeek model selection: %s -> %s",
        deepseek_profile,
        deepseek_model_id,
    )

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
            min_request_interval_seconds=config.vision_api.min_request_interval_seconds,
        )
    elif not scrape_only:
        logger.warning("No GEMINI_API_KEY set - Vision API extraction will fail")

    all_products: list[RawProduct] = []
    total_documents_found = 0
    final_status = "failed"
    final_extra = None
    onedrive_upload_status = "not run"
    matched: dict = {}
    unmatched: list[RawProduct] = []
    report_path: Path | None = None
    legacy_report_status = "not run"

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
        deepseek_model=deepseek_model_id,
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
                if use_cache and not scrape_only:
                    cached_csv_path = find_existing_supplier_parsed_csv_path(
                        supplier_name,
                        week,
                        year,
                        config.storage.parsed_dir,
                    )
                    if cached_csv_path.exists():
                        cached_products = load_products_from_parsed_csv(cached_csv_path)
                        all_products.extend(cached_products)
                        logger.info(
                            "%s: Loaded %d products from existing parsed CSV %s; skipping PDF extraction",
                            supplier_name,
                            len(cached_products),
                            cached_csv_path,
                        )
                        log_event(
                            logger,
                            f"{supplier_name} parsed CSV cache loaded",
                            event="supplier_cache",
                            status="ok",
                            supplier=supplier_name,
                            product_count=len(cached_products),
                            input_path=str(cached_csv_path),
                        )
                        continue

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
                        failed_documents = []
                        for doc in documents:
                            doc_name = Path(doc.file_path).name if doc.file_path else (doc.title or "unknown")
                            try:
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
                            except Exception as e:
                                failed_documents.append(doc_name)
                                logger.error(
                                    "%s: Skipping failed document %s - %s",
                                    supplier_name,
                                    doc_name,
                                    e,
                                    exc_info=True,
                                )
                                log_event(
                                    logger,
                                    f"{supplier_name} document skipped after extraction failure",
                                    event="document_extract",
                                    status="skipped",
                                    supplier=supplier_name,
                                    document=doc_name,
                                    source_file=doc.file_path,
                                    title=doc.title,
                                    tab=doc.tab,
                                    error_type=type(e).__name__,
                                    error_message=str(e),
                                    exc_info=True,
                                )
                                continue

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

                        if failed_documents:
                            logger.warning(
                                "%s: Skipped %d failed documents and continued: %s",
                                supplier_name,
                                len(failed_documents),
                                ", ".join(failed_documents),
                            )
                            log_event(
                                logger,
                                f"{supplier_name} document extraction completed with skipped documents",
                                event="supplier_pipeline",
                                status="partial",
                                supplier=supplier_name,
                                skipped_document_count=len(failed_documents),
                                skipped_documents=failed_documents,
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
                        all_products.extend(load_products_from_parsed_csv(csv_path))
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
            from split_brand_product import (
                DEFAULT_DEEPSEEK_MODEL as TRANSLATION_DEEPSEEK_MODEL,
                run_brand_product_split,
            )

            relevant_csv_path, relevant_yes_count, relevant_no_count = run_relevance_classification(
                input_path=combined_csv_path,
                output_path=combined_csv_path.with_name("all_suppliers_relevant.csv"),
                deepseek_model=deepseek_model_id,
            )
            logger.info(
                "Product-name translation model: %s (thinking disabled)",
                TRANSLATION_DEEPSEEK_MODEL,
            )
            relevant_csv_path, split_count = run_brand_product_split(
                input_path=relevant_csv_path,
                workers=25,
                deepseek_model=TRANSLATION_DEEPSEEK_MODEL,
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
            run_logged_subprocess(
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
                    "--deepseek-model",
                    deepseek_model_id,
                ],
                logger=logger,
                label="competitor_matching",
                env=deepseek_env,
            )
        logger.info("Saved competitor matching output to %s", matched_competitor_path)

        # Stage 4c: Search-based market forecast signals.
        # This runs once per generated market group (e.g. "Rindfleisch"), not once per
        # individual product row, combines Brave and Tavily evidence, and then writes
        # the same signal back to all matching rows.
        if not skip_market_forecast:
            logger.info("--- MARKET FORECAST SIGNALS ---")
            forecast_cache_dir = matched_competitor_path.parent / "market_forecast_cache"
            try:
                with log_stage(
                    logger,
                    "Market forecast signal enrichment",
                    event="market_forecast",
                    workbook=str(matched_competitor_path),
                    cache_dir=str(forecast_cache_dir),
                ):
                    run_logged_subprocess(
                        [
                            sys.executable,
                            str(Path(__file__).parent / "add_market_forecast_signals.py"),
                            "--workbook",
                            str(matched_competitor_path),
                            "--cache-dir",
                            str(forecast_cache_dir),
                            "--deepseek-model",
                            deepseek_model_id,
                            "--deepseek-signal-model",
                            deepseek_model_id,
                        ],
                        logger=logger,
                        label="market_forecast",
                        env=deepseek_env,
                    )
                logger.info("Added market forecast signals to %s", matched_competitor_path)
            except Exception as exc:
                logger.warning("Market forecast signal enrichment skipped/failed: %s", exc, exc_info=True)
        else:
            logger.info("Market forecast signal enrichment skipped by CLI flag")

        try:
            from openpyxl import load_workbook
            from add_market_forecast_signals import finalize_customer_workbook

            workbook = load_workbook(matched_competitor_path)
            final_sheet = finalize_customer_workbook(workbook, matched_competitor_path)
            workbook.save(matched_competitor_path)
            logger.info("Final customer workbook reduced to one sheet: %s", final_sheet)
        except Exception as exc:
            logger.warning("Final customer workbook cleanup failed: %s", exc, exc_info=True)

        # Stage 4d: Upload the final customer workbook to the shared LIST OneDrive folders.
        # This uses the browser UI because the client folder is accessible in the browser,
        # but not reliably through Microsoft Graph shared-link APIs.
        if not skip_onedrive_upload:
            logger.info("--- ONEDRIVE UPLOAD ---")
            upload_targets: list[tuple[str, str | None]] = [
                ("primary", onedrive_url),
            ]
            if additional_onedrive_url:
                upload_targets.append(("additional", additional_onedrive_url))

            upload_statuses: list[str] = []
            for target_label, target_url in upload_targets:
                upload_command = [
                    sys.executable,
                    str(Path(__file__).parent / "upload_to_onedrive_browser.py"),
                    "--file",
                    str(matched_competitor_path),
                    "--filename",
                    matched_competitor_path.name,
                    "--timeout",
                    "240",
                ]
                if target_url:
                    upload_command.extend(["--target-url", target_url])
                if not onedrive_login_pause:
                    upload_command.append("--no-login-pause")

                try:
                    with log_stage(
                        logger,
                        f"OneDrive browser upload ({target_label})",
                        event="onedrive_upload",
                        target=target_label,
                        target_url=target_url or "default",
                        workbook=str(matched_competitor_path),
                        filename=matched_competitor_path.name,
                    ):
                        run_logged_subprocess(
                            upload_command,
                            logger=logger,
                            label=f"onedrive_upload_{target_label}",
                        )
                    upload_statuses.append(f"{target_label}: uploaded")
                    logger.info(
                        "Uploaded final customer workbook to OneDrive (%s): %s",
                        target_label,
                        matched_competitor_path.name,
                    )
                except Exception as exc:
                    upload_statuses.append(f"{target_label}: failed: {exc}")
                    logger.warning("OneDrive upload failed (%s): %s", target_label, exc, exc_info=True)
                    log_event(
                        logger,
                        "OneDrive upload failed",
                        event="onedrive_upload",
                        status="error",
                        target=target_label,
                        target_url=target_url or "default",
                        workbook=str(matched_competitor_path),
                        filename=matched_competitor_path.name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        exc_info=True,
                    )
            onedrive_upload_status = "; ".join(upload_statuses)
        else:
            onedrive_upload_status = "skipped"
            logger.info("OneDrive upload skipped by CLI flag")

        # Stage 5: Legacy harmonization/report.
        # The LIST customer workbook above is the primary weekly output. This older
        # canonical report is useful only when its reference file is available, so it
        # must not make an otherwise successful run fail on a fresh server.
        logger.info("--- HARMONIZATION ---")
        ref_path = Path(config.storage.reference_dir) / "canonical_products.csv"
        if not ref_path.exists():
            legacy_report_status = f"skipped; missing {ref_path}"
            logger.warning("Harmonization skipped: missing canonical reference file %s", ref_path)
        else:
            try:
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
                legacy_report_status = f"written: {report_path}"
            except Exception as exc:
                legacy_report_status = f"failed: {exc}"
                logger.warning("Legacy harmonization/report skipped after failure: %s", exc, exc_info=True)

        logger.info("Pipeline complete! Customer workbook: %s", matched_competitor_path)
        if report_path:
            logger.info("Legacy report: %s", report_path)
        logger.info(f"  Total products extracted: {len(all_products)}")
        logger.info(f"  Matched: {len(matched)}")
        logger.info(f"  Unmatched: {len(unmatched)}")
        final_status = "completed"
        legacy_report_label = Path(report_path).name if report_path else legacy_report_status
        final_extra = (
            f"Done. Documents found: {total_documents_found}. "
            f"Products: {len(all_products)}. Legacy report: {legacy_report_label}. "
            f"OneDrive: {onedrive_upload_status}."
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
            report_path=str(report_path) if report_path else "",
            legacy_report_status=legacy_report_status,
            customer_workbook_path=str(matched_competitor_path),
            onedrive_upload_status=onedrive_upload_status,
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
    parser.add_argument("--skip-market-forecast", action="store_true",
                        help="Skip search-based market forecast enrichment of the final Artikelvergleich workbook")
    parser.add_argument("--skip-onedrive-upload", action="store_true",
                        help="Skip browser-based upload of the final Artikelvergleich workbook to OneDrive")
    parser.add_argument("--onedrive-url",
                        help="Override the shared OneDrive folder URL used by the browser uploader")
    parser.add_argument("--additional-onedrive-url",
                        default=ADDITIONAL_ONEDRIVE_URL,
                        help="Additional shared OneDrive/SharePoint folder URL for the final workbook upload")
    parser.add_argument("--onedrive-login-pause", action="store_true",
                        help="Pause browser upload so you can manually complete Microsoft login/email-code verification")
    parser.add_argument("--deepseek-model", default="pro", choices=sorted(DEEPSEEK_MODEL_ALIASES),
                        help=(
                            "DeepSeek Pro model profile. The pipeline is locked to deepseek-v4-pro."
                        ))
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
        skip_market_forecast=args.skip_market_forecast,
        skip_onedrive_upload=args.skip_onedrive_upload,
        onedrive_url=args.onedrive_url,
        additional_onedrive_url=args.additional_onedrive_url,
        onedrive_login_pause=args.onedrive_login_pause,
        deepseek_model=args.deepseek_model,
    )


if __name__ == "__main__":
    main()
