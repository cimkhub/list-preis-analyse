#!/usr/bin/env python3
"""Download Handelshof PDFs for a demo run and store them in the usual data folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.utils.logging_setup import log_event, log_stage, setup_logging
from src.utils.week import current_week, format_week, week_dir


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Handelshof PDF download demo and save files to the standard data folder."
    )
    parser.add_argument("--week", type=int, help="Calendar week to use (default: current week)")
    parser.add_argument("--year", type=int, help="Year to use (default: current year)")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse cached files if they already exist instead of forcing a fresh download",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    logger = setup_logging(config.storage.logs_dir)

    week, year = (args.week, args.year) if args.week and args.year else current_week()
    supplier_config = config.suppliers.get("handelshof")
    if supplier_config is None:
        raise RuntimeError("Handelshof configuration not found in config.yaml")

    from src.acquire.handelshof import HandelshofScraper

    scraper = HandelshofScraper(supplier_config.model_dump())
    data_dir = Path(scraper.storage_base) / week_dir("handelshof", week, year)

    log_event(
        logger,
        "Handelshof demo started",
        event="handelshof_demo",
        status="start",
        week=week,
        year=year,
        use_cache=args.cache,
        output_dir=str(data_dir),
    )

    try:
        with log_stage(
            logger,
            "Handelshof demo acquisition",
            event="handelshof_demo",
            supplier="handelshof",
            week=week,
            year=year,
            use_cache=args.cache,
            output_dir=str(data_dir),
        ):
            documents = scraper.get_current_offers(week, year, force=not args.cache)
    except Exception:
        raise

    pdf_paths = sorted(str(path) for path in data_dir.glob("*.pdf"))
    xlsx_paths = sorted(str(path) for path in data_dir.glob("*.xlsx"))

    print("\n" + "=" * 80)
    print("HANDELSHOF DEMO COMPLETE")
    print("=" * 80)
    print(f"Week: {format_week(week, year)}")
    print(f"Output folder: {data_dir.resolve()}")
    print(f"Selected downstream PDF documents: {len(documents)}")
    print(f"Saved PDF files in folder: {len(pdf_paths)}")
    print(f"Saved XLSX files in folder: {len(xlsx_paths)}")

    if pdf_paths:
        print("\nPDF FILES")
        for pdf_path in pdf_paths:
            print(f"- {pdf_path}")
    else:
        print("\nNo PDF files were saved.")

    if xlsx_paths:
        print("\nXLSX FILES")
        for xlsx_path in xlsx_paths:
            print(f"- {xlsx_path}")

    raw_brochures = data_dir / "raw_brochures.json"
    if raw_brochures.exists():
        print(f"\nMetadata: {raw_brochures}")

    log_event(
        logger,
        "Handelshof demo finished",
        event="handelshof_demo",
        status="ok",
        week=week,
        year=year,
        selected_documents=len(documents),
        saved_pdf_files=len(pdf_paths),
        saved_xlsx_files=len(xlsx_paths),
        output_dir=str(data_dir),
    )


if __name__ == "__main__":
    main()
