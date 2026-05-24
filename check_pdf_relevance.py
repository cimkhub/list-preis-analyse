#!/usr/bin/env python3
"""Run the PDF relevance filter for already downloaded supplier PDFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.extract.pdf_relevance import filter_relevant_documents
from src.extract.vision import configure_gemini
from src.models import AcquiredDocument
from src.utils.week import next_week


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check downloaded PDFs for food-offer relevance using the PDF title "
            "and first-page Gemini visual analysis."
        )
    )
    parser.add_argument("--week", type=int, help="ISO calendar week. Defaults to next week.")
    parser.add_argument("--year", type=int, help="ISO year. Defaults to next week year.")
    parser.add_argument(
        "--supplier",
        action="append",
        dest="suppliers",
        help="Supplier to check. Can be repeated. Defaults to all enabled suppliers.",
    )
    parser.add_argument(
        "--no-previous-week-duplicate-skip",
        action="store_true",
        help="Do not skip PDFs that are byte-identical to the previous ISO week.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    week, year = (args.week, args.year) if args.week and args.year else next_week()

    if not config.vision_api.api_key:
        raise RuntimeError("GEMINI_API_KEY fehlt in .env oder in der Umgebung.")
    print(
        "Gemini config: "
        f"provider={config.vision_api.provider}, "
        f"model={config.vision_api.model}, "
        "endpoint=Google Gemini Developer API default via google.genai.Client(api_key=...)",
        flush=True,
    )
    configure_gemini(
        config.vision_api.api_key,
        model_name=config.vision_api.model,
        max_retries=config.vision_api.max_retries,
        temperature=config.vision_api.temperature,
        max_concurrent_requests=config.vision_api.max_concurrent_requests,
        min_request_interval_seconds=config.vision_api.min_request_interval_seconds,
    )

    selected_suppliers = [
        name
        for name, supplier_config in config.suppliers.items()
        if supplier_config.enabled and (not args.suppliers or name in args.suppliers)
    ]

    all_decisions = []
    for supplier in selected_suppliers:
        supplier_config = config.suppliers[supplier]
        pdf_dir = Path(config.storage.data_dir) / supplier / str(year) / f"{week:02d}"
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        print(f"{supplier}: checking {len(pdfs)} PDFs in {pdf_dir}", flush=True)
        documents = [
            AcquiredDocument(
                supplier=supplier,
                location=supplier_config.location,
                doc_type="pdf",
                file_path=str(pdf),
                title=pdf.stem,
                calendar_week=week,
                year=year,
            )
            for pdf in pdfs
        ]
        relevant, skipped = filter_relevant_documents(
            documents,
            supplier=supplier,
            images_base_dir=config.storage.images_dir,
            skip_previous_week_duplicates=not args.no_previous_week_duplicate_skip,
        )
        print(f"{supplier}: {len(relevant)} relevant, {len(skipped)} skipped", flush=True)
        all_decisions.extend(_decision_row(document) for document in documents)

    output_path = Path(config.storage.data_dir) / f"pdf_relevance_KW{week:02d}_{year}.json"
    output_path.write_text(
        json.dumps(all_decisions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved relevance decisions to {output_path}", flush=True)


def _decision_row(document: AcquiredDocument) -> dict:
    return {
        "supplier": document.supplier,
        "location": document.location,
        "file_path": document.file_path,
        "title": document.title,
        "is_relevant": document.is_relevant,
        "relevance_label": document.relevance_label,
        "relevance_reason": document.relevance_reason,
        "relevance_confidence": document.relevance_confidence,
        "valid_from": document.valid_from.isoformat() if document.valid_from else None,
        "valid_to": document.valid_to.isoformat() if document.valid_to else None,
    }


if __name__ == "__main__":
    main()
