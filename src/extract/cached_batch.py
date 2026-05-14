import json
import logging
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.models import AcquiredDocument

logger = logging.getLogger("birkenhof.extract.cached_batch")


@dataclass(frozen=True)
class CachedExtractionTarget:
    supplier: str
    week: int
    year: int
    data_dir: Path


def get_scraper(name: str, config: dict):
    if name == "metro":
        from src.acquire.metro import MetroScraper
        return MetroScraper(config)
    if name == "edeka":
        from src.acquire.edeka import EdekaScraper
        return EdekaScraper(config)
    if name == "selgros":
        from src.acquire.selgros import SelgrosScraper
        return SelgrosScraper(config)
    if name == "handelshof":
        from src.acquire.handelshof import HandelshofScraper
        return HandelshofScraper(config)
    raise ValueError(f"Unknown supplier: {name}")


def discover_cached_targets(
    data_root: str | Path,
    supplier_names: list[str],
    *,
    week: int | None = None,
    year: int | None = None,
) -> list[CachedExtractionTarget]:
    root = Path(data_root)
    targets: list[CachedExtractionTarget] = []

    for supplier in supplier_names:
        supplier_root = root / supplier
        if not supplier_root.exists():
            continue

        for year_dir in sorted(supplier_root.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            target_year = int(year_dir.name)
            if year is not None and target_year != year:
                continue

            for week_dir in sorted(year_dir.iterdir()):
                if not week_dir.is_dir() or not week_dir.name.isdigit():
                    continue
                target_week = int(week_dir.name)
                if week is not None and target_week != week:
                    continue

                if _has_cached_assets(week_dir):
                    targets.append(
                        CachedExtractionTarget(
                            supplier=supplier,
                            week=target_week,
                            year=target_year,
                            data_dir=week_dir,
                        )
                    )

    return sorted(targets, key=lambda item: (item.year, item.week, item.supplier))


def load_downloaded_documents(
    target: CachedExtractionTarget,
    supplier_config: dict,
) -> list[AcquiredDocument]:
    if target.supplier == "metro":
        return _load_metro_documents(target, supplier_config)

    raw_file = target.data_dir / "raw_brochures.json"
    if raw_file.exists():
        documents = _load_documents_from_raw_brochures(raw_file, target, supplier_config)
        if documents:
            return documents

    return _load_documents_from_pdf_files(target, supplier_config)


def dedupe_documents(documents: list[AcquiredDocument]) -> list[AcquiredDocument]:
    unique_documents: list[AcquiredDocument] = []
    seen_hashes: set[tuple[str, str]] = set()

    for document in documents:
        if not document.file_path:
            continue

        file_path = Path(document.file_path)
        if not file_path.exists():
            continue

        file_hash = _file_sha256(file_path)
        dedupe_key = (document.supplier, file_hash)
        if dedupe_key in seen_hashes:
            continue

        seen_hashes.add(dedupe_key)
        unique_documents.append(document)

    return unique_documents


def _has_cached_assets(data_dir: Path) -> bool:
    return (data_dir / "raw_brochures.json").exists() or any(data_dir.glob("*.pdf"))


def _load_documents_from_raw_brochures(
    raw_file: Path,
    target: CachedExtractionTarget,
    supplier_config: dict,
) -> list[AcquiredDocument]:
    try:
        items = json.loads(raw_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s: %s", raw_file, exc)
        return []

    category_key = {
        "edeka": "catalog_category_name",
        "selgros": "category_name",
        "handelshof": "catalog_category_name",
    }.get(target.supplier)

    documents: list[AcquiredDocument] = []
    seen_paths: set[str] = set()

    for item in items:
        pdf_path = item.get("pdf_local_path")
        if not pdf_path:
            continue

        file_path = Path(pdf_path)
        if not file_path.exists():
            continue

        resolved_path = str(file_path.resolve())
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)

        documents.append(
            AcquiredDocument(
                supplier=target.supplier,
                location=supplier_config.get("location", target.supplier),
                doc_type="pdf",
                file_path=str(file_path),
                url=item.get("viewer_url") or item.get("download_url"),
                title=item.get("title"),
                tab=item.get("tab"),
                valid_from=_parse_iso_date(item.get("valid_from")),
                valid_to=_parse_iso_date(item.get("valid_to")),
                category=item.get(category_key) if category_key else None,
                calendar_week=target.week,
                year=target.year,
            )
        )

    if documents:
        logger.info(
            "Loaded %d downloaded %s PDFs from %s",
            len(documents),
            target.supplier,
            raw_file,
        )
    return documents


def _load_documents_from_pdf_files(
    target: CachedExtractionTarget,
    supplier_config: dict,
) -> list[AcquiredDocument]:
    pdfs = sorted(target.data_dir.glob("*.pdf"))
    documents = [
        AcquiredDocument(
            supplier=target.supplier,
            location=supplier_config.get("location", target.supplier),
            doc_type="pdf",
            file_path=str(pdf),
            title=pdf.stem,
            calendar_week=target.week,
            year=target.year,
        )
        for pdf in pdfs
    ]
    if documents:
        logger.info(
            "Loaded %d fallback %s PDFs from %s",
            len(documents),
            target.supplier,
            target.data_dir,
        )
    return documents


def _load_metro_documents(
    target: CachedExtractionTarget,
    supplier_config: dict,
) -> list[AcquiredDocument]:
    scraper = get_scraper("metro", supplier_config)
    pdfs = scraper._dedupe_pdf_files(list(target.data_dir.glob("*.pdf")))
    documents: list[AcquiredDocument] = []

    for pdf in pdfs:
        meta = scraper._parse_filename_dates(pdf.name)
        documents.append(
            AcquiredDocument(
                supplier="metro",
                location=supplier_config.get("location", "goslar"),
                doc_type="pdf",
                file_path=str(pdf),
                title=pdf.stem,
                valid_from=meta.get("valid_from"),
                valid_to=meta.get("valid_to"),
                calendar_week=target.week,
                year=target.year,
            )
        )

    if documents:
        logger.info(
            "Loaded %d downloaded metro PDFs from %s",
            len(documents),
            target.data_dir,
        )
    return documents


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
