import json
import logging
import hashlib
from datetime import date, timedelta
from pathlib import Path

from src.convert.pdf_to_images import pdf_first_page_to_image
from src.extract.prompts import RELEVANCE_SYSTEM_PROMPT, get_relevance_prompt
from src.extract.vision import analyze_image_json
from src.models import AcquiredDocument

logger = logging.getLogger("birkenhof.extract.pdf_relevance")

RELEVANCE_CACHE_SCHEMA_VERSION = 2

RELEVANT_FILENAME_KEYWORDS = {
    "aktuell",
    "angebot",
    "angebote",
    "basis",
    "food",
    "fisch",
    "fleisch",
    "frisch",
    "gastro",
    "gv",
    "gross",
    "medium",
    "metro",
    "nord",
    "mitte",
    "sued",
    "sud",
    "wochen",
}

IRRELEVANT_FILENAME_KEYWORDS = {
    "bestellkatalog",
    "eiskonzept",
    "gastromoebel",
    "katalog",
    "lifestyle",
    "living",
    "magazin",
    "menu",
    "mix",
    "moebel",
    "non food",
    "non-food",
    "nonfood",
    "outdoor",
    "saisonkatalog",
    "tabletop",
    "textilien",
    "tischgesprach",
}

IRRELEVANT_LABELS = {
    "irrelevant_non_food_only",
    "irrelevant_catalog_or_order_guide",
    "irrelevant_marketing_or_magazine",
    "heuristic_irrelevant_filename",
    "duplicate_previous_week",
    "irrelevant_market_scope",
}

TARGET_MARKETS = {
    "hannover",
    "goslar",
    "hildesheim",
    "wernigerode",
    "braunschweig",
}


def filter_relevant_documents(
    documents: list[AcquiredDocument],
    supplier: str,
    images_base_dir: str = "images",
    skip_previous_week_duplicates: bool = True,
) -> tuple[list[AcquiredDocument], list[AcquiredDocument]]:
    if not documents:
        return [], []

    decisions_path = _relevance_decisions_path(documents)
    cached = _load_cached_decisions(decisions_path)

    for document in documents:
        if document.doc_type != "pdf" or not document.file_path:
            _apply_decision(document, {
                "is_relevant": True,
                "relevance_label": "non_pdf_passthrough",
                "relevance_reason": "Dokument ist kein PDF und wird nicht durch den Relevanzfilter blockiert.",
                "relevance_confidence": 1.0,
            })
            continue

        cache_key = str(Path(document.file_path).resolve())
        if cache_key in cached:
            _apply_decision(document, cached[cache_key])
            continue

        if skip_previous_week_duplicates:
            duplicate_path = _find_previous_week_duplicate(document)
            if duplicate_path:
                decision = {
                    "is_relevant": False,
                    "relevance_label": "duplicate_previous_week",
                    "relevance_reason": (
                        "PDF ist byte-identisch mit einem Dokument aus der Vorwoche: "
                        f"{duplicate_path.name}"
                    ),
                    "relevance_confidence": 1.0,
                    "duplicate_of": str(duplicate_path),
                }
                _apply_decision(document, decision)
                cached[cache_key] = _serialize_decision(document, extra=decision)
                continue

        decision = classify_document_relevance(
            document=document,
            supplier=supplier,
            images_base_dir=images_base_dir,
        )
        _apply_decision(document, decision)
        cached[cache_key] = _serialize_decision(document)

    if decisions_path:
        _save_cached_decisions(decisions_path, documents)

    relevant = [doc for doc in documents if doc.is_relevant is not False]
    skipped = [doc for doc in documents if doc.is_relevant is False]
    return relevant, skipped


def classify_document_relevance(
    document: AcquiredDocument,
    supplier: str,
    images_base_dir: str = "images",
) -> dict:
    if not document.file_path:
        return {
            "is_relevant": True,
            "relevance_label": "missing_file_passthrough",
            "relevance_reason": "Kein Dateipfad vorhanden, deshalb kein Filter.",
            "relevance_confidence": 0.0,
        }

    file_path = Path(document.file_path)
    filename = file_path.name
    hints = _filename_hints(filename, document.title)

    try:
        preview_dir = Path(images_base_dir) / "_relevance" / supplier / file_path.stem
        preview_path = pdf_first_page_to_image(str(file_path), str(preview_dir))
        result = analyze_image_json(
            image_path=preview_path,
            prompt=get_relevance_prompt(
                supplier=supplier,
                filename=filename,
                title=document.title,
                tab=document.tab,
                relevant_hits=hints["relevant_hits"],
                irrelevant_hits=hints["irrelevant_hits"],
            ),
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            operation="document_relevance",
            supplier=supplier,
            source_file=str(file_path),
            source_filename=filename,
            title=document.title,
            tab=document.tab,
        )
        return _normalize_decision(result, document, hints)
    except Exception as e:
        logger.warning(f"Relevance classification failed for {filename}: {e}")
        return _fallback_decision(document, hints, error=str(e))


def _filename_hints(filename: str, title: str | None) -> dict:
    combined = f"{filename} {title or ''}".lower().replace("_", " ").replace("-", " ")
    relevant_hits = sorted(
        keyword for keyword in RELEVANT_FILENAME_KEYWORDS
        if keyword in combined
    )
    irrelevant_hits = sorted(
        keyword for keyword in IRRELEVANT_FILENAME_KEYWORDS
        if keyword in combined
    )
    return {
        "relevant_hits": relevant_hits,
        "irrelevant_hits": irrelevant_hits,
    }


def _normalize_decision(result, document: AcquiredDocument, hints: dict) -> dict:
    if not isinstance(result, dict):
        return _fallback_decision(document, hints, error="model returned no JSON object")

    label = str(result.get("relevance_label") or "unclear").strip()
    reason = str(result.get("reason") or "").strip()
    confidence = _coerce_confidence(result.get("confidence"))
    is_relevant = result.get("is_relevant")
    valid_from = _parse_iso_date(result.get("valid_from"))
    valid_to = _parse_iso_date(result.get("valid_to"))
    market_scope = _normalize_market_scope(result.get("market_scope"))
    valid_markets = _normalize_markets(result.get("valid_markets"))

    if isinstance(is_relevant, str):
        is_relevant = is_relevant.lower() == "true"
    if is_relevant is None:
        is_relevant = label not in IRRELEVANT_LABELS

    if label == "unclear":
        if hints["irrelevant_hits"] and not hints["relevant_hits"]:
            label = "heuristic_irrelevant_filename"
            is_relevant = False
            if not reason:
                reason = "Dateiname deutet stark auf irrelevanten Katalog/Non-Food-Inhalt hin."
        else:
            label = "unclear_kept"
            is_relevant = True
            if not reason:
                reason = "Modell war unsicher, Dokument bleibt vorsorglich im Prozess."

    if is_relevant and market_scope == "specific" and not _contains_target_market(valid_markets):
        label = "irrelevant_market_scope"
        is_relevant = False
        markets = ", ".join(valid_markets) if valid_markets else "keine Zielmärkte erkannt"
        reason = (
            "Erste Seite nennt teilnehmende Märkte, aber keinen Zielmarkt "
            f"(Hannover, Goslar, Hildesheim, Wernigerode, Braunschweig): {markets}"
        )

    if valid_from:
        document.valid_from = valid_from
    if valid_to:
        document.valid_to = valid_to

    return {
        "is_relevant": bool(is_relevant),
        "relevance_label": label,
        "relevance_reason": reason or "Keine Begründung vom Modell zurückgegeben.",
        "relevance_confidence": confidence,
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_to": valid_to.isoformat() if valid_to else None,
        "market_scope": market_scope,
        "valid_markets": valid_markets,
    }


def _fallback_decision(document: AcquiredDocument, hints: dict, error: str | None = None) -> dict:
    if hints["irrelevant_hits"] and not hints["relevant_hits"]:
        return {
            "is_relevant": False,
            "relevance_label": "heuristic_irrelevant_filename",
            "relevance_reason": (
                "Dateiname enthält starke Irrelevanzsignale"
                + (f" ({error})" if error else "")
            ),
            "relevance_confidence": 0.6,
        }

    return {
        "is_relevant": True,
        "relevance_label": "heuristic_keep_unclassified",
        "relevance_reason": (
            "Relevanzmodell fehlgeschlagen oder unsicher, Dokument bleibt im Prozess"
            + (f" ({error})" if error else "")
        ),
        "relevance_confidence": 0.2,
    }


def _coerce_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _apply_decision(document: AcquiredDocument, decision: dict):
    document.is_relevant = bool(decision.get("is_relevant", True))
    document.relevance_label = decision.get("relevance_label")
    document.relevance_reason = decision.get("relevance_reason")
    document.relevance_confidence = _coerce_confidence(decision.get("relevance_confidence"))
    document.market_scope = _normalize_market_scope(decision.get("market_scope"))
    document.valid_markets = _normalize_markets(decision.get("valid_markets"))
    valid_from = _parse_iso_date(decision.get("valid_from"))
    valid_to = _parse_iso_date(decision.get("valid_to"))
    if valid_from:
        document.valid_from = valid_from
    if valid_to:
        document.valid_to = valid_to


def _normalize_market_scope(value) -> str:
    text = str(value or "").strip().lower()
    if text in {"all", "specific", "unknown"}:
        return text
    return "unknown"


def _normalize_markets(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        return []
    return [item for item in raw_items if item]


def _contains_target_market(markets: list[str]) -> bool:
    for market in markets:
        normalized = market.casefold()
        if any(target in normalized for target in TARGET_MARKETS):
            return True
    return False


def _parse_iso_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _relevance_decisions_path(documents: list[AcquiredDocument]) -> Path | None:
    for document in documents:
        if document.file_path:
            return Path(document.file_path).resolve().parent / "relevance_decisions.json"
    return None


def _load_cached_decisions(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}

    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not read relevance cache {path}: {e}")
        return {}

    decisions = {}
    for item in items:
        if item.get("schema_version") != RELEVANCE_CACHE_SCHEMA_VERSION:
            continue
        file_path = item.get("file_path")
        if not file_path:
            continue
        decisions[file_path] = item
    return decisions


def _save_cached_decisions(path: Path | None, documents: list[AcquiredDocument]):
    if not path:
        return

    serializable = []
    for document in documents:
        serializable.append(_serialize_decision(document))

    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _serialize_decision(document: AcquiredDocument, extra: dict | None = None) -> dict:
    resolved_path = (
        str(Path(document.file_path).resolve())
        if document.file_path
        else None
    )
    item = {
        "file_path": resolved_path,
        "schema_version": RELEVANCE_CACHE_SCHEMA_VERSION,
        "filename": Path(document.file_path).name if document.file_path else None,
        "title": document.title,
        "tab": document.tab,
        "is_relevant": document.is_relevant,
        "relevance_label": document.relevance_label,
        "relevance_reason": document.relevance_reason,
        "relevance_confidence": document.relevance_confidence,
        "valid_from": document.valid_from.isoformat() if document.valid_from else None,
        "valid_to": document.valid_to.isoformat() if document.valid_to else None,
        "market_scope": document.market_scope,
        "valid_markets": document.valid_markets,
    }
    if extra:
        for key in ["duplicate_of"]:
            if extra.get(key):
                item[key] = extra[key]
    return item


def _find_previous_week_duplicate(document: AcquiredDocument) -> Path | None:
    if not document.file_path:
        return None

    current_path = Path(document.file_path).resolve()
    if not current_path.exists():
        return None

    supplier = document.supplier
    week = document.calendar_week
    year = document.year
    if week is None or year is None:
        inferred = _infer_week_from_path(current_path)
        if inferred:
            supplier, year, week = inferred
    if week is None or year is None or not supplier:
        return None

    previous_monday = date.fromisocalendar(int(year), int(week), 1) - timedelta(days=7)
    previous_iso = previous_monday.isocalendar()
    data_root = _infer_data_root(current_path, supplier)
    previous_dir = data_root / supplier / str(previous_iso.year) / f"{previous_iso.week:02d}"
    if not previous_dir.exists():
        previous_dir = data_root / supplier / str(previous_iso.year) / str(previous_iso.week)
    if not previous_dir.exists():
        return None

    current_hash = _sha256_file(current_path)
    for candidate in sorted(previous_dir.glob("*.pdf")):
        try:
            if candidate.resolve() == current_path:
                continue
            if candidate.stat().st_size != current_path.stat().st_size:
                continue
            if _sha256_file(candidate) == current_hash:
                return candidate.resolve()
        except OSError:
            continue
    return None


def _infer_week_from_path(path: Path) -> tuple[str, int, int] | None:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "data" and idx + 3 < len(parts):
            supplier = parts[idx + 1]
            try:
                year = int(parts[idx + 2])
                week = int(parts[idx + 3])
            except ValueError:
                return None
            return supplier, year, week
    return None


def _infer_data_root(path: Path, supplier: str) -> Path:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "data" and idx + 1 < len(parts) and parts[idx + 1] == supplier:
            return Path(*parts[: idx + 1])
    return Path("data").resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
