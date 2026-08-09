from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
import math
import random
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai.types import GenerateContentConfig, Part

from src.extract.prompts import SYSTEM_PROMPT, get_extraction_prompt
from src.models import (
    CONTENT_UNITS,
    EXTRACTION_SCHEMA_VERSION,
    PACKAGING_TYPES,
    PRICE_BASES,
    RawProduct,
)
from src.utils.logging_setup import log_event

logger = logging.getLogger("birkenhof.extract.vision")

_client = None
_settings = {
    "model_name": "gemini-2.5-flash",
    "max_retries": 3,
    "temperature": 0.1,
    "max_concurrent_requests": 10,
    "min_request_interval_seconds": 0.0,
}
QUALITY_RETRY_MODEL = "gemini-3.6-flash"
QUALITY_RETRY_CONFIDENCE_THRESHOLD = 0.7
_COMMON_PRODUCT_PUNCTUATION = set("-–—/&.,()[]+%:'\"°")
_NUMERIC_RAW_ITEM_FIELDS = {
    "quantity",
    "price",
    "price_per_kg",
    "price_gross",
    "min_qty",
    "max_qty",
    "package_count",
    "package_size_value",
    "total_content_value",
}
_BOOLEAN_RAW_ITEM_FIELDS = {"price_is_net"}
_CONTENT_UNIT_ALIASES = {
    "gramm": "g",
    "kilogramm": "kg",
    "liter": "l",
    "litre": "l",
    "stück": "piece",
    "stueck": "piece",
    "pcs": "piece",
}
_PACKAGING_TYPE_ALIASES = {
    "beutel": "bag",
    "tüte": "bag",
    "tuete": "bag",
    "packung": "pack",
    "paket": "pack",
    "karton": "box",
    "kiste": "crate",
    "korb": "basket",
    "schale": "tray",
    "eimer": "bucket",
    "flasche": "bottle",
    "dose": "can",
    "bund": "bundle",
    "stück": "piece",
    "stueck": "piece",
}
_PACKAGING_WORD_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>kg|g|ml|l)\s*[- ]*"
    r"(?:pro\s+)?(?P<pack>beutel|packung|karton|kiste|korb|schale|eimer|flasche|dose)\b",
    flags=re.IGNORECASE,
)
_MULTIPACK_PATTERN = re.compile(
    r"(?P<count>\d+)\s*(?:stück|stueck|stk\.?|pcs?)?\s*(?:x|×|à)\s*"
    r"(?P<size>\d+(?:[.,]\d+)?)\s*(?P<unit>kg|g|ml|l)\b",
    flags=re.IGNORECASE,
)
_CALIBRE_PER_LB_PATTERN = re.compile(
    r"(?P<low>\d+)\s*/\s*(?P<high>\d+)\s*(?:stück|stueck|pcs?)?\s*(?:pro|per|/)\s*lb\b",
    flags=re.IGNORECASE,
)
_rate_limit_lock = threading.Lock()
_last_request_started_at = 0.0


@dataclass(frozen=True)
class PageExtractionOutcome:
    """Auditable result of one primary extraction and at most one quality retry."""

    primary_items: list[dict] | None
    selected_items: list[dict]
    primary_model: str
    selected_model: str
    primary_failed: bool
    primary_quality_issues: tuple[str, ...]
    quality_retry_attempted: bool
    quality_retry_status: str
    quality_retry_model: str | None = None
    retry_items: list[dict] | None = None
    retry_quality_issues: tuple[str, ...] = ()

    def to_manifest_record(self, **context) -> dict[str, Any]:
        record = asdict(self)
        record.update(context)
        record["primary_item_count"] = len(self.primary_items or [])
        record["selected_item_count"] = len(self.selected_items)
        record["retry_item_count"] = len(self.retry_items or [])
        record["primary_items_sha256"] = _raw_items_sha256(self.primary_items)
        record["selected_items_sha256"] = _raw_items_sha256(self.selected_items)
        record["retry_items_sha256"] = _raw_items_sha256(self.retry_items)
        # Keep the manifest compact while retaining cryptographic evidence of
        # both candidates and the selected result.
        record.pop("primary_items", None)
        record.pop("selected_items", None)
        record.pop("retry_items", None)
        record["primary_quality_issues"] = list(self.primary_quality_issues)
        record["retry_quality_issues"] = list(self.retry_quality_issues)
        return record


def _raw_items_sha256(raw_items: list[dict] | None) -> str | None:
    if raw_items is None:
        return None
    payload = json.dumps(
        raw_items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configure_gemini(
    api_key: str,
    model_name: str | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    max_concurrent_requests: int | None = None,
    min_request_interval_seconds: float | None = None,
):
    global _client
    _client = genai.Client(api_key=api_key)
    if model_name is not None:
        _settings["model_name"] = model_name
    if max_retries is not None:
        _settings["max_retries"] = max_retries
    if temperature is not None:
        _settings["temperature"] = temperature
    if max_concurrent_requests is not None:
        _settings["max_concurrent_requests"] = max_concurrent_requests
    if min_request_interval_seconds is not None:
        _settings["min_request_interval_seconds"] = max(0.0, float(min_request_interval_seconds))


def _resolve_setting(name: str, value):
    return _settings[name] if value is None else value


def analyze_image_json(
    image_path: str,
    prompt: str,
    system_prompt: str,
    model_name: str | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    operation: str = "vision_json",
    **context,
):
    global _client
    if not _client:
        raise RuntimeError("Gemini not configured. Call configure_gemini() first.")

    model_name = _resolve_setting("model_name", model_name)
    max_retries = _resolve_setting("max_retries", max_retries)
    temperature = _resolve_setting("temperature", temperature)
    min_request_interval_seconds = _resolve_setting("min_request_interval_seconds", None)
    call_started = time.perf_counter()

    img_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"

    log_event(
        logger,
        f"LLM call started for {Path(image_path).name}",
        event="llm_call",
        status="start",
        operation=operation,
        model_name=model_name,
        image_path=str(image_path),
        max_retries=max_retries,
        **context,
    )

    for attempt in range(max_retries):
        attempt_started = time.perf_counter()
        try:
            _wait_for_gemini_rate_limit(min_request_interval_seconds)
            response = _client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    Part.from_bytes(data=img_bytes, mime_type=mime_type),
                ],
                config=_generation_config(model_name, system_prompt, temperature),
            )
            text = response.text.strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]

            parsed = json.loads(text)
            result_count = len(parsed) if isinstance(parsed, list) else 1
            usage = getattr(response, "usage_metadata", None)
            log_event(
                logger,
                f"LLM call completed for {Path(image_path).name}",
                event="llm_call",
                status="ok",
                operation=operation,
                model_name=model_name,
                image_path=str(image_path),
                attempt=attempt + 1,
                duration_ms=round((time.perf_counter() - attempt_started) * 1000, 2),
                total_duration_ms=round((time.perf_counter() - call_started) * 1000, 2),
                result_count=result_count,
                model_version=getattr(response, "model_version", None),
                prompt_token_count=getattr(usage, "prompt_token_count", None),
                candidates_token_count=getattr(usage, "candidates_token_count", None),
                total_token_count=getattr(usage, "total_token_count", None),
                **context,
            )
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
            log_event(
                logger,
                f"LLM JSON parse error for {Path(image_path).name}",
                event="llm_call",
                level=logging.WARNING,
                status="retry",
                operation=operation,
                model_name=model_name,
                image_path=str(image_path),
                attempt=attempt + 1,
                duration_ms=round((time.perf_counter() - attempt_started) * 1000, 2),
                error_type=type(e).__name__,
                error_message=str(e),
                **context,
            )
            if attempt < max_retries - 1:
                time.sleep(_retry_delay_seconds(attempt, e))
        except Exception as e:
            logger.warning(f"Vision API error on attempt {attempt + 1}: {e}")
            log_event(
                logger,
                f"LLM request error for {Path(image_path).name}",
                event="llm_call",
                level=logging.WARNING,
                status="retry",
                operation=operation,
                model_name=model_name,
                image_path=str(image_path),
                attempt=attempt + 1,
                duration_ms=round((time.perf_counter() - attempt_started) * 1000, 2),
                error_type=type(e).__name__,
                error_message=str(e),
                **context,
            )
            if attempt < max_retries - 1:
                time.sleep(_retry_delay_seconds(attempt, e))

    logger.error(f"Failed to analyze {image_path} after {max_retries} attempts")
    log_event(
        logger,
        f"LLM call failed for {Path(image_path).name}",
        event="llm_call",
        level=logging.ERROR,
        status="error",
        operation=operation,
        model_name=model_name,
        image_path=str(image_path),
        total_duration_ms=round((time.perf_counter() - call_started) * 1000, 2),
        attempts=max_retries,
        **context,
    )
    return None


def _generation_config(
    model_name: str,
    system_prompt: str,
    temperature: float | None,
) -> GenerateContentConfig:
    config: dict = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
    }
    # Gemini 3.5 Flash, 3.6 Flash, and later models deprecate sampling parameters.
    match = re.match(r"gemini-(\d+)\.(\d+)", (model_name or "").casefold())
    if not match or (int(match.group(1)), int(match.group(2))) < (3, 5):
        config["temperature"] = temperature
    return GenerateContentConfig(**config)


def _retry_delay_seconds(attempt: int, error: Exception) -> float:
    message = str(error).lower()
    if (
        "429" in message
        or "quota" in message
        or "rate" in message
        or "resource_exhausted" in message
        or "too many requests" in message
    ):
        return min(60 * (2 ** attempt), 300) + random.uniform(0, 10)
    if "503" in message or "unavailable" in message or "high demand" in message:
        return min(15 * (2 ** attempt), 90) + random.uniform(0, 3)
    return 2 ** attempt


def _wait_for_gemini_rate_limit(min_interval_seconds: float | None) -> None:
    global _last_request_started_at
    min_interval_seconds = float(min_interval_seconds or 0.0)
    if min_interval_seconds <= 0:
        return
    with _rate_limit_lock:
        now = time.monotonic()
        wait_seconds = (_last_request_started_at + min_interval_seconds) - now
        if wait_seconds > 0:
            logger.info("Gemini low-tier throttle: waiting %.1fs before next request", wait_seconds)
            time.sleep(wait_seconds)
        _last_request_started_at = time.monotonic()


def extract_products_from_image(
    image_path: str,
    supplier: str,
    model_name: str | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    operation: str = "product_extraction",
    quality_issues: list[str] | None = None,
) -> list[dict] | None:
    prompt = get_extraction_prompt(supplier)
    if quality_issues:
        prompt += (
            "\n\nDies ist eine einmalige Qualitäts-Nachextraktion. "
            "Extrahiere die komplette Seite neu und prüfe besonders Produktnamen und Preise. "
            f"Auffälligkeiten im ersten Ergebnis: {', '.join(quality_issues)}."
        )
    result = analyze_image_json(
        image_path=image_path,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        model_name=model_name,
        max_retries=max_retries,
        temperature=temperature,
        operation=operation,
        supplier=supplier,
    )
    if result is None:
        # Preserve the distinction between an API/JSON failure and a successful
        # JSON response containing an empty product array.
        return None

    products = result if isinstance(result, list) else [result]
    logger.debug(f"Extracted {len(products)} products from {Path(image_path).name}")
    return products


def extraction_quality_issues(raw_items: list[dict] | None) -> list[str]:
    """Return compact reasons that make one page worth exactly one quality retry."""
    if raw_items is None:
        return ["Extraktion/API fehlgeschlagen"]
    if not isinstance(raw_items, list):
        return ["ungültiges Seitenformat"]
    if not raw_items:
        return ["keine Produkte erkannt"]

    issues: list[str] = []
    for index, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            issues.append(f"Produkt {index}: ungültiges Format")
            continue

        product_name = str(item.get("product_name") or "").strip()
        if not product_name:
            issues.append(f"Produkt {index}: Name fehlt")
        elif _looks_garbled(product_name):
            issues.append(f"Produkt {index}: Name unleserlich")

        price = item.get("price")
        try:
            numeric_price = float(price)
        except (TypeError, ValueError):
            numeric_price = 0.0
        if not math.isfinite(numeric_price) or numeric_price <= 0:
            issues.append(f"Produkt {index}: Preis fehlt/ungültig")

        description = item.get("description")
        if description and _looks_garbled(str(description)):
            issues.append(f"Produkt {index}: Beschreibung unleserlich")

        confidence = item.get("confidence")
        if confidence not in (None, ""):
            try:
                if float(confidence) < QUALITY_RETRY_CONFIDENCE_THRESHOLD:
                    issues.append(f"Produkt {index}: geringe Sicherheit")
            except (TypeError, ValueError):
                issues.append(f"Produkt {index}: Sicherheit ungültig")

        issues.extend(_structured_packaging_issues(item, index))

    return list(dict.fromkeys(issues))


def _normalized_content_unit(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace(" ", "_")
    return _CONTENT_UNIT_ALIASES.get(normalized, normalized)


def _normalized_packaging_type(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    return _PACKAGING_TYPE_ALIASES.get(normalized, normalized)


def _positive_float(value: Any) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _content_base_value(value: Any, unit: Any) -> tuple[str, float] | None:
    number = _positive_float(value)
    normalized_unit = _normalized_content_unit(unit)
    if number is None:
        return None
    if normalized_unit == "kg":
        return ("mass", number * 1000)
    if normalized_unit == "g":
        return ("mass", number)
    if normalized_unit == "l":
        return ("volume", number * 1000)
    if normalized_unit == "ml":
        return ("volume", number)
    if normalized_unit == "piece":
        return ("piece", number)
    return None


def _same_content_amount(
    first_value: Any,
    first_unit: Any,
    second_value: Any,
    second_unit: Any,
    *,
    multiplier: float = 1.0,
) -> bool:
    first = _content_base_value(first_value, first_unit)
    second = _content_base_value(second_value, second_unit)
    if first is None or second is None or first[0] != second[0]:
        return False
    expected = first[1] * multiplier
    tolerance = max(abs(second[1]) * 0.03, 0.01)
    return abs(expected - second[1]) <= tolerance


def _structured_packaging_issues(item: dict[str, Any], index: int) -> list[str]:
    prefix = f"Produkt {index}: "
    issues: list[str] = []

    count_raw = item.get("package_count")
    count = _positive_float(count_raw)
    if count_raw not in (None, "", "None", "null") and (
        count is None or not count.is_integer()
    ):
        issues.append(prefix + "Packungsanzahl ungültig")

    size_value = item.get("package_size_value")
    size_unit = _normalized_content_unit(item.get("package_size_unit"))
    total_value = item.get("total_content_value")
    total_unit = _normalized_content_unit(item.get("total_content_unit"))
    packaging_type = _normalized_packaging_type(item.get("packaging_type"))
    price_basis = str(item.get("price_basis") or "unknown").strip().casefold().replace("-", "_")

    if _positive_float(size_value) is not None and size_unit not in CONTENT_UNITS - {"unknown"}:
        issues.append(prefix + "Einzelinhalt ohne gültige Maßeinheit")
    if _positive_float(total_value) is not None and total_unit not in CONTENT_UNITS - {"unknown"}:
        issues.append(prefix + "Gesamtinhalt ohne gültige Maßeinheit")
    if item.get("packaging_type") not in (None, "", "unknown") and packaging_type not in PACKAGING_TYPES:
        issues.append(prefix + "Verpackungsart ungültig")
    if item.get("price_basis") not in (None, "", "unknown") and price_basis not in PRICE_BASES:
        issues.append(prefix + "Preisbasis ungültig")

    if count is not None and _positive_float(size_value) is not None and _positive_float(total_value) is not None:
        if not _same_content_amount(
            size_value,
            size_unit,
            total_value,
            total_unit,
            multiplier=count,
        ):
            issues.append(prefix + "Packungsanzahl × Einzelinhalt widerspricht Gesamtinhalt")

    packaging_text = " ".join(
        str(value or "").strip()
        for value in (item.get("packaging_raw"), item.get("description"))
        if str(value or "").strip()
    )
    multipack_matches = list(_MULTIPACK_PATTERN.finditer(packaging_text))
    for match in _PACKAGING_WORD_PATTERN.finditer(packaging_text):
        expected_value = match.group("value").replace(",", ".")
        expected_unit = _normalized_content_unit(match.group("unit"))
        expected_type = _normalized_packaging_type(match.group("pack"))
        if not multipack_matches:
            if not _same_content_amount(
                expected_value,
                expected_unit,
                size_value,
                size_unit,
            ):
                issues.append(prefix + "sichtbare Inhaltsmenge fehlt/widerspricht Packungsstruktur")
        if not _same_content_amount(
            expected_value,
            expected_unit,
            total_value,
            total_unit,
        ):
            issues.append(prefix + "sichtbarer Gesamtinhalt fehlt/widerspricht Packungsstruktur")
        if packaging_type != expected_type:
            issues.append(prefix + "sichtbare Verpackungsart fehlt/widerspricht Packungsstruktur")
        if not multipack_matches and (count is None or not math.isclose(count, 1.0)):
            issues.append(prefix + "einfache Packung muss package_count 1 haben")

        legacy_unit = _normalized_packaging_type(item.get("unit"))
        legacy_quantity = _positive_float(item.get("quantity"))
        if (
            _positive_float(size_value) is None
            and legacy_unit == expected_type
            and legacy_quantity is not None
            and math.isclose(legacy_quantity, float(expected_value), rel_tol=0.0, abs_tol=0.001)
        ):
            issues.append(prefix + "Inhaltsmenge wurde als Packungsanzahl extrahiert")

    for match in multipack_matches:
        expected_count = float(match.group("count"))
        expected_size = match.group("size").replace(",", ".")
        expected_unit = _normalized_content_unit(match.group("unit"))
        if count is None or not math.isclose(count, expected_count):
            issues.append(prefix + "sichtbare Multipack-Anzahl fehlt/widerspricht")
        if not _same_content_amount(
            expected_size,
            expected_unit,
            size_value,
            size_unit,
        ):
            issues.append(prefix + "sichtbarer Multipack-Einzelinhalt fehlt/widerspricht")
        if _positive_float(total_value) is None or total_unit not in CONTENT_UNITS - {"unknown"}:
            issues.append(prefix + "Multipack-Gesamtinhalt fehlt")

    calibre_match = _CALIBRE_PER_LB_PATTERN.search(packaging_text)
    if calibre_match and count is not None and count in {
        float(calibre_match.group("low")),
        float(calibre_match.group("high")),
    }:
        issues.append(prefix + "Kaliber Stück/lb wurde als Packungsanzahl extrahiert")

    source_brand = str(item.get("source_brand") or "").strip().casefold()
    if source_brand in {"asc", "msc", "bio", "halal", "qs"}:
        issues.append(prefix + "Zertifizierung wurde als Marke extrahiert")

    if _positive_float(item.get("price_per_kg")) is not None and price_basis == "unknown":
        issues.append(prefix + "Preisbasis fehlt trotz sichtbarem Grundpreis")

    return list(dict.fromkeys(issues))


def _looks_garbled(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "\ufffd" in text:
        return True
    if any(ord(char) < 32 and not char.isspace() for char in text):
        return True
    if re.search(r"([^\w\s])\1{3,}", text, flags=re.UNICODE):
        return True

    visible = [char for char in text if not char.isspace()]
    unusual = [
        char
        for char in visible
        if not char.isalnum() and char not in _COMMON_PRODUCT_PUNCTUATION
    ]
    return len(unusual) >= 3 and len(unusual) / max(len(visible), 1) > 0.2


def _complete_item_count(raw_items: list[dict] | None) -> int:
    if not isinstance(raw_items, list):
        return 0
    count = 0
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("product_name") or "").strip()
        try:
            price = float(item.get("price"))
        except (TypeError, ValueError):
            price = 0.0
        if name and not _looks_garbled(name) and math.isfinite(price) and price > 0:
            count += 1
    return count


def _page_quality_rank(
    raw_items: list[dict] | None,
    quality_issues: list[str] | tuple[str, ...] | None = None,
) -> tuple[int, int, int, int, int]:
    items = raw_items if isinstance(raw_items, list) else []
    issues = extraction_quality_issues(raw_items) if quality_issues is None else quality_issues
    return (
        _complete_item_count(items),
        int(bool(items)),
        len(items),
        int(raw_items is not None),
        -len(issues),
    )


def apply_quality_retry_once(
    primary_items: list[dict] | None,
    *,
    image_path: str,
    supplier: str,
    primary_model: str,
    retry_executor: Callable[..., list[dict] | None] | None = None,
) -> PageExtractionOutcome:
    """Assess one primary page once and submit at most one fixed quality retry."""
    primary_issues = extraction_quality_issues(primary_items)
    selected_primary = primary_items if isinstance(primary_items, list) else []
    if not primary_issues:
        return PageExtractionOutcome(
            primary_items=primary_items,
            selected_items=selected_primary,
            primary_model=primary_model,
            selected_model=primary_model,
            primary_failed=primary_items is None,
            primary_quality_issues=(),
            quality_retry_attempted=False,
            quality_retry_status="not_needed",
        )

    executor = retry_executor or extract_products_from_image
    try:
        retry_items = executor(
            image_path,
            supplier,
            model_name=QUALITY_RETRY_MODEL,
            max_retries=1,
            temperature=None,
            operation="product_extraction_quality_retry",
            quality_issues=primary_issues,
        )
    except Exception as exc:
        logger.warning("Gemini quality retry failed for %s: %s", image_path, exc)
        return PageExtractionOutcome(
            primary_items=primary_items,
            selected_items=selected_primary,
            primary_model=primary_model,
            selected_model=primary_model,
            primary_failed=primary_items is None,
            primary_quality_issues=tuple(primary_issues),
            quality_retry_attempted=True,
            quality_retry_status="failed",
            quality_retry_model=QUALITY_RETRY_MODEL,
            retry_items=None,
            retry_quality_issues=("Qualitätsretry fehlgeschlagen",),
        )

    if retry_items is None or not isinstance(retry_items, list):
        retry_issues = ["Extraktion/API fehlgeschlagen"]
        retry_status = "failed"
        use_retry = False
    else:
        retry_issues = extraction_quality_issues(retry_items)
        # Never trade recall for cleaner-looking output: a retry with fewer raw
        # observations cannot replace a successful primary page.
        recall_preserved = primary_items is None or len(retry_items) >= len(primary_items)
        use_retry = recall_preserved and _page_quality_rank(
            retry_items, retry_issues
        ) > _page_quality_rank(primary_items, primary_issues)
        retry_status = "selected" if use_retry else "kept_primary"

    return PageExtractionOutcome(
        primary_items=primary_items,
        selected_items=retry_items if use_retry and retry_items is not None else selected_primary,
        primary_model=primary_model,
        selected_model=QUALITY_RETRY_MODEL if use_retry else primary_model,
        primary_failed=primary_items is None,
        primary_quality_issues=tuple(primary_issues),
        quality_retry_attempted=True,
        quality_retry_status=retry_status,
        quality_retry_model=QUALITY_RETRY_MODEL,
        retry_items=retry_items,
        retry_quality_issues=tuple(retry_issues),
    )


def file_sha256(path: str | Path | None) -> str | None:
    """Hash the actual local source document once for stable provenance."""
    if not path:
        return None
    source_path = Path(path)
    if not source_path.is_file():
        return None
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_batch_page_outcome_manifest(
    image_paths: list[str],
    records: list[dict[str, Any]],
) -> Path | None:
    """Persist page completeness even when no product row can carry provenance."""
    if not image_paths:
        return None
    first_image = Path(image_paths[0])
    if not first_image.is_file():
        return None
    manifest_path = first_image.parent / "extraction_outcomes.jsonl"
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: int(item.get("source_page") or 0)):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_path.replace(manifest_path)
    return manifest_path


def _normalized_decimal_text(value: Any) -> str:
    try:
        number = Decimal(str(value).strip().replace(",", "."))
    except (InvalidOperation, TypeError, ValueError):
        return str(value or "").strip()
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"", "-0"} else normalized


def _canonical_raw_value(value: Any, field_name: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_raw_value(item, str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item is not None and item != "" and item != []
        }
    if isinstance(value, (list, tuple)):
        normalized = [_canonical_raw_value(item) for item in value]
        if field_name in {"certifications", "price_tiers"}:
            return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return normalized
    if field_name in _NUMERIC_RAW_ITEM_FIELDS and value not in (None, ""):
        return _normalized_decimal_text(value)
    if field_name in _BOOLEAN_RAW_ITEM_FIELDS:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().casefold() in {"1", "true", "yes", "ja"}
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value)
        return re.sub(r"\s+", " ", normalized).strip().casefold()
    return value


def _canonical_raw_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    # Confidence is a model-side score, not source identity. A changed confidence
    # value must not churn the stable source-item ID.
    filtered = {
        key: value
        for key, value in raw_item.items()
        if key != "confidence"
        and value is not None
        and value != ""
        and value != []
    }
    return _canonical_raw_value(filtered)


def _raw_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "ja", "y"}


def _raw_item_fingerprint(raw_item: dict[str, Any]) -> str:
    payload = json.dumps(
        _canonical_raw_item(raw_item),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_source_item_id(
    source_document_sha256: str,
    source_page: int | None,
    raw_item: dict[str, Any],
    occurrence_index: int = 0,
) -> str:
    """Create a row-order-independent ID; ordinal separates identical duplicates."""
    document_hash = str(source_document_sha256 or "").strip().casefold()
    if document_hash.startswith("sha256:"):
        document_hash = document_hash.split(":", 1)[1]
    payload = {
        "document_sha256": document_hash,
        "source_page": int(source_page) if source_page is not None else None,
        "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        "raw_item_fingerprint": _raw_item_fingerprint(raw_item),
        "occurrence_index": int(occurrence_index),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "si_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _raw_items_to_products(
    raw_items: list[dict],
    *,
    supplier: str,
    source_file: str = "",
    source_page: int | None = None,
    valid_from=None,
    valid_to=None,
    calendar_week: int | None = None,
    year: int | None = None,
    location: str | None = None,
    source_title: str | None = None,
    source_tab: str | None = None,
    fallback_category: str | None = None,
    source_document_sha256: str | None = None,
    extraction_outcome: PageExtractionOutcome | None = None,
) -> list[RawProduct]:
    products: list[RawProduct] = []
    document_sha256 = source_document_sha256 or file_sha256(source_file)
    if source_file and not document_sha256:
        logger.warning(
            "Source PDF hash unavailable for %s; source_item_id will remain empty",
            source_file,
        )
    occurrence_counts: dict[str, int] = {}

    for source_item_index, item in enumerate(raw_items, 1):
        try:
            fingerprint = _raw_item_fingerprint(item)
            occurrence_index = occurrence_counts.get(fingerprint, 0)
            occurrence_counts[fingerprint] = occurrence_index + 1
            category = item.get("category") or "sonstiges"
            if category == "sonstiges" and fallback_category:
                category = fallback_category
            unit = item.get("unit") or item.get("total_content_unit") or "stueck"
            quantity = (
                item.get("quantity")
                if item.get("quantity") not in (None, "")
                else item.get("total_content_value")
            )
            price = item.get("price")
            if price is None:
                logger.warning(
                    "Skipping product on page %s because price is missing - data: %s",
                    source_page,
                    item,
                )
                continue

            product = RawProduct(
                supplier=supplier,
                location=location,
                product_name=item.get("product_name", ""),
                description=item.get("description"),
                category=category,
                product_family=item.get("product_family") or "unknown",
                temperature_state=item.get("temperature_state") or "unknown",
                processing_state=item.get("processing_state") or "unknown",
                calibre=item.get("calibre"),
                source_brand=item.get("source_brand"),
                brand_evidence=item.get("brand_evidence"),
                brand_evidence_source=item.get("brand_evidence_source") or "unknown",
                certifications=item.get("certifications") or [],
                origin=item.get("origin"),
                unit=unit,
                quantity=quantity,
                price=float(price),
                price_basis=item.get("price_basis") or "unknown",
                price_is_net=_raw_bool(item.get("price_is_net", False)),
                price_gross=item.get("price_gross"),
                price_per_kg=item.get("price_per_kg"),
                price_tiers=item.get("price_tiers"),
                package_count=item.get("package_count"),
                package_size_value=item.get("package_size_value"),
                package_size_unit=item.get("package_size_unit") or "unknown",
                total_content_value=item.get("total_content_value"),
                total_content_unit=item.get("total_content_unit") or "unknown",
                packaging_type=item.get("packaging_type") or "unknown",
                packaging_raw=item.get("packaging_raw"),
                valid_from=valid_from,
                valid_to=valid_to,
                calendar_week=calendar_week,
                year=year,
                source_file=source_file,
                source_title=source_title,
                source_tab=source_tab,
                source_page=source_page,
                extraction_confidence=item.get("confidence", 0.8),
                source_item_index=source_item_index,
                source_item_id=(
                    make_source_item_id(
                        document_sha256,
                        source_page,
                        item,
                        occurrence_index=occurrence_index,
                    )
                    if document_sha256
                    else None
                ),
                source_document_sha256=document_sha256,
                primary_extraction_model=(
                    extraction_outcome.primary_model
                    if extraction_outcome
                    else _settings["model_name"]
                ),
                selected_extraction_model=(
                    extraction_outcome.selected_model
                    if extraction_outcome
                    else _settings["model_name"]
                ),
                quality_retry_attempted=(
                    extraction_outcome.quality_retry_attempted
                    if extraction_outcome
                    else False
                ),
                quality_retry_status=(
                    extraction_outcome.quality_retry_status
                    if extraction_outcome
                    else "not_needed"
                ),
                quality_retry_model=(
                    extraction_outcome.quality_retry_model
                    if extraction_outcome
                    else None
                ),
                quality_retry_issues=(
                    list(extraction_outcome.primary_quality_issues)
                    if extraction_outcome
                    else []
                ),
                extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
            )
            products.append(product)
        except Exception as e:
            logger.warning(
                "Failed to parse product on page %s: %s - data: %s",
                source_page,
                e,
                item,
            )

    return products


def extract_products_from_pdf_images(
    image_paths: list[str],
    supplier: str,
    source_file: str = "",
    valid_from=None,
    valid_to=None,
    calendar_week: int | None = None,
    year: int | None = None,
    location: str | None = None,
    source_title: str | None = None,
    source_tab: str | None = None,
    fallback_category: str | None = None,
    model_name: str | None = None,
    max_concurrent_requests: int | None = None,
    source_document_sha256: str | None = None,
) -> list[RawProduct]:
    model_name = _resolve_setting("model_name", model_name)
    max_retries = _resolve_setting("max_retries", None)
    temperature = _resolve_setting("temperature", None)
    max_concurrent_requests = _resolve_setting("max_concurrent_requests", max_concurrent_requests)
    all_products = []
    page_manifest_records: list[dict[str, Any]] = []
    page_results: dict[int, tuple[str, list[dict] | None]] = {}
    page_outcomes: dict[int, tuple[str, PageExtractionOutcome]] = {}
    document_sha256 = source_document_sha256 or file_sha256(source_file)

    if not image_paths:
        return all_products

    extraction_started = time.perf_counter()
    logger.info(
        f"Processing {len(image_paths)} pages from {Path(source_file).name or supplier} "
        f"with up to {min(max_concurrent_requests, len(image_paths))} concurrent Vision calls"
    )
    log_event(
        logger,
        f"Vision PDF extraction started for {Path(source_file).name or supplier}",
        event="vision_pdf_extract",
        status="start",
        supplier=supplier,
        source_file=source_file,
        page_count=len(image_paths),
        concurrency=min(max_concurrent_requests, len(image_paths)),
        model_name=model_name,
    )

    with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(image_paths))) as executor:
        future_map = {
            executor.submit(
                extract_products_from_image,
                img_path,
                supplier,
                model_name=model_name,
                max_retries=max_retries,
                temperature=temperature,
            ): (i, img_path)
            for i, img_path in enumerate(image_paths, 1)
        }

        for future in as_completed(future_map):
            i, img_path = future_map[future]
            logger.info(f"Processing page {i}/{len(image_paths)}: {Path(img_path).name}")
            try:
                raw_items = future.result()
            except Exception as e:
                logger.warning(f"Vision worker failed on page {i}: {e}")
                raw_items = None
            page_results[i] = (img_path, raw_items)

    # All pages pass through the same coordinator. Clear pages return without an
    # API call; unclear pages invoke the injected retry executor at most once.
    with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(page_results))) as executor:
        outcome_future_map = {
            executor.submit(
                apply_quality_retry_once,
                raw_items,
                image_path=img_path,
                supplier=supplier,
                primary_model=model_name,
                retry_executor=extract_products_from_image,
            ): (page_number, img_path)
            for page_number, (img_path, raw_items) in page_results.items()
        }
        for future in as_completed(outcome_future_map):
            page_number, img_path = outcome_future_map[future]
            outcome = future.result()
            page_outcomes[page_number] = (img_path, outcome)

    for i, img_path in enumerate(image_paths, 1):
        _, outcome = page_outcomes.get(
            i,
            (
                img_path,
                PageExtractionOutcome(
                    primary_items=None,
                    selected_items=[],
                    primary_model=model_name,
                    selected_model=model_name,
                    primary_failed=True,
                    primary_quality_issues=("Seitenergebnis fehlt",),
                    quality_retry_attempted=False,
                    quality_retry_status="failed",
                ),
            ),
        )
        page_products = _raw_items_to_products(
            outcome.selected_items,
            supplier=supplier,
            source_file=source_file,
            source_page=i,
            valid_from=valid_from,
            valid_to=valid_to,
            calendar_week=calendar_week,
            year=year,
            location=location,
            source_title=source_title,
            source_tab=source_tab,
            fallback_category=fallback_category,
            source_document_sha256=document_sha256,
            extraction_outcome=outcome,
        )
        all_products.extend(page_products)
        manifest_record = outcome.to_manifest_record(
            supplier=supplier,
            source_file=source_file,
            source_document_sha256=document_sha256,
            source_page=i,
            image_path=img_path,
        )
        manifest_record["accepted_product_count"] = len(page_products)
        manifest_record["page_complete"] = bool(outcome.selected_items) and (
            len(page_products) == len(outcome.selected_items)
        ) and not extraction_quality_issues(outcome.selected_items)
        page_manifest_records.append(manifest_record)
        log_event(
            logger,
            f"Vision page outcome recorded for page {i}",
            event="vision_page_outcome",
            status=outcome.quality_retry_status,
            **manifest_record,
        )

    logger.info(f"Total extracted: {len(all_products)} products from {len(image_paths)} pages")
    outcome_manifest_path = write_batch_page_outcome_manifest(
        image_paths,
        page_manifest_records,
    )
    quality_retry_page_count = sum(
        outcome.quality_retry_attempted
        for _img_path, outcome in page_outcomes.values()
    )
    log_event(
        logger,
        f"Vision PDF extraction completed for {Path(source_file).name or supplier}",
        event="vision_pdf_extract",
        status="ok",
        supplier=supplier,
        source_file=source_file,
        page_count=len(image_paths),
        quality_retry_page_count=quality_retry_page_count,
        product_count=len(all_products),
        duration_ms=round((time.perf_counter() - extraction_started) * 1000, 2),
        model_name=model_name,
        source_document_sha256=document_sha256,
        outcome_manifest_path=str(outcome_manifest_path) if outcome_manifest_path else None,
    )
    return all_products
