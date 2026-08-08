from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
import random
import re
import threading
import time
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig, Part

from src.extract.prompts import SYSTEM_PROMPT, get_extraction_prompt
from src.harmonize.customer_rules import apply_customer_category_overrides
from src.models import RawProduct
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
_rate_limit_lock = threading.Lock()
_last_request_started_at = 0.0


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
) -> list[dict]:
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
        return []

    products = result if isinstance(result, list) else [result]
    logger.debug(f"Extracted {len(products)} products from {Path(image_path).name}")
    return products


def extraction_quality_issues(raw_items: list[dict] | None) -> list[str]:
    """Return compact reasons that make one page worth exactly one quality retry."""
    if raw_items is None:
        return ["kein Ergebnis"]
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

    return issues


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


def _page_quality_rank(raw_items: list[dict] | None) -> tuple[int, int, int, int]:
    items = raw_items if isinstance(raw_items, list) else []
    return (
        _complete_item_count(items),
        int(bool(items)),
        -len(extraction_quality_issues(items)),
        len(items),
    )


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
) -> list[RawProduct]:
    products: list[RawProduct] = []

    for item in raw_items:
        try:
            category = item.get("category") or "sonstiges"
            if category == "sonstiges" and fallback_category:
                category = fallback_category
            category = apply_customer_category_overrides(
                category,
                product_name=item.get("product_name"),
                description=item.get("description"),
                unit=item.get("unit"),
            )
            unit = item.get("unit") or "stueck"
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
                origin=item.get("origin"),
                unit=unit,
                quantity=item.get("quantity"),
                price=float(price),
                price_is_net=bool(item.get("price_is_net", False)),
                price_gross=item.get("price_gross"),
                price_per_kg=item.get("price_per_kg"),
                price_tiers=item.get("price_tiers"),
                valid_from=valid_from,
                valid_to=valid_to,
                calendar_week=calendar_week,
                year=year,
                source_file=source_file,
                source_title=source_title,
                source_tab=source_tab,
                source_page=source_page,
                extraction_confidence=item.get("confidence", 0.8),
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
) -> list[RawProduct]:
    model_name = _resolve_setting("model_name", model_name)
    max_retries = _resolve_setting("max_retries", None)
    temperature = _resolve_setting("temperature", None)
    max_concurrent_requests = _resolve_setting("max_concurrent_requests", max_concurrent_requests)
    all_products = []
    page_results: dict[int, tuple[str, list[dict]]] = {}

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
                raw_items = []
            page_results[i] = (img_path, raw_items)

    retry_pages = {
        page_number: (img_path, raw_items, extraction_quality_issues(raw_items))
        for page_number, (img_path, raw_items) in page_results.items()
        if extraction_quality_issues(raw_items)
    }
    if retry_pages:
        logger.info(
            "Retrying %d unclear PDF pages exactly once with %s",
            len(retry_pages),
            QUALITY_RETRY_MODEL,
        )
        log_event(
            logger,
            f"Vision quality retry started for {Path(source_file).name or supplier}",
            event="vision_quality_retry",
            status="start",
            supplier=supplier,
            source_file=source_file,
            retry_page_count=len(retry_pages),
            model_name=QUALITY_RETRY_MODEL,
        )
        with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(retry_pages))) as executor:
            retry_future_map = {
                executor.submit(
                    extract_products_from_image,
                    img_path,
                    supplier,
                    model_name=QUALITY_RETRY_MODEL,
                    max_retries=1,
                    temperature=None,
                    operation="product_extraction_quality_retry",
                    quality_issues=issues,
                ): (page_number, img_path, original_items, issues)
                for page_number, (img_path, original_items, issues) in retry_pages.items()
            }
            for future in as_completed(retry_future_map):
                page_number, img_path, original_items, original_issues = retry_future_map[future]
                try:
                    retry_items = future.result()
                except Exception as exc:
                    logger.warning("Gemini quality retry failed on page %d: %s", page_number, exc)
                    retry_items = []

                use_retry = _page_quality_rank(retry_items) > _page_quality_rank(original_items)
                selected_items = retry_items if use_retry else original_items
                page_results[page_number] = (img_path, selected_items)
                log_event(
                    logger,
                    f"Vision quality retry completed for page {page_number}",
                    event="vision_quality_retry",
                    status="ok" if use_retry else "kept_original",
                    supplier=supplier,
                    source_file=source_file,
                    source_page=page_number,
                    model_name=QUALITY_RETRY_MODEL,
                    original_issues=original_issues,
                    retry_issues=extraction_quality_issues(retry_items),
                    selected_result="retry" if use_retry else "original",
                )

    for i, img_path in enumerate(image_paths, 1):
        _, raw_items = page_results.get(i, (img_path, []))
        all_products.extend(
            _raw_items_to_products(
                raw_items,
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
            )
        )

    logger.info(f"Total extracted: {len(all_products)} products from {len(image_paths)} pages")
    log_event(
        logger,
        f"Vision PDF extraction completed for {Path(source_file).name or supplier}",
        event="vision_pdf_extract",
        status="ok",
        supplier=supplier,
        source_file=source_file,
        page_count=len(image_paths),
        quality_retry_page_count=len(retry_pages),
        product_count=len(all_products),
        duration_ms=round((time.perf_counter() - extraction_started) * 1000, 2),
        model_name=model_name,
    )
    return all_products
