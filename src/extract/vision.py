from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
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

QUALITY_RETRY_MODEL = "gemini-3.6-flash"
PACKAGING_UNITS_REQUIRING_QUANTITY = {
    "becher",
    "beutel",
    "dose",
    "eimer",
    "flasche",
    "kanister",
    "karton",
    "kasten",
    "kiste",
    "korb",
    "packung",
    "schale",
}

_client = None
_settings = {
    "model_name": "gemini-2.5-flash",
    "max_retries": 3,
    "temperature": 0.1,
    "max_concurrent_requests": 10,
    "min_request_interval_seconds": 0.0,
}
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
    include_temperature: bool = True,
    operation: str = "vision_json",
    **context,
):
    global _client
    if not _client:
        raise RuntimeError("Gemini not configured. Call configure_gemini() first.")

    model_name = _resolve_setting("model_name", model_name)
    max_retries = _resolve_setting("max_retries", max_retries)
    if include_temperature:
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
            config_kwargs = {
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
            }
            if include_temperature:
                config_kwargs["temperature"] = temperature
            response = _client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    Part.from_bytes(data=img_bytes, mime_type=mime_type),
                ],
                config=GenerateContentConfig(**config_kwargs),
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
) -> list[dict]:
    prompt = get_extraction_prompt(supplier)
    result = analyze_image_json(
        image_path=image_path,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        model_name=model_name,
        max_retries=max_retries,
        temperature=temperature,
        operation="product_extraction",
        supplier=supplier,
    )
    if result is None:
        return []

    products = result if isinstance(result, list) else [result]
    logger.debug(f"Extracted {len(products)} products from {Path(image_path).name}")
    return products


def page_extraction_quality_issues(raw_items: list[dict]) -> list[str]:
    """Return retry triggers before lossy RawProduct conversion.

    Price-less entries would otherwise be silently skipped and missing unit or
    quantity values would be defaulted. Checking the raw page result preserves
    the evidence needed to decide whether the complete page needs one retry.
    """
    issues: list[str] = []
    for index, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            issues.append(f"item_{index}:invalid_item")
            continue
        if not _positive_number(item.get("price")):
            issues.append(f"item_{index}:missing_price")
        unit = str(item.get("unit") or "").strip().casefold()
        if not _positive_number(item.get("quantity")) and _quantity_should_be_present(item, unit):
            issues.append(f"item_{index}:missing_quantity")
        if unit in {"", "unknown", "null", "none"}:
            issues.append(f"item_{index}:missing_unit")
    return issues


def _positive_number(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return float(str(value).strip().replace(",", ".")) > 0
    except (TypeError, ValueError):
        return False


def _quantity_should_be_present(item: dict, unit: str) -> bool:
    if unit in PACKAGING_UNITS_REQUIRING_QUANTITY:
        return True
    visible_text = " ".join(
        str(item.get(field) or "")
        for field in ("product_name", "description")
    ).casefold()
    return bool(
        re.search(
            r"(?<!\d)\d+(?:[.,]\d+)?\s*(?:kg|g|l|ml|stück|stueck|x)\b",
            visible_text,
        )
    )


def _product_items(result) -> list[dict] | None:
    if result is None:
        return None
    return result if isinstance(result, list) else [result]


def extract_products_from_image_with_quality_retry(
    image_path: str,
    supplier: str,
    model_name: str | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    *,
    page_number: int | None = None,
    source_file: str = "",
) -> list[dict]:
    """Extract one page and atomically replace it after one quality retry.

    The retry uses the exact same output prompt/schema with Gemini 3.6 Flash.
    It is called at most once per page and never merged field-by-field with the
    primary result. A failed or unexpectedly empty retry keeps the primary page
    to prevent destructive data loss.
    """
    primary_items = extract_products_from_image(
        image_path,
        supplier,
        model_name=model_name,
        max_retries=max_retries,
        temperature=temperature,
    )
    issues = page_extraction_quality_issues(primary_items)
    if not issues:
        return primary_items

    primary_model = _resolve_setting("model_name", model_name)
    context = {
        "supplier": supplier,
        "source_file": source_file,
        "source_page": page_number,
        "primary_model": primary_model,
        "retry_model": QUALITY_RETRY_MODEL,
        "quality_issues": issues,
        "primary_product_count": len(primary_items),
    }
    logger.warning(
        "Page %s has incomplete price/amount data; retrying complete page once with %s (%s)",
        page_number if page_number is not None else Path(image_path).name,
        QUALITY_RETRY_MODEL,
        ", ".join(issues),
    )
    log_event(
        logger,
        f"Gemini quality retry started for {Path(image_path).name}",
        event="vision_page_quality_retry",
        status="start",
        **context,
    )

    retry_result = analyze_image_json(
        image_path=image_path,
        prompt=get_extraction_prompt(supplier),
        system_prompt=SYSTEM_PROMPT,
        model_name=QUALITY_RETRY_MODEL,
        # Exactly one additional page analysis, not the normal retry loop.
        max_retries=1,
        include_temperature=False,
        operation="product_extraction_quality_retry",
        **context,
    )
    retry_items = _product_items(retry_result)
    if retry_items is None or (primary_items and not retry_items):
        status = "failed" if retry_items is None else "rejected_empty"
        logger.error(
            "Gemini quality retry %s for page %s; retaining primary page result",
            status,
            page_number if page_number is not None else Path(image_path).name,
        )
        log_event(
            logger,
            f"Gemini quality retry did not replace {Path(image_path).name}",
            event="vision_page_quality_retry",
            level=logging.ERROR,
            status=status,
            **context,
        )
        return primary_items

    remaining_issues = page_extraction_quality_issues(retry_items)
    log_event(
        logger,
        f"Gemini quality retry replaced complete page {Path(image_path).name}",
        event="vision_page_quality_retry",
        status="replaced",
        retry_product_count=len(retry_items),
        remaining_quality_issues=remaining_issues,
        **context,
    )
    logger.info(
        "Replaced complete primary extraction for page %s with %d %s products; remaining issues=%d",
        page_number if page_number is not None else Path(image_path).name,
        len(retry_items),
        QUALITY_RETRY_MODEL,
        len(remaining_issues),
    )
    return retry_items


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
                extract_products_from_image_with_quality_retry,
                img_path,
                supplier,
                model_name=model_name,
                max_retries=max_retries,
                temperature=temperature,
                page_number=i,
                source_file=source_file,
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
        product_count=len(all_products),
        duration_ms=round((time.perf_counter() - extraction_started) * 1000, 2),
        model_name=model_name,
    )
    return all_products
