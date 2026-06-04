#!/usr/bin/env python3
"""Classify parsed supplier rows as fresh-food relevant via the DeepSeek API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import local
from typing import Any

import requests
from src.report.parsed_csv import find_existing_combined_parsed_csv_path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps script usable.
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "parsed" / "KW18_2026" / "all_suppliers.csv"
DEFAULT_WORKERS = 25
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 80
REASONER_MAX_TOKENS = 4096
_THREAD_LOCAL = local()
PACKAGED_REQUIRED_BRANDS = [
    "aro",
    "chef",
    "metro",
    "milram",
    "schleiz",
    "quality",
    "economy",
    "edeka",
    "foodservice",
    "henkelmann",
    "meemken",
    "aviko",
]
CORE_FRESH_CATEGORY_REASONS = {
    "fisch": "Fisch",
    "fischtheke": "Fisch",
    "fish": "Fisch",
    "seafood": "Fisch",
    "fleisch": "Fleisch",
    "meat": "Fleisch",
    "poultry": "Fleisch",
    "gefluegel": "Fleisch",
    "geflügel": "Fleisch",
    "obst_gemuese": "Obst Gemüse",
    "obst_gemüse": "Obst Gemüse",
    "obst & gemüse": "Obst Gemüse",
    "obst und gemüse": "Obst Gemüse",
    "fruit": "Obst Gemüse",
    "vegetables": "Obst Gemüse",
}
SEAFOOD_KEYWORDS = [
    "fisch",
    "lachs",
    "seelachs",
    "scholle",
    "schollenfilet",
    "seeteufel",
    "dorade",
    "forelle",
    "garnelen",
    "garnele",
    "kammmuschel",
    "muschel",
    "octopus",
    "oktopus",
    "pulpo",
    "tintenfisch",
    "calamari",
]


def load_env_file(path: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(path, override=False)
        return

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a parsed supplier CSV, classify each row via DeepSeek, "
            "and append a Relevant column with Ja/Nein."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV path. Defaults to parsed/KW18_2026/all_suppliers.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to <input_stem>_relevant.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent DeepSeek requests. Default: 25",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional row limit for test runs.",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--deepseek-model",
        default=DEFAULT_DEEPSEEK_MODEL,
        help="DeepSeek model to use. Default: deepseek-v4-flash",
    )
    parser.add_argument(
        "--deepseek-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout per DeepSeek request in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retry count per row for API or parsing failures.",
    )
    return parser


def derive_output_path(input_path: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output
    return input_path.with_name(f"{input_path.stem}_relevant.csv")


def resolve_input_path(input_path: Path) -> Path:
    if input_path.exists():
        return input_path

    if input_path == DEFAULT_INPUT:
        legacy_or_canonical = find_existing_combined_parsed_csv_path(18, 2026, str(ROOT / "parsed"))
        if legacy_or_canonical.exists():
            return legacy_or_canonical

    return input_path


def load_rows(input_path: Path, limit: int | None) -> tuple[list[str], list[dict[str, str]]]:
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"CSV has no header row: {input_path}")
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    return fieldnames, rows


def get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def extract_message_content(message: object) -> str:
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None:
            return extract_message_content(content)
    return ""


def extract_response_text(response_json: dict[str, Any]) -> str:
    error = response_json.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error
        raise RuntimeError(f"DeepSeek API error: {message}")

    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek response does not contain choices.")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message", {})
    content = extract_message_content(message.get("content"))
    if not content:
        finish_reason = choice.get("finish_reason")
        reasoning_content = ""
        if isinstance(message, dict):
            reasoning_content = extract_message_content(message.get("reasoning_content"))
        detail = f" finish_reason={finish_reason!r}" if finish_reason else ""
        if reasoning_content:
            detail += f"; reasoning_without_final={reasoning_content[:120]!r}"
        raise RuntimeError(f"DeepSeek response did not contain text content.{detail}")
    return content


def max_tokens_for_model(model: str) -> int:
    model_key = (model or "").casefold()
    if "reasoner" in model_key or "pro" in model_key:
        return REASONER_MAX_TOKENS
    return DEFAULT_MAX_TOKENS


def normalize_classification(raw_text: str) -> tuple[str, str]:
    text = raw_text.strip()
    match = re.search(r"\b(ja|nein)\b", text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Expected Ja/Nein from DeepSeek, got: {raw_text!r}")
    label = "Ja" if match.group(1).lower() == "ja" else "Nein"

    reason = text[match.end():].strip(" \t\r\n-|:;,")
    reason = re.sub(r"\s+", " ", reason)
    if not reason:
        reason = "Fresh Product" if label == "Ja" else "Not Relevant"
    return label, _short_reason(reason)


def _short_reason(reason: str) -> str:
    reason = reason.strip(" \t\r\n-|:;,")
    words = reason.split()
    if len(words) <= 2:
        return reason
    return " ".join(words[:2])


def explicit_packaged_reason(row: dict[str, str]) -> str | None:
    product_name = (row.get("product_name") or "").casefold()
    description = (row.get("description") or "").casefold()
    category = (row.get("category") or "").casefold()
    brand = (row.get("brand") or "").casefold()
    text = f"{brand} {product_name} {description}"

    if not has_required_packaged_brand(row):
        return None

    if any(term in product_name for term in ["öl", "oel", "oil"]):
        return "Oil"
    for term, reason in [
        ("sahne", "Cream"),
        ("cream", "Cream"),
        ("quark", "Quark"),
        ("pommes", "French fries"),
        ("french fries", "French fries"),
        ("milch", "Milk"),
        ("milk", "Milk"),
    ]:
        if term in product_name:
            return reason
    if _is_frozen_vegetable_product(product_name, description, category):
        return "Frozen vegetables"
    if _is_large_packaged_product(product_name, description, row, ["käse", "kaese", "cheese"]):
        return "Cheese"
    if _is_large_packaged_product(product_name, description, row, ["wurst", "würst", "wuerst", "wiener", "sausage"]):
        return "Sausage"

    return None


def explicit_exclusion_reason(row: dict[str, str]) -> str | None:
    if _is_packaged_keyword_product(row) and not has_required_packaged_brand(row):
        return "Brand missing"
    return None


def _is_packaged_keyword_product(row: dict[str, str]) -> bool:
    product_name = (row.get("product_name") or "").casefold()
    description = (row.get("description") or "").casefold()
    category = (row.get("category") or "").casefold()
    text = f"{product_name} {description} {category}"
    return any(
        term in text
        for term in [
            "öl",
            "oel",
            "oil",
            "sahne",
            "cream",
            "quark",
            "pommes",
            "french fries",
            "milch",
            "milk",
            "käse",
            "kaese",
            "cheese",
            "wurst",
            "würst",
            "wuerst",
            "wiener",
            "sausage",
        ]
    )


def explicit_core_food_reason(row: dict[str, str]) -> str | None:
    category = " ".join((row.get("category") or "").casefold().replace("-", "_").split())
    if category in CORE_FRESH_CATEGORY_REASONS:
        return CORE_FRESH_CATEGORY_REASONS[category]

    product_name = (row.get("product_name") or "").casefold()
    description = (row.get("description") or "").casefold()
    text = f"{product_name} {description}"
    if any(keyword in text for keyword in SEAFOOD_KEYWORDS):
        return "Fisch"

    return None


def has_required_packaged_brand(row: dict[str, str]) -> bool:
    product_name = (row.get("product_name") or "").casefold()
    description = (row.get("description") or "").casefold()
    brand = (row.get("brand") or "").casefold()
    text = f"{brand} {product_name} {description}"
    return any(re.search(rf"(?<![a-z0-9]){re.escape(required)}(?![a-z0-9])", text) for required in PACKAGED_REQUIRED_BRANDS)


def _is_ice_cream_product(product_name: str, description: str, category: str) -> bool:
    text = f"{product_name} {description}"
    if "eiskugelbeutel" in text or "eisbergsalat" in text:
        return False
    return category == "tk" and any(
        term in text
        for term in [" eis", "eiscreme", "eis-swirl", "milcheis", "eiswannen"]
    )


def _is_frozen_vegetable_product(product_name: str, description: str, category: str) -> bool:
    text = f"{product_name} {description}"
    return category == "tk" and any(term in text for term in ["gemüse", "gemuese", "vegetable"]) and any(
        term in text for term in ["gefroren", "tiefkühl", "tiefkuehl", "tk", "frozen"]
    )


def _is_large_packaged_product(
    product_name: str,
    description: str,
    row: dict[str, str],
    product_terms: list[str],
) -> bool:
    if not any(term in product_name for term in product_terms):
        return False

    package_grams = _infer_package_grams(row, description)
    return package_grams is not None and package_grams >= 500


def _infer_package_grams(row: dict[str, str], description: str) -> float | None:
    grams: list[float] = []
    quantity = _parse_float(row.get("quantity"))
    unit = (row.get("unit") or "").casefold()

    if quantity is not None:
        if unit in {"kg", "kilogramm"}:
            grams.append(quantity * 1000)
        elif unit in {"g", "gramm"}:
            grams.append(quantity)
        elif unit in {"packung", "beutel", "dose", "eimer", "schale"} and quantity >= 100:
            grams.append(quantity)

    for match in re.finditer(r"(\d+(?:[,.]\d+)?)\s*(kg|g)\b", description):
        value = _parse_float(match.group(1))
        if value is None:
            continue
        grams.append(value * 1000 if match.group(2) == "kg" else value)

    return max(grams) if grams else None


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def relevant_time_label(row: dict[str, str]) -> str:
    valid_to = _parse_iso_date(row.get("valid_to"))
    week = _parse_int(row.get("calendar_week"))
    year = _parse_int(row.get("year"))
    if valid_to is None or week is None or year is None:
        return "Nein"
    try:
        friday = date.fromisocalendar(year, week, 5)
    except ValueError:
        return "Nein"
    return "Ja" if valid_to >= friday else "Nein"


def _parse_iso_date(value: object) -> date | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_int(value: object) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_product_name(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return text

    def normalize_word(match: re.Match[str]) -> str:
        word = match.group(0)
        return word[:1].upper() + word[1:].lower()

    return re.sub(r"[^\W\d_]+", normalize_word, text, flags=re.UNICODE)


def build_prompt(row: dict[str, str]) -> str:
    compact_row = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            "Classify this single wholesaler offer row.",
            "Return exactly this format: Ja|Reason or Nein|Reason.",
            "Reason must be 1-2 words, for example: Fresh Product, Oil, Cream, Quark, French fries, Milk, Cheese, Sausage, Frozen vegetables, Brand missing, Not Relevant.",
            "",
            "Relevant = Ja if the offered product itself is a core fresh-food product, for example:",
            "- meat or poultry",
            "- fish or seafood",
            "- vegetables, salads, herbs, mushrooms, or potatoes",
            "- fruit",
            "Fish/seafood remains relevant if it is chilled, thawed/aufgetaut, frozen/gefroren, MSC-certified, or sold from a fish counter.",
            "Do not answer Nein for fish/seafood only because it is frozen, thawed, or not explicitly called fresh.",
            "",
            "Relevant = Ja also for these packaged / not-fresh food products, but ONLY if one of the required brands is clearly present in product_name, brand, or description.",
            "Required brands for packaged / not-fresh products: ARO, Chef, Metro, Milram, Schleiz, Quality, economy, Edeka, Foodservice, Henkelmann, Meemken, Aviko.",
            "If the product is one of the packaged keyword products below but none of these brands is present, answer Nein|Brand missing.",
            "",
            "Still relevant packaged / not-fresh keyword products with required brand:",
            "- edible oil / Speiseöl, including Olivenöl and other cooking oils",
            "- cream / Sahne",
            "- Quark",
            "- cheese / Käse only when the package size is at least 500 g",
            "- sausage / Wurst only when the package size is at least 500 g",
            "- frozen vegetables / Tiefkühl Gemüse",
            "- French fries / Pommes",
            "- milk / Milch",
            "",
            "For Käse and Wurst, use product_name, description, quantity, and unit to infer package size.",
            "Treat 0.5 kg, 500 g, 500 ml, 1 kg, 1.5 kg, 800 g Abtropfgewicht, or larger as at least 500 g.",
            "",
            "Relevant = Nein for everything else, including:",
            "- frozen items except branded French fries and branded frozen vegetables",
            "- canned, jarred, dried, shelf-stable, or preserved food except branded edible oil",
            "- beverages, spices, sauces, dairy, cheese, eggs, bakery, desserts except the explicitly relevant packaged products above",
            "- packaging, kitchen equipment, household goods, non-food items",
            "- processed or ready-made products",
            "",
            "If the row is ambiguous, answer Nein.",
            "",
            "CSV row with header:",
            compact_row,
        ]
    )


def call_deepseek(
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: int,
    prompt: str,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict binary classifier. Reply exactly as "
                    "Ja|Reason or Nein|Reason. Reason must be 1-2 words."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens_for_model(model),
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = get_session().post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def classify_row(
    index: int,
    row: dict[str, str],
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[int, str, str]:
    core_reason = explicit_core_food_reason(row)
    if core_reason:
        return index, "Ja", core_reason

    explicit_reason = explicit_packaged_reason(row)
    if explicit_reason:
        return index, "Ja", explicit_reason

    exclusion_reason = explicit_exclusion_reason(row)
    if exclusion_reason:
        return index, "Nein", exclusion_reason

    prompt = build_prompt(row)
    product_name = row.get("product_name", "").strip() or "<unknown product>"

    for attempt in range(1, max_retries + 1):
        try:
            response_json = call_deepseek(
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                prompt=prompt,
            )
            raw_text = extract_response_text(response_json)
            label, reason = normalize_classification(raw_text)
            return index, label, reason
        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"DeepSeek classification failed for row {index + 1} ({product_name}): {exc}"
                ) from exc
            time.sleep(min(2 ** (attempt - 1), 8))

    raise AssertionError("unreachable")


def save_rows(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    results: list[tuple[str, str]],
) -> None:
    output_fields = [
        name for name in fieldnames
        if name not in {"Relevant", "Reason", "Relevant Time"}
    ] + ["Relevant", "Reason", "Relevant Time"]
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tmp_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row, (label, reason) in zip(rows, results, strict=True):
            output_row = dict(row)
            output_row["product_name"] = normalize_product_name(output_row.get("product_name"))
            output_row["Relevant"] = label
            output_row["Reason"] = reason
            output_row["Relevant Time"] = relevant_time_label(output_row)
            writer.writerow(output_row)

    tmp_path.replace(output_path)


def main() -> None:
    load_env_file(ROOT / ".env")
    args = build_parser().parse_args()

    if args.workers < 1:
        raise RuntimeError("--workers must be at least 1")
    if args.max_retries < 1:
        raise RuntimeError("--max-retries must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise RuntimeError("--limit must be at least 1")

    run_relevance_classification(
        input_path=args.input,
        output_path=args.output,
        workers=args.workers,
        limit=args.limit,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_model=args.deepseek_model,
        deepseek_timeout=args.deepseek_timeout,
        max_retries=args.max_retries,
    )


def run_relevance_classification(
    input_path: Path,
    output_path: Path | None = None,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    deepseek_base_url: str | None = None,
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL,
    deepseek_timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[Path, int, int]:
    load_env_file(ROOT / ".env")

    if workers < 1:
        raise RuntimeError("--workers must be at least 1")
    if max_retries < 1:
        raise RuntimeError("--max-retries must be at least 1")
    if limit is not None and limit < 1:
        raise RuntimeError("--limit must be at least 1")

    input_path = resolve_input_path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = derive_output_path(input_path, output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY fehlt in .env oder in der Umgebung.")

    fieldnames, rows = load_rows(input_path, limit)
    if not rows:
        raise RuntimeError(f"CSV contains no data rows: {input_path}")

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Using model {deepseek_model} with {workers} concurrent requests")

    results: list[tuple[str, str]] = [("", "")] * len(rows)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                classify_row,
                index,
                row,
                deepseek_api_key,
                deepseek_model,
                deepseek_base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
                deepseek_timeout,
                max_retries,
            ): index
            for index, row in enumerate(rows)
        }

        try:
            for future in as_completed(future_to_index):
                index, label, reason = future.result()
                results[index] = (label, reason)
                completed += 1
                if completed == len(rows) or completed % workers == 0:
                    print(f"Processed {completed}/{len(rows)} rows")
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    save_rows(output_path, fieldnames, rows, results)

    yes_count = sum(1 for label, _reason in results if label == "Ja")
    no_count = sum(1 for label, _reason in results if label == "Nein")
    print(f"Saved {len(rows)} classified rows to {output_path}")
    print(f"Relevant=Ja: {yes_count}")
    print(f"Relevant=Nein: {no_count}")
    return output_path, yes_count, no_count


if __name__ == "__main__":
    main()
