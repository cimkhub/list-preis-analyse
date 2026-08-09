#!/usr/bin/env python3
"""Translate product names to German while preserving source-backed identity evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "parsed" / "KW19_2026" / "all_suppliers_relevant.csv"
DEFAULT_WORKERS = 25
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 80
CACHE_PATH = ROOT / "embeddings" / "product_matching" / "brand_product_splits.jsonl"
CACHE_PROMPT_VERSION = "product-name-de-plain-v3"
RELEVANT_VALUES = {"yes", "ja", "true", "1", "relevant", "x"}
CERTIFICATION_ORDER = ("ASC", "MSC", "QS", "BIO")
CERTIFICATION_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:ASC|MSC|QS|BIO)(?![A-Za-z0-9])", re.IGNORECASE)
PROTECTED_ACRONYMS = ("ASC", "MSC", "ARO", "EU", "BBQ", "TK")
_THREAD_LOCAL = local()


class TranslationFormatError(RuntimeError):
    """The model did not return one plain-text product name."""


class SemanticProtectionError(RuntimeError):
    """The translated name lost or changed source-backed identity information."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add a German product name and source-verified identity fields to a supplier CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Defaults to overwriting --input safely.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--deepseek-base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL))
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL, choices=[DEFAULT_DEEPSEEK_MODEL])
    parser.add_argument("--deepseek-timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    return parser


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


def get_env_any(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def load_rows(input_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"CSV has no header row: {input_path}")
        return fieldnames, list(reader)


def relevant_row(row: dict[str, str]) -> bool:
    if "Relevant" not in row:
        relevant = True
    else:
        relevant = str(row.get("Relevant", "")).strip().casefold() in RELEVANT_VALUES
    if not relevant:
        return False
    if "Relevant Time" in row:
        return str(row.get("Relevant Time", "")).strip().casefold() in RELEVANT_VALUES
    return True


def source_product_name(row: dict[str, str]) -> str:
    """Return the immutable source name, including on safe re-runs of this stage."""
    original = row.get("product_name_original")
    if original is not None and str(original) != "":
        return str(original)
    return str(row.get("product_name") or "")


def source_cache_payload(row: dict[str, str]) -> dict[str, str]:
    """Use only source/context fields, never a previous translation or split result."""
    return {
        "prompt_version": CACHE_PROMPT_VERSION,
        "model": DEFAULT_DEEPSEEK_MODEL,
        "product_name_original": source_product_name(row),
        "description": str(row.get("description") or ""),
        "category": str(row.get("category") or ""),
        "origin": str(row.get("origin") or ""),
        "calibre": str(row.get("calibre") or ""),
    }


def cache_key(row: dict[str, str]) -> str:
    payload = json.dumps(
        source_cache_payload(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{CACHE_PROMPT_VERSION}:{digest}"


def read_cache(path: Path = CACHE_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    items: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("cache_key"):
                items[item["cache_key"]] = item
    return items


def append_cache(item: dict[str, str], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def title_case_product(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return text

    def normalize_word(match: re.Match[str]) -> str:
        word = match.group(0)
        return word[:1].upper() + word[1:].lower()

    return re.sub(r"[^\W\d_]+", normalize_word, text, flags=re.UNICODE)


def extract_certifications(row: dict[str, str]) -> list[str]:
    """Collect certifications from source-backed local evidence without using the LLM."""
    evidence = "\n".join(
        str(row.get(field) or "")
        for field in (
            "product_name_original",
            "product_name",
            "description",
            "source_brand",
            "brand_evidence",
        )
    )
    found = {match.group(0).upper() for match in CERTIFICATION_PATTERN.finditer(evidence)}
    return [certification for certification in CERTIFICATION_ORDER if certification in found]


def certifications_json(row: dict[str, str]) -> str:
    return json.dumps(extract_certifications(row), ensure_ascii=False, separators=(",", ":"))


def is_certification_only(value: object) -> bool:
    tokens = [token.upper() for token in CERTIFICATION_PATTERN.findall(str(value or ""))]
    normalized = re.sub(r"[^A-Za-z0-9]+", "", str(value or "")).upper()
    return bool(tokens) and normalized == "".join(tokens)


def _normalized_evidence_text(value: object) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", str(value or "").casefold()).strip()


def _evidence_occurs_in(value: object, evidence: object) -> bool:
    normalized_value = _normalized_evidence_text(value)
    normalized_evidence = _normalized_evidence_text(evidence)
    if not normalized_value or not normalized_evidence:
        return False
    return f" {normalized_evidence} " in f" {normalized_value} "


def verified_local_brand(row: dict[str, str]) -> str:
    """Prefer the relevance audit brand, then explicit source brand evidence."""
    relevance_brand = str(row.get("relevance_brand") or "").strip()
    if relevance_brand and not is_certification_only(relevance_brand):
        return relevance_brand

    source_brand = str(row.get("source_brand") or "").strip()
    evidence = str(row.get("brand_evidence") or "").strip()
    evidence_source = str(row.get("brand_evidence_source") or "").strip().casefold()
    brand_is_supported_by_text_evidence = (
        _evidence_occurs_in(evidence, source_brand)
        or _evidence_occurs_in(source_brand, evidence)
    )
    evidence_is_verified = (
        evidence_source == "image"
        or (
            evidence_source == "product_name"
            and brand_is_supported_by_text_evidence
            and _evidence_occurs_in(source_product_name(row), evidence)
        )
        or (
            evidence_source == "description"
            and brand_is_supported_by_text_evidence
            and _evidence_occurs_in(row.get("description"), evidence)
        )
    )
    if (
        source_brand
        and evidence
        and evidence_is_verified
        and not is_certification_only(source_brand)
    ):
        return source_brand
    return ""


def fallback_split(row: dict[str, str]) -> dict[str, str]:
    product_name = source_product_name(row)
    return {
        "brand": verified_local_brand(row),
        "product_name_de": product_name,
        "product": product_name,
        "certifications": certifications_json(row),
        # Kept blank only for backwards-compatible CSV/API shape. The translation
        # model no longer returns or determines a confidence value.
        "confidence": "",
        "source": "fallback",
    }


def build_prompt(row: dict[str, str]) -> str:
    payload = {
        "product_name_original": source_product_name(row),
        "description": row.get("description", ""),
        "category": row.get("category", ""),
    }
    return "\n".join([
        "Return exactly one final product name in German as plain text.",
        "Do not return JSON, Markdown, quotes, labels, explanations, confidence, reasons, or alternatives.",
        "Translate only wording that is clearly English. If the name is already German or unclear, return the original name unchanged.",
        "Use description and category only to disambiguate; never invent, generalize, or replace the product type.",
        "Return the complete product name. Do not split off or remove a brand.",
        "Preserve ASC, MSC, ARO, EU, BBQ and TK in exactly this uppercase spelling when present.",
        "Preserve every number, calibre, range and percentage, allowing only German decimal punctuation.",
        "Preserve Duroc, Black Angus and Free-Range when present.",
        "Keep Red Snapper as Red Snapper; never translate or normalize it to Rotbarsch.",
        "Keep Lachsforelle as Lachsforelle; never shorten or normalize it to Forelle or Forelle Rot.",
        "Keep ASC, MSC, QS and BIO in the returned name whenever they occur in product_name_original.",
        "",
        "Examples:",
        "Pork Back -> Schweinerücken",
        "ASC Red Snapper Filet 8/12 -> ASC Red Snapper Filet 8/12",
        "Black Angus Beef Burger BBQ -> Black Angus Rindfleischburger BBQ",
        "Lachsforelle -> Lachsforelle",
        "",
        "Input:",
        json.dumps(payload, ensure_ascii=False),
    ])


def call_deepseek(
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: int,
    prompt: str,
) -> str:
    if model != DEFAULT_DEEPSEEK_MODEL:
        raise RuntimeError(
            f"Product-name translation must use {DEFAULT_DEEPSEEK_MODEL}; got {model!r}."
        )
    response = get_session().post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate product names to German. Return only the single final "
                        "product name as plain text, with no JSON or explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error
        raise RuntimeError(f"DeepSeek API error: {message}")
    choice = ((payload.get("choices") or [{}])[0] or {})
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        finish_reason = choice.get("finish_reason")
        reasoning_content = (message.get("reasoning_content") or "").strip()
        detail = f" finish_reason={finish_reason!r}" if finish_reason else ""
        if reasoning_content:
            detail += f"; reasoning_without_final={reasoning_content[:120]!r}"
        raise RuntimeError(f"DeepSeek response has no content.{detail}")
    return normalize_plain_text_response(content)


def normalize_plain_text_response(text: object) -> str:
    product_name = str(text or "").strip()
    if not product_name:
        raise TranslationFormatError("DeepSeek returned an empty product name.")
    if (
        product_name.startswith("```")
        or product_name.startswith("{")
        or product_name.startswith("[")
        or "\n" in product_name
        or product_name[:1] in {'"', "'"}
        or product_name[-1:] in {'"', "'"}
    ):
        raise TranslationFormatError("DeepSeek did not return one unquoted plain-text name.")
    if len(product_name) > 300:
        raise TranslationFormatError("DeepSeek returned an implausibly long product name.")
    return product_name


def _contains_token(text: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text, re.IGNORECASE))


def _contains_upper_token(text: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text))


def _contains_phrase(text: str, phrase: str) -> bool:
    words = [re.escape(word) for word in re.split(r"[\s-]+", phrase) if word]
    return bool(re.search(r"(?<![A-Za-z0-9])" + r"[\s-]+".join(words) + r"(?![A-Za-z0-9])", text, re.IGNORECASE))


def _normalized_number_tokens(text: str) -> list[str]:
    tokens = re.findall(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?(?:\s*[/\-–]\s*\d+(?:[.,]\d+)?)?\s*%?", text)
    normalized = []
    for token in tokens:
        value = re.sub(r"\s+", "", token).replace(",", ".").replace("–", "-")
        normalized.append(value)
    return normalized


def semantic_protection_violations(original: str, translated: str) -> list[str]:
    violations: list[str] = []
    for token in PROTECTED_ACRONYMS:
        if _contains_token(original, token) and not _contains_upper_token(translated, token):
            violations.append(f"missing protected token {token}")

    for phrase in ("Duroc", "Black Angus", "Free-Range"):
        if _contains_phrase(original, phrase) and not _contains_phrase(translated, phrase):
            violations.append(f"missing protected phrase {phrase}")

    for certification in CERTIFICATION_ORDER:
        if _contains_token(original, certification) and not _contains_upper_token(translated, certification):
            violations.append(f"missing certification {certification}")

    original_numbers = _normalized_number_tokens(original)
    translated_numbers = _normalized_number_tokens(translated)
    for number in original_numbers:
        if number not in translated_numbers:
            violations.append(f"missing or changed number {number}")

    if _contains_phrase(original, "Red Snapper"):
        if not _contains_phrase(translated, "Red Snapper"):
            violations.append("Red Snapper identity changed")
        if re.search(r"\brotbarsch\w*\b", translated, re.IGNORECASE):
            violations.append("Red Snapper was changed to Rotbarsch")

    if re.search(r"\blachsforell\w*\b", original, re.IGNORECASE):
        if not re.search(r"\blachsforell\w*\b", translated, re.IGNORECASE):
            violations.append("Lachsforelle identity changed")
        if re.search(r"\bforelle(?:\s+rot)?\b", translated, re.IGNORECASE) and not re.search(
            r"\blachsforell\w*\b", translated, re.IGNORECASE
        ):
            violations.append("Lachsforelle was generalized to Forelle")
    return violations


def validate_translated_name(row: dict[str, str], translated: str) -> str:
    product_name = normalize_plain_text_response(translated)
    original = source_product_name(row)
    violations = semantic_protection_violations(original, product_name)
    if violations:
        raise SemanticProtectionError("; ".join(violations))
    return product_name


def normalize_split(row: dict[str, str], data: object, source: str) -> dict[str, str]:
    # Accepting a legacy mapping keeps the local API tolerant, but the production
    # DeepSeek call above only returns plain text.
    if isinstance(data, dict):
        data = data.get("product") or ""
    product = validate_translated_name(row, str(data or ""))
    return {
        "brand": verified_local_brand(row),
        "product_name_de": product,
        "product": product,
        "certifications": certifications_json(row),
        "confidence": "",
        "source": source,
    }


def split_row(
    index: int,
    row: dict[str, str],
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[int, dict[str, str]]:
    prompt = build_prompt(row)
    for attempt in range(1, max_retries + 1):
        try:
            data = call_deepseek(api_key, model, base_url, timeout_seconds, prompt)
            return index, normalize_split(row, data, "deepseek")
        except SemanticProtectionError:
            # A second translation attempt could silently choose another incorrect
            # product identity. Preserve the source immediately instead.
            return index, fallback_split(row)
        except Exception:
            if attempt >= max_retries:
                return index, fallback_split(row)
            time.sleep(min(2 ** (attempt - 1), 8))
    return index, fallback_split(row)


def output_fieldnames(fieldnames: list[str]) -> list[str]:
    managed_fields = {
        "product_name_original",
        "product_name_de",
        "brand",
        "product",
        "certifications",
        "brand_product_confidence",
    }
    fields = [name for name in fieldnames if name not in managed_fields]
    if "product_name" in fields:
        pos = fields.index("product_name") + 1
        fields[pos:pos] = [
            "product_name_original",
            "product_name_de",
            "brand",
            "product",
            "certifications",
            "brand_product_confidence",
        ]
        return fields
    return fields + [
        "product_name_original",
        "product_name_de",
        "brand",
        "product",
        "certifications",
        "brand_product_confidence",
    ]


def save_rows(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp_path.replace(output_path)


def run_brand_product_split(
    input_path: Path,
    output_path: Path | None = None,
    workers: int = DEFAULT_WORKERS,
    deepseek_base_url: str | None = None,
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL,
    deepseek_timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    force_refresh: bool = False,
    skip_llm: bool = False,
) -> tuple[Path, int]:
    load_env_file(ROOT / ".env")
    input_path = input_path.resolve()
    output_path = (output_path or input_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if workers < 1:
        raise RuntimeError("--workers must be at least 1")
    if max_retries < 1:
        raise RuntimeError("--max-retries must be at least 1")
    if deepseek_model != DEFAULT_DEEPSEEK_MODEL:
        raise RuntimeError(
            f"Product-name translation must use {DEFAULT_DEEPSEEK_MODEL}; got {deepseek_model!r}."
        )

    fieldnames, rows = load_rows(input_path)
    cache = read_cache()
    results: list[dict[str, str] | None] = [None] * len(rows)
    jobs: list[tuple[int, dict[str, str]]] = []

    for index, row in enumerate(rows):
        if not relevant_row(row):
            results[index] = fallback_split(row)
            continue
        key = cache_key(row)
        cached = cache.get(key)
        if cached and not force_refresh and str(cached.get("model") or "") == deepseek_model:
            try:
                results[index] = normalize_split(
                    row,
                    cached.get("product_name_de") or cached.get("product") or "",
                    "cache",
                )
            except (TranslationFormatError, SemanticProtectionError):
                jobs.append((index, row))
        else:
            jobs.append((index, row))

    api_key = get_env_any("DEEPSEEK_API_KEY", "deepseek_api_key", "Deepseek_api_key")
    if skip_llm or not api_key:
        if not api_key and not skip_llm:
            print("DEEPSEEK_API_KEY missing; preserving source product names.")
        for index, row in jobs:
            results[index] = fallback_split(row)
    elif jobs:
        jobs_by_key: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for index, row in jobs:
            jobs_by_key.setdefault(cache_key(row), []).append((index, row))
        print(
            f"Product-name translation: {len(jobs_by_key)} DeepSeek calls "
            f"for {len(jobs)} rows with {workers} workers"
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    split_row,
                    members[0][0],
                    members[0][1],
                    api_key,
                    deepseek_model,
                    deepseek_base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
                    deepseek_timeout,
                    max_retries,
                ): (key, members)
                for key, members in jobs_by_key.items()
            }
            completed = 0
            for future in as_completed(future_map):
                key, members = future_map[future]
                _idx, representative_split = future.result()
                translated_name = representative_split.get("product_name_de", "")
                result_source = representative_split.get("source", "fallback")
                for index, row in members:
                    if result_source == "fallback":
                        results[index] = fallback_split(row)
                    else:
                        results[index] = normalize_split(row, translated_name, result_source)
                item = {
                    "cache_key": key,
                    "prompt_version": CACHE_PROMPT_VERSION,
                    "model": deepseek_model,
                    "product_name_de": translated_name,
                    "certifications": representative_split.get("certifications", "[]"),
                    "source": result_source,
                }
                append_cache(item)
                completed += 1
                if completed == len(future_map) or completed % workers == 0:
                    print(f"Translated {completed}/{len(future_map)} unique product names")

    for index, row in enumerate(rows):
        split = results[index] or fallback_split(row)
        original_name = source_product_name(row)
        translated_name = split.get("product_name_de", "") or original_name
        row["product_name_original"] = original_name
        row["product_name_de"] = translated_name
        row["brand"] = verified_local_brand(row)
        row["product"] = translated_name
        row["certifications"] = certifications_json(row)
        row["brand_product_confidence"] = ""

    out_fields = output_fieldnames(fieldnames)
    save_rows(output_path, out_fields, rows)
    relevant_count = sum(1 for row in rows if relevant_row(row))
    print(f"Saved translated product names to {output_path} (relevant rows={relevant_count})")
    return output_path, relevant_count


def main() -> None:
    load_env_file(ROOT / ".env")
    args = build_parser().parse_args()
    run_brand_product_split(
        input_path=args.input,
        output_path=args.output,
        workers=args.workers,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_model=args.deepseek_model,
        deepseek_timeout=args.deepseek_timeout,
        max_retries=args.max_retries,
        force_refresh=args.force_refresh,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
