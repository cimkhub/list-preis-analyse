#!/usr/bin/env python3
"""Split product_name into brand and product columns using DeepSeek when needed."""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
CACHE_PATH = ROOT / "embeddings" / "product_matching" / "brand_product_splits.jsonl"
RELEVANT_VALUES = {"yes", "ja", "true", "1", "relevant", "x"}
_THREAD_LOCAL = local()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add brand and product columns to a relevant supplier CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Defaults to overwriting --input safely.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--deepseek-base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL))
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
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


def cache_key(row: dict[str, str]) -> str:
    parts = [
        row.get("product_name", ""),
        row.get("description", ""),
        row.get("category", ""),
        row.get("supplier", ""),
    ]
    return "||".join(parts).casefold()


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


def fallback_split(row: dict[str, str]) -> dict[str, str]:
    product_name = title_case_product(row.get("product_name", ""))
    return {
        "brand": "",
        "product": product_name,
        "confidence": "55",
        "source": "fallback",
    }


def build_prompt(row: dict[str, str]) -> str:
    payload = {
        "product_name": row.get("product_name", ""),
        "description": row.get("description", ""),
        "category": row.get("category", ""),
        "supplier": row.get("supplier", ""),
    }
    return "\n".join([
        "Split the wholesaler product name into brand and product.",
        "Only put a value in brand if the text clearly contains a real manufacturer or product brand.",
        "If there is no clear brand, brand must be an empty string.",
        "The product field must contain the actual product without the brand.",
        "Do not invent brands. Do not treat generic product descriptors as brands.",
        "Keep German product wording, but normalize capitalization to Title Case.",
        "",
        "Examples:",
        '{"product_name":"Adelholzener Active O2"} -> {"brand":"Adelholzener","product":"Active O2","confidence":95}',
        '{"product_name":"Spargel Weiß"} -> {"brand":"","product":"Spargel Weiß","confidence":95}',
        '{"product_name":"The Duke Of Berkshire Schweinenacken"} -> {"brand":"The Duke Of Berkshire","product":"Schweinenacken","confidence":95}',
        '{"product_name":"Edeka Foodservice Classic Haltbare Sahne"} -> {"brand":"Edeka Foodservice Classic","product":"Haltbare Sahne","confidence":95}',
        "",
        "Return strict JSON only:",
        '{"brand":"","product":"","confidence":0}',
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
) -> dict[str, Any]:
    response = get_session().post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You split product names. Return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 160,
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("DeepSeek response has no content")
    return safe_json_loads(content)


def safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def normalize_split(row: dict[str, str], data: dict[str, Any], source: str) -> dict[str, str]:
    brand = title_case_product(str(data.get("brand") or "").strip())
    product = title_case_product(str(data.get("product") or "").strip())
    if not product:
        product = title_case_product(row.get("product_name", ""))
    confidence = str(int(float(data.get("confidence") or 0)))
    return {
        "brand": brand,
        "product": product,
        "confidence": confidence,
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
        except Exception:
            if attempt >= max_retries:
                return index, fallback_split(row)
            time.sleep(min(2 ** (attempt - 1), 8))
    return index, fallback_split(row)


def output_fieldnames(fieldnames: list[str]) -> list[str]:
    fields = [name for name in fieldnames if name not in {"brand", "product", "brand_product_confidence"}]
    if "product_name" in fields:
        pos = fields.index("product_name") + 1
        fields[pos:pos] = ["brand", "product", "brand_product_confidence"]
        return fields
    return fields + ["brand", "product", "brand_product_confidence"]


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
            results[index] = {
                "brand": cached.get("brand", ""),
                "product": cached.get("product", "") or title_case_product(row.get("product_name", "")),
                "confidence": cached.get("confidence", ""),
                "source": cached.get("source", "cache"),
            }
        else:
            jobs.append((index, row))

    api_key = get_env_any("DEEPSEEK_API_KEY", "deepseek_api_key", "Deepseek_api_key")
    if skip_llm or not api_key:
        if not api_key and not skip_llm:
            print("DEEPSEEK_API_KEY missing; using fallback brand/product split.")
        for index, row in jobs:
            results[index] = fallback_split(row)
    elif jobs:
        print(f"Brand/product split: {len(jobs)} DeepSeek calls with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    split_row,
                    index,
                    row,
                    api_key,
                    deepseek_model,
                    deepseek_base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
                    deepseek_timeout,
                    max_retries,
                ): (index, row)
                for index, row in jobs
            }
            completed = 0
            for future in as_completed(future_map):
                index, row = future_map[future]
                _idx, split = future.result()
                results[index] = split
                item = {"cache_key": cache_key(row), **split}
                item["model"] = deepseek_model
                append_cache(item)
                completed += 1
                if completed == len(future_map) or completed % workers == 0:
                    print(f"Split {completed}/{len(future_map)} product names")

    for index, row in enumerate(rows):
        split = results[index] or fallback_split(row)
        row["product_name"] = title_case_product(row.get("product_name", ""))
        row["brand"] = split.get("brand", "")
        row["product"] = split.get("product", "") or row["product_name"]
        row["brand_product_confidence"] = split.get("confidence", "")

    out_fields = output_fieldnames(fieldnames)
    save_rows(output_path, out_fields, rows)
    relevant_count = sum(1 for row in rows if relevant_row(row))
    print(f"Saved brand/product split to {output_path} (relevant rows={relevant_count})")
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
