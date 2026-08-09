#!/usr/bin/env python3
"""Check flat JSON/JSONL/CSV results against a versioned policy fixture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "kw32_2026_golden_set.v1.json"
CONDITION_OPS = {"contains", "normalized_equals", "not_regex", "number_equals", "regex"}
ASSERTION_TYPES = {
    "field_distinct",
    "field_equals",
    "field_nonempty",
    "field_not_equals",
    "field_same",
}
DEFAULT_ALIASES = {
    "source_product_name": [
        "source_product_name",
        "product_name_original",
        "original_product_name",
        "raw_product_name",
        "product_name",
    ],
    "normalized_product_name": [
        "normalized_product_name",
        "canonical_product_name",
        "product",
        "Produkt",
        "product_name",
    ],
    "relevance": ["relevance", "relevance_decision", "Relevant", "is_relevant"],
    "family_id": ["product_family_id", "family_id", "product_family"],
    "variant_id": ["variant_id", "product_variant_id"],
    "offer_id": ["offer_id", "source_offer_id"],
}
YES_VALUES = {"1", "ja", "yes", "true", "relevant", "x"}
NO_VALUES = {"0", "nein", "no", "false", "not relevant", "irrelevant"}


class FixtureValidationError(ValueError):
    pass


def load_fixture(path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise FixtureValidationError("Golden-set fixture must be one JSON object.")
    errors = validate_fixture(fixture)
    if errors:
        raise FixtureValidationError("Invalid golden-set fixture: " + "; ".join(errors))
    return fixture


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read flat records from JSON, JSONL/NDJSON, or CSV."""
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records = payload.get("records") if isinstance(payload, dict) and isinstance(payload.get("records"), list) else payload
        if isinstance(records, dict):
            records = [records]
    else:
        raise ValueError("Results must use .json, .jsonl/.ndjson, or .csv")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("Every result record must be a JSON object")
    return [dict(item) for item in records]


def validate_fixture(fixture: dict[str, Any]) -> list[str]:
    """Validate the fixture without requiring the unavailable KW32 raw files."""
    errors: list[str] = []
    for key in ("schema_version", "fixture_id", "fixture_version", "cases"):
        if key not in fixture:
            errors.append(f"missing top-level field {key}")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        return [*errors, "cases must be a non-empty list"]

    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"cases[{index}] must be an object")
            continue
        case_id = str(case.get("case_id") or "").strip()
        label = case_id or f"cases[{index}]"
        if not case_id:
            errors.append(f"{label}: missing case_id")
        elif case_id in seen:
            errors.append(f"duplicate case_id {case_id}")
        seen.add(case_id)

        if case.get("evaluation") not in {"enforced", "review_only"}:
            errors.append(f"{label}: evaluation must be enforced or review_only")
        source = case.get("source_reference")
        if not isinstance(source, dict) or not isinstance(source.get("raw_row_available"), bool):
            errors.append(f"{label}: source_reference must state boolean raw_row_available")

        selector = case.get("selector")
        if not isinstance(selector, dict) or not any(key in selector for key in ("all", "any", "none")):
            errors.append(f"{label}: selector requires all, any, or none")
        else:
            for group in ("all", "any", "none"):
                for condition in selector.get(group, []):
                    if not isinstance(condition, dict) or not (condition.get("field") or condition.get("fields")):
                        errors.append(f"{label}: invalid selector.{group} condition")
                    elif condition.get("op") not in CONDITION_OPS:
                        errors.append(f"{label}: unsupported selector op {condition.get('op')!r}")

        cardinality = case.get("cardinality", {})
        if not isinstance(cardinality, dict) or any(
            not isinstance(cardinality[key], int) or cardinality[key] < 0
            for key in ("min", "max")
            if key in cardinality
        ):
            errors.append(f"{label}: cardinality bounds must be non-negative integers")

        assertions = case.get("assertions")
        if not isinstance(assertions, list):
            errors.append(f"{label}: assertions must be a list")
        elif case.get("evaluation") == "enforced" and not assertions:
            errors.append(f"{label}: enforced cases require assertions")
        else:
            for assertion in assertions:
                if assertion.get("type") not in ASSERTION_TYPES or not assertion.get("field"):
                    errors.append(f"{label}: invalid assertion")
    return errors


def validate_records(
    fixture: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    errors = validate_fixture(fixture)
    if errors:
        raise FixtureValidationError("Invalid golden-set fixture: " + "; ".join(errors))
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every result record must be an object/dict")

    aliases = _aliases(fixture)
    results = []
    for case in fixture["cases"]:
        matched = [row for row in records if _matches(row, case["selector"], aliases)]
        result = {
            "case_id": case["case_id"],
            "issue": case.get("issue", ""),
            "evaluation": case["evaluation"],
            "matched_records": len(matched),
            "errors": [],
        }
        if case["evaluation"] == "review_only":
            result["status"] = "review_only"
            results.append(result)
            continue

        bounds = case.get("cardinality", {})
        minimum = int(bounds.get("min", 1))
        maximum = bounds.get("max")
        if len(matched) < minimum:
            message = f"matched {len(matched)} record(s), expected at least {minimum}"
            if allow_missing:
                result.update(status="not_evaluable", errors=[message])
                results.append(result)
                continue
            result["errors"].append(message)
        if maximum is not None and len(matched) > int(maximum):
            result["errors"].append(f"matched {len(matched)} record(s), expected at most {maximum}")

        if not result["errors"]:
            for assertion in case["assertions"]:
                failure = _assert(assertion, matched, aliases)
                if failure:
                    result["errors"].append(failure)
        result["status"] = "failed" if result["errors"] else "passed"
        results.append(result)

    summary: dict[str, int] = {}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    return {
        "fixture_id": fixture["fixture_id"],
        "fixture_version": fixture["fixture_version"],
        "record_count": len(records),
        "passed": not summary.get("failed", 0),
        "summary": summary,
        "cases": results,
    }


def _aliases(fixture: dict[str, Any]) -> dict[str, list[str]]:
    aliases = {key: list(values) for key, values in DEFAULT_ALIASES.items()}
    for key, values in fixture.get("field_aliases", {}).items():
        if isinstance(values, list):
            aliases[key] = list(dict.fromkeys([*map(str, values), *aliases.get(key, [])]))
    return aliases


def _value(record: dict[str, Any], field: str, aliases: dict[str, list[str]]) -> Any:
    keys = {str(key).strip().casefold(): key for key in record}
    fallback = None
    for candidate in [field, *aliases.get(field, [])]:
        actual = keys.get(str(candidate).strip().casefold())
        if actual is None:
            continue
        fallback = record.get(actual) if fallback is None else fallback
        if not _empty(record.get(actual)):
            return record.get(actual)
    return fallback


def _matches(record: dict[str, Any], selector: dict[str, Any], aliases: dict[str, list[str]]) -> bool:
    checks = lambda group: [_condition(record, item, aliases) for item in selector.get(group, [])]
    all_checks, any_checks, none_checks = checks("all"), checks("any"), checks("none")
    return (
        (not all_checks or all(all_checks))
        and (not any_checks or any(any_checks))
        and not any(none_checks)
    )


def _condition(record: dict[str, Any], condition: dict[str, Any], aliases: dict[str, list[str]]) -> bool:
    fields = condition.get("fields") or [condition.get("field")]
    values = [_value(record, str(field), aliases) for field in fields]
    values = [value for value in values if not _empty(value)]
    op, expected = condition["op"], condition.get("value")
    if op == "not_regex":
        return all(re.search(str(expected), str(value), re.IGNORECASE) is None for value in values)
    return any(_compare(value, op, expected) for value in values)


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "normalized_equals":
        return _text(actual) == _text(expected)
    if op == "contains":
        return _text(expected) in _text(actual)
    if op == "regex":
        return re.search(str(expected), str(actual), re.IGNORECASE) is not None
    if op == "number_equals":
        left, right = _number(actual), _number(expected)
        return left is not None and right is not None and math.isclose(left, right, abs_tol=1e-9)
    raise ValueError(f"Unsupported condition op: {op}")


def _assert(assertion: dict[str, Any], records: list[dict[str, Any]], aliases: dict[str, list[str]]) -> str | None:
    kind, field = assertion["type"], assertion["field"]
    values = [_value(record, field, aliases) for record in records]
    normalized = [_canonical(field, value) for value in values]
    if kind in {"field_same", "field_distinct"}:
        if len(values) < 2:
            return f"{field}: {kind} requires at least two records"
        if any(_empty(value) for value in values):
            return f"{field}: expected non-empty values, got {values!r}"
        good = len(set(normalized)) == (1 if kind == "field_same" else len(values))
        if not good:
            expectation = "one shared" if kind == "field_same" else "distinct"
            return f"{field}: expected {expectation} values, got {values!r}"
        return None

    expected = _canonical(field, assertion.get("value"))
    if kind == "field_nonempty":
        good = all(not _empty(value) for value in values)
    elif kind == "field_equals":
        good = all(not _empty(value) and actual == expected for value, actual in zip(values, normalized))
    elif kind == "field_not_equals":
        good = all(not _empty(value) and actual != expected for value, actual in zip(values, normalized))
    else:  # fixture validation prevents this path
        raise ValueError(f"Unsupported assertion: {kind}")
    return None if good else f"{field}: {kind} expected {assertion.get('value')!r}, got {values!r}"


def _canonical(field: str, value: Any) -> str:
    normalized = _text(value)
    if field == "relevance":
        if isinstance(value, bool):
            return "yes" if value else "no"
        if normalized in YES_VALUES:
            return "yes"
        if normalized in NO_VALUES:
            return "no"
    return normalized


def _text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)).strip()


def _number(value: Any) -> float | None:
    text = str(value or "").strip().replace("€", "").replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip().casefold() in {"", "none", "null", "nan"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--results", type=Path, help="Optional .json/.jsonl/.ndjson/.csv result file")
    parser.add_argument("--allow-missing", action="store_true", help="Mark absent cases not_evaluable")
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.fixture)
        report = (
            validate_records(fixture, load_records(args.results), allow_missing=args.allow_missing)
            if args.results
            else {
                "fixture_id": fixture["fixture_id"],
                "fixture_version": fixture["fixture_version"],
                "fixture_valid": True,
                "case_count": len(fixture["cases"]),
                "results_validated": False,
            }
        )
    except (FixtureValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed", report.get("fixture_valid", False)) else 1


if __name__ == "__main__":
    sys.exit(main())
