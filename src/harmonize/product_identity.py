"""Stable product-family, variant and offer identity for competitor observations.

The helpers in this module are deliberately pure and network-free. Semantic LLM
judgements may propose that two observations describe the same product, but the
structural rules here are authoritative and are re-checked for every complete
cross-cluster merge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_SCHEMA_VERSION = 2
UNKNOWN_VALUES = {"", "unknown", "unbekannt", "none", "null", "n/a", "na"}
CERTIFICATIONS = {"asc", "msc", "qs", "bio", "halal"}
NON_FROZEN_STATES = {"fresh", "chilled", "thawed"}
FROZEN_MARKERS = ("tiefgek", "gefroren", "frozen", " tiefkühl", " tiefkuehl", " tk ")
NON_FROZEN_MARKERS = ("frisch", "fresh", "gekühlt", "gekuehlt", "chilled", "aufgetaut", "thawed")

PACKAGED_POLICY_GROUPS = {
    "oil",
    "öl",
    "oel",
    "cream",
    "sahne",
    "quark",
    "cheese",
    "käse",
    "kaese",
    "sausage",
    "wurst",
    "frozen vegetables",
    "tiefkühl gemüse",
    "tiefkuehl gemuese",
    "french fries",
    "pommes",
    "milk",
    "milch",
}

PRODUCT_LINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("duroc", re.compile(r"\bduroc\b", re.IGNORECASE)),
    ("black_angus", re.compile(r"\bblack[\s-]+angus\b", re.IGNORECASE)),
    ("angus", re.compile(r"\bangus\b", re.IGNORECASE)),
    ("free_range", re.compile(r"\bfree[\s-]+range\b", re.IGNORECASE)),
    ("wagyu", re.compile(r"\bwagyu\b", re.IGNORECASE)),
    ("iberico", re.compile(r"\bib[eé]rico\b", re.IGNORECASE)),
    ("hereford", re.compile(r"\bhereford\b", re.IGNORECASE)),
    ("label_rouge", re.compile(r"\blabel[\s-]+rouge\b", re.IGNORECASE)),
    ("kikok", re.compile(r"\bkikok\b", re.IGNORECASE)),
)

COUNTRY_ALIASES = {
    "de": "DE", "deutschland": "DE", "germany": "DE",
    "es": "ES", "spanien": "ES", "spain": "ES",
    "fr": "FR", "frankreich": "FR", "france": "FR",
    "it": "IT", "italien": "IT", "italy": "IT",
    "nl": "NL", "niederlande": "NL", "holland": "NL", "netherlands": "NL",
    "be": "BE", "belgien": "BE", "belgium": "BE",
    "pl": "PL", "polen": "PL", "poland": "PL",
    "pt": "PT", "portugal": "PT",
    "at": "AT", "österreich": "AT", "oesterreich": "AT", "austria": "AT",
    "dk": "DK", "dänemark": "DK", "daenemark": "DK", "denmark": "DK",
    "no": "NO", "norwegen": "NO", "norway": "NO",
    "se": "SE", "schweden": "SE", "sweden": "SE",
    "ie": "IE", "irland": "IE", "ireland": "IE",
    "gb": "GB", "uk": "GB", "großbritannien": "GB", "grossbritannien": "GB",
    "gr": "GR", "griechenland": "GR", "greece": "GR",
    "tr": "TR", "türkei": "TR", "tuerkei": "TR", "turkey": "TR",
    "ro": "RO", "rumänien": "RO", "rumaenien": "RO", "romania": "RO",
    "ua": "UA", "ukraine": "UA",
    "hu": "HU", "ungarn": "HU", "hungary": "HU",
    "cz": "CZ", "tschechien": "CZ", "czechia": "CZ",
    "ma": "MA", "marokko": "MA", "morocco": "MA",
    "za": "ZA", "südafrika": "ZA", "suedafrika": "ZA", "south africa": "ZA",
    "ar": "AR", "argentinien": "AR", "argentina": "AR",
    "uy": "UY", "uruguay": "UY",
    "br": "BR", "brasilien": "BR", "brazil": "BR",
    "cl": "CL", "chile": "CL",
    "pe": "PE", "peru": "PE",
    "nz": "NZ", "neuseeland": "NZ", "new zealand": "NZ",
    "au": "AU", "australien": "AU", "australia": "AU",
    "us": "US", "usa": "US", "vereinigte staaten": "US", "united states": "US",
    "ca": "CA", "kanada": "CA", "canada": "CA",
    "cn": "CN", "china": "CN",
    "vn": "VN", "vietnam": "VN",
    "in": "IN", "indien": "IN", "india": "IN",
    "eu": "EU", "europa": "EU", "european union": "EU",
}

OBSERVATION_FIELDS = (
    "supplier", "location", "description", "category", "product_family", "temperature_state",
    "processing_state", "origin", "calibre", "source_brand", "brand_evidence",
    "brand_evidence_source", "certifications", "unit", "quantity",
    "price", "price_per_kg", "price_is_net", "price_gross", "price_tiers", "price_basis",
    "package_count", "package_size_value", "package_size_unit", "total_content_value",
    "total_content_unit", "packaging_type", "packaging_raw", "valid_from", "valid_to",
    "calendar_week", "year", "source_file", "source_title", "source_tab", "source_page",
    "source_document_sha256", "source_item_id",
)

OFFER_FIELDS = (
    "supplier", "location", "price", "price_per_kg", "price_is_net", "price_gross",
    "price_tiers", "price_basis", "package_count",
    "package_size_value", "package_size_unit", "total_content_value", "total_content_unit",
    "packaging_type", "calibre", "valid_from", "valid_to",
    "calendar_week", "year",
)

STRUCTURED_PACKAGE_FIELDS = (
    "package_count",
    "package_size_value",
    "package_size_unit",
    "total_content_value",
    "total_content_unit",
    "packaging_type",
)


@dataclass(frozen=True)
class OriginEvidence:
    values: tuple[str, ...] = ()
    raw: str = ""

    @property
    def known(self) -> bool:
        return bool(self.values)


@dataclass(frozen=True)
class IdentityEvidence:
    observation_id: str
    family_key: str
    protected_kind: str
    temperature_group: str
    origin: OriginEvidence
    product_line: str
    packaged_group: bool
    brand_key: str
    calibre: str


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class OfferRecord:
    offer_id: str
    variant_id: str
    product: dict[str, Any]
    observation_ids: list[str]
    source_refs: list[dict[str, str]]


def _value(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("ß", "ss")
    return re.sub(r"[^a-z0-9äöü]+", " ", text).strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _normalized_json_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    return canonical_json(parsed)


def _normalized_number(value: Any) -> str:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return str(value or "").strip()
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def observation_payload(row: Mapping[str, Any]) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "product_name_original": (
            _value(row, "product_name_original") or _value(row, "product_name")
        ).strip(),
    }
    for field in OBSERVATION_FIELDS:
        value = _value(row, field).strip()
        if field in {
            "price", "price_per_kg", "price_gross", "quantity", "package_count",
            "package_size_value", "total_content_value",
        }:
            value = _normalized_number(value)
        elif field in {"price_tiers", "certifications"}:
            value = _normalized_json_text(value)
        if field == "source_file" and _value(row, "source_document_sha256").strip():
            value = ""
        payload[field] = value
    return payload


def make_observation_id(row: Mapping[str, Any]) -> str:
    return stable_id("obs", observation_payload(row))


def dedupe_observation_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, str]] = []
    for raw_row in rows:
        row = dict(raw_row)
        observation_id = make_observation_id(row)
        row["observation_id"] = observation_id
        row["product_id"] = observation_id  # compatibility with existing caches/APIs
        if observation_id in kept:
            kept_row = kept[observation_id]
            audit.append({
                "removed_observation_id": observation_id,
                "kept_observation_id": observation_id,
                "reason": "exact_observation_duplicate",
                "removed_source_row_index": _value(row, "source_row_index"),
                "kept_source_row_index": _value(kept_row, "source_row_index"),
            })
            continue
        kept[observation_id] = row
    return [kept[key] for key in sorted(kept)], audit


def source_name_candidates(row: Mapping[str, Any]) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for priority, field in enumerate(
        ("product_name_de", "product", "product_name_original", "product_name"),
    ):
        value = _value(row, field).strip()
        key = normalize_text(value)
        if value and key and key not in seen and source_name_is_safe(row, value):
            candidates.append((priority, value))
            seen.add(key)
    return candidates


def source_name_is_safe(row: Mapping[str, Any], candidate: str) -> bool:
    original = _value(row, "product_name_original") or _value(row, "product_name")
    original_norm = normalize_text(original)
    candidate_norm = normalize_text(candidate)
    if "red snapper" in original_norm:
        return "red snapper" in candidate_norm and "rotbarsch" not in candidate_norm
    if "lachsforell" in original_norm:
        return "lachsforell" in candidate_norm
    return True


def preferred_source_name(row: Mapping[str, Any]) -> str:
    candidates = source_name_candidates(row)
    if candidates:
        return candidates[0][1]
    return (_value(row, "product_name_original") or _value(row, "product_name")).strip()


def protected_product_kind(value: Any) -> str:
    text = normalize_text(value)
    if "red snapper" in text:
        return "red_snapper"
    if "rotbarsch" in text:
        return "rotbarsch"
    if "lachsforell" in text:
        return "lachsforelle"
    if re.search(r"\bforell\w*\b", text):
        return "forelle"
    return ""


def _remove_normalized_phrase(text: str, phrase: Any) -> str:
    normalized = normalize_text(phrase)
    if not normalized:
        return text
    return re.sub(rf"(?:^|\s){re.escape(normalized)}(?:$|\s)", " ", f" {text} ").strip()


def semantic_family_key(row: Mapping[str, Any], attributes: Mapping[str, Any] | None = None) -> str:
    name = preferred_source_name(row)
    text = normalize_text(name)
    if not text:
        text = normalize_text((attributes or {}).get("base_product"))

    for brand_field in ("relevance_brand", "brand", "source_brand"):
        text = _remove_normalized_phrase(text, _value(row, brand_field))
    for _line, pattern in PRODUCT_LINE_PATTERNS:
        text = normalize_text(pattern.sub(" ", text))

    text = re.sub(
        r"\b(?:asc|msc|qs|bio|halal|tk|tiefgekühlt|tiefgekuehlt|gefroren|frozen|"
        r"frisch\w*|fresh|gekühlt|gekuehlt|chilled|aufgetaut|thawed)\b",
        " ",
        text,
    )
    text = re.sub(r"\b\d+(?:[.,]\d+)?(?:\s*[/\-]\s*\d+(?:[.,]\d+)?)?\s*%?\b", " ", text)
    text = re.sub(
        r"\b(?:g|kg|gramm|kilogramm|ml|l|liter|stück|stueck|packung|beutel|karton|"
        r"kiste|korb|schale|box|eimer|flasche)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()

    singular_aliases = {
        "pfifferlinge": "pfifferling",
        "gemüsezwiebeln": "gemüsezwiebel",
        "gemuesezwiebeln": "gemuesezwiebel",
        "karotten": "karotte",
        "möhren": "karotte",
        "moehren": "karotte",
        "möhre": "karotte",
        "moehre": "karotte",
        "forellen": "forelle",
    }
    tokens = [singular_aliases.get(token, token) for token in text.split()]
    # "Karotte/Möhre" becomes one semantic key rather than two aliases.
    tokens = list(dict.fromkeys(tokens))
    return " ".join(tokens) or "unknown"


def normalize_temperature(row: Mapping[str, Any], attributes: Mapping[str, Any] | None = None) -> str:
    raw = normalize_text(_value(row, "temperature_state"))
    if raw in NON_FROZEN_STATES:
        return "non_frozen"
    if raw == "frozen":
        return "frozen"
    attr_raw = normalize_text((attributes or {}).get("fresh_or_frozen"))
    if attr_raw in NON_FROZEN_STATES or attr_raw == "fresh":
        return "non_frozen"
    if attr_raw == "frozen":
        return "frozen"
    text = f" {normalize_text(preferred_source_name(row))} {normalize_text(_value(row, 'description'))} "
    if any(marker in text for marker in FROZEN_MARKERS):
        return "frozen"
    if any(marker in text for marker in NON_FROZEN_MARKERS):
        return "non_frozen"
    return "unknown"


def normalize_origin(value: Any) -> OriginEvidence:
    raw = str(value or "").strip()
    normalized = normalize_text(raw)
    if normalized in UNKNOWN_VALUES:
        return OriginEvidence(raw=raw)
    parts = [
        normalize_text(part)
        for part in re.split(r"[,;/|+&]+|\b(?:und|oder|and|or)\b", raw, flags=re.IGNORECASE)
        if normalize_text(part)
    ]
    values: set[str] = set()
    for part in parts or [normalized]:
        cleaned = re.sub(r"^(?:aus|herkunft|origin|von)\s+", "", part).strip()
        code = COUNTRY_ALIASES.get(cleaned)
        if code:
            values.add(code)
            continue
        matches = {
            alias_code
            for alias, alias_code in COUNTRY_ALIASES.items()
            if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", cleaned)
        }
        if matches:
            values.update(matches)
        elif cleaned not in UNKNOWN_VALUES:
            values.add(f"TEXT:{cleaned}")
    return OriginEvidence(tuple(sorted(values)), raw)


def detect_product_line(row: Mapping[str, Any]) -> str:
    text = " ".join(
        _value(row, field)
        for field in (
            "product_name", "product_name_original", "product_name_de", "product",
            "description", "brand_evidence",
        )
    )
    text = f"{text} {normalized_brand(row)}"
    for line, pattern in PRODUCT_LINE_PATTERNS:
        if pattern.search(text):
            return line
    return "generic" if preferred_source_name(row) else "unknown"


def is_packaged_group(row: Mapping[str, Any]) -> bool:
    route = normalize_text(_value(row, "relevance_route"))
    if route == "packaged exception":
        return True
    group = normalize_text(_value(row, "relevance_policy_group"))
    if group in PACKAGED_POLICY_GROUPS:
        return True
    category = normalize_text(_value(row, "category"))
    return category in {"mopro", "wurst"}


def normalized_brand(row: Mapping[str, Any]) -> str:
    value = _value(row, "relevance_brand").strip() or _value(row, "brand").strip()
    if not value:
        source_brand = _value(row, "source_brand").strip()
        source_brand_key = normalize_text(source_brand)
        evidence = normalize_text(_value(row, "brand_evidence"))
        source = _value(row, "brand_evidence_source").strip().casefold()
        evidence_field = normalize_text(_value(row, source)) if source in {"product_name", "description"} else ""
        if source == "image" and source_brand and evidence:
            value = source_brand
        elif (
            source in {"product_name", "description"}
            and source_brand_key
            and source_brand_key in evidence
            and evidence in evidence_field
        ):
            value = source_brand
    normalized = normalize_text(value)
    return "" if normalized in CERTIFICATIONS or normalized in UNKNOWN_VALUES else normalized


def build_identity_evidence(
    row: Mapping[str, Any],
    attributes: Mapping[str, Any] | None = None,
) -> IdentityEvidence:
    observation_id = _value(row, "observation_id") or _value(row, "product_id") or make_observation_id(row)
    name = preferred_source_name(row)
    return IdentityEvidence(
        observation_id=observation_id,
        family_key=semantic_family_key(row, attributes),
        protected_kind=protected_product_kind(name),
        temperature_group=normalize_temperature(row, attributes),
        origin=normalize_origin(_value(row, "origin") or (attributes or {}).get("origin")),
        product_line=detect_product_line(row),
        packaged_group=is_packaged_group(row),
        brand_key=normalized_brand(row),
        calibre=_value(row, "calibre") or str((attributes or {}).get("calibre") or ""),
    )


def family_compatibility(a: IdentityEvidence, b: IdentityEvidence) -> CompatibilityResult:
    protected_conflicts = {
        frozenset({"red_snapper", "rotbarsch"}),
        frozenset({"lachsforelle", "forelle"}),
    }
    if frozenset({a.protected_kind, b.protected_kind}) in protected_conflicts:
        return CompatibilityResult(False, ("protected product identity differs",))
    if a.family_key == b.family_key:
        return CompatibilityResult(True)
    return CompatibilityResult(True, warnings=("family core requires semantic judgement",))


def variant_compatibility(a: IdentityEvidence, b: IdentityEvidence) -> CompatibilityResult:
    family = family_compatibility(a, b)
    if not family.compatible:
        return family
    reasons: list[str] = []
    warnings: list[str] = list(family.warnings)
    if {a.temperature_group, b.temperature_group} == {"frozen", "non_frozen"}:
        reasons.append("frozen vs non-frozen")
    elif "unknown" in {a.temperature_group, b.temperature_group} and a.temperature_group != b.temperature_group:
        warnings.append("temperature missing on one side")

    if a.origin.known and b.origin.known and a.origin.values != b.origin.values:
        reasons.append("known origin differs")
    elif a.origin.known != b.origin.known:
        warnings.append("origin missing on one side")

    if a.product_line not in {"", "unknown"} and b.product_line not in {"", "unknown"}:
        if a.product_line != b.product_line:
            reasons.append("product line differs")
    elif a.product_line != b.product_line:
        warnings.append("product line missing on one side")

    if a.packaged_group and b.packaged_group and a.brand_key and b.brand_key and a.brand_key != b.brand_key:
        reasons.append("known packaged-product brand differs")
    elif (a.packaged_group or b.packaged_group) and bool(a.brand_key) != bool(b.brand_key):
        warnings.append("brand missing on one side")

    return CompatibilityResult(not reasons, tuple(reasons), tuple(dict.fromkeys(warnings)))


def structural_compatibility(
    row_a: Mapping[str, Any],
    row_b: Mapping[str, Any],
    attr_a: Mapping[str, Any] | None = None,
    attr_b: Mapping[str, Any] | None = None,
) -> CompatibilityResult:
    return variant_compatibility(
        build_identity_evidence(row_a, attr_a),
        build_identity_evidence(row_b, attr_b),
    )


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _merge_components(
    members: Sequence[str],
    edges: Sequence[tuple[float, str, str]],
    can_merge,
) -> list[list[str]]:
    clusters: dict[str, set[str]] = {member: {member} for member in sorted(members)}
    owner = {member: member for member in members}
    for _score, a, b in sorted(edges, key=lambda item: (-item[0], item[1], item[2])):
        root_a, root_b = owner[a], owner[b]
        if root_a == root_b:
            continue
        cluster_a, cluster_b = clusters[root_a], clusters[root_b]
        if not can_merge(cluster_a, cluster_b):
            continue
        merged = cluster_a | cluster_b
        new_root = min(merged)
        del clusters[root_a]
        del clusters[root_b]
        clusters[new_root] = merged
        for member in merged:
            owner[member] = new_root
    return [sorted(cluster) for _root, cluster in sorted(clusters.items())]


def make_product_family_id(family_keys: Iterable[str]) -> str:
    normalized_keys = sorted({normalize_text(key) or "unknown" for key in family_keys})
    return stable_id("pf", {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        # A family can contain source-backed aliases accepted by the pair judge.
        # Keeping the sorted alias set makes the ID independent of input/edge order.
        "family_keys": normalized_keys,
    })


def _variant_signature(product_family_id: str, evidence: Sequence[IdentityEvidence]) -> dict[str, Any]:
    temperatures = sorted({item.temperature_group for item in evidence if item.temperature_group != "unknown"})
    origins = sorted({item.origin.values for item in evidence if item.origin.known})
    lines = sorted({item.product_line for item in evidence if item.product_line not in {"", "unknown"}})
    brands = sorted({item.brand_key for item in evidence if item.packaged_group and item.brand_key})
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "product_family_id": product_family_id,
        "temperature": temperatures[0] if len(temperatures) == 1 else temperatures,
        "origin": list(origins[0]) if len(origins) == 1 else [list(value) for value in origins],
        "product_line": lines[0] if len(lines) == 1 else lines,
        "brand": brands[0] if len(brands) == 1 else brands,
    }


def make_variant_id(product_family_id: str, evidence: Sequence[IdentityEvidence]) -> str:
    return stable_id("pv", _variant_signature(product_family_id, evidence))


def deterministic_canonical_name(rows: Sequence[Mapping[str, Any]]) -> str:
    candidates: list[tuple[str, int, str]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        for priority, value in source_name_candidates(row):
            key = normalize_text(value)
            counts[key] += 1
            candidates.append((key, priority, value))
    if not candidates:
        return "Unknown"
    candidates.sort(key=lambda item: (-counts[item[0]], item[1], len(item[2]), item[0], item[2]))
    return candidates[0][2]


def build_identity_clusters(
    rows: Sequence[Mapping[str, Any]],
    attributes: Mapping[str, Mapping[str, Any]],
    accepted_pairs: Mapping[tuple[str, str], float] | None = None,
) -> list[dict[str, Any]]:
    accepted_pairs = dict(accepted_pairs or {})
    row_by_id = {_value(row, "product_id"): dict(row) for row in rows}
    evidence = {
        product_id: build_identity_evidence(row, attributes.get(product_id, {}))
        for product_id, row in row_by_id.items()
    }
    ids = sorted(row_by_id)

    family_edges: list[tuple[float, str, str]] = []
    for index, a in enumerate(ids):
        for b in ids[index + 1:]:
            if evidence[a].family_key == evidence[b].family_key:
                family_edges.append((101.0, a, b))
            elif _pair_key(a, b) in accepted_pairs:
                family_edges.append((accepted_pairs[_pair_key(a, b)], a, b))

    def family_clusters_compatible(cluster_a: set[str], cluster_b: set[str]) -> bool:
        for a in cluster_a:
            for b in cluster_b:
                if not family_compatibility(evidence[a], evidence[b]).compatible:
                    return False
                if evidence[a].family_key != evidence[b].family_key and _pair_key(a, b) not in accepted_pairs:
                    return False
        return True

    family_clusters = _merge_components(ids, family_edges, family_clusters_compatible)
    output: list[dict[str, Any]] = []
    for family_members in family_clusters:
        product_family_id = make_product_family_id(evidence[item].family_key for item in family_members)
        variant_edges = [
            (101.0, a, b)
            for index, a in enumerate(family_members)
            for b in family_members[index + 1:]
            if variant_compatibility(evidence[a], evidence[b]).compatible
        ]

        def variant_clusters_compatible(cluster_a: set[str], cluster_b: set[str]) -> bool:
            return all(
                variant_compatibility(evidence[a], evidence[b]).compatible
                for a in cluster_a
                for b in cluster_b
            )

        variant_clusters = _merge_components(family_members, variant_edges, variant_clusters_compatible)
        for members in variant_clusters:
            member_evidence = [evidence[item] for item in members]
            variant_id = make_variant_id(product_family_id, member_evidence)
            warnings = sorted({
                warning
                for index, a in enumerate(members)
                for b in members[index + 1:]
                for warning in variant_compatibility(evidence[a], evidence[b]).warnings
            })
            output.append({
                "canonical_product_id": variant_id,
                "product_family_id": product_family_id,
                "variant_id": variant_id,
                "product_ids": members,
                "canonical_product_name": deterministic_canonical_name([row_by_id[item] for item in members]),
                "identity_warnings": warnings,
            })
    return sorted(output, key=lambda item: (item["product_family_id"], item["variant_id"]))


def offer_payload(variant_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "variant_id": variant_id,
    }
    for field in OFFER_FIELDS:
        if field == "supplier":
            value = _value(row, "supplier_norm").strip() or _value(row, "supplier").strip()
            value = normalize_text(value)
        else:
            value = _value(row, field).strip()
        if field in {
            "price", "price_per_kg", "price_gross", "quantity", "package_count",
            "package_size_value", "total_content_value",
        }:
            value = _normalized_number(value)
        elif field == "price_tiers":
            value = _normalized_json_text(value)
        elif field in {
            "location", "price_basis", "unit", "package_size_unit", "total_content_unit",
            "packaging_type", "packaging_raw", "calibre",
        }:
            value = normalize_text(value)
        payload[field] = value
    structured_package_complete = all(
        _value(row, field).strip()
        and normalize_text(_value(row, field)) not in UNKNOWN_VALUES
        for field in STRUCTURED_PACKAGE_FIELDS
    )
    if not structured_package_complete:
        # Historical rows have only the ambiguous unit/quantity pair. Raw text
        # is an audit/fallback input, but does not churn IDs once the complete
        # structured package contract is present.
        payload["legacy_unit"] = normalize_text(_value(row, "unit"))
        payload["legacy_quantity"] = _normalized_number(_value(row, "quantity"))
        payload["packaging_raw_fallback"] = normalize_text(
            _value(row, "packaging_raw")
        )
    return payload


def make_offer_id(variant_id: str, row: Mapping[str, Any]) -> str:
    return stable_id("of", offer_payload(variant_id, row))


def source_reference(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_file": _value(row, "source_file"),
        "source_page": _value(row, "source_page"),
        "source_title": _value(row, "source_title"),
        "source_item_index": _value(row, "source_item_index"),
        "source_tab": _value(row, "source_tab"),
        "source_document_sha256": _value(row, "source_document_sha256"),
        "source_item_id": _value(row, "source_item_id"),
    }


def build_offer_records(variant_id: str, rows: Sequence[Mapping[str, Any]]) -> list[OfferRecord]:
    records: dict[str, OfferRecord] = {}
    ordered_rows = sorted(
        (dict(raw_row) for raw_row in rows),
        key=lambda row: (
            _value(row, "observation_id")
            or _value(row, "product_id")
            or make_observation_id(row)
        ),
    )
    for row in ordered_rows:
        offer_id = make_offer_id(variant_id, row)
        observation_id = _value(row, "observation_id") or _value(row, "product_id")
        reference = source_reference(row)
        if offer_id not in records:
            records[offer_id] = OfferRecord(
                offer_id=offer_id,
                variant_id=variant_id,
                product=row,
                observation_ids=[observation_id] if observation_id else [],
                source_refs=[reference],
            )
            continue
        record = records[offer_id]
        if observation_id and observation_id not in record.observation_ids:
            record.observation_ids.append(observation_id)
        if reference not in record.source_refs:
            record.source_refs.append(reference)
    for record in records.values():
        record.observation_ids.sort()
        record.source_refs.sort(key=canonical_json)
    return [records[key] for key in sorted(records)]


def _structured_amount_sort_key(row: Mapping[str, Any]) -> tuple[int, int, Decimal, str]:
    value_text = _value(row, "total_content_value").strip()
    unit = normalize_text(_value(row, "total_content_unit"))
    if value_text and unit not in UNKNOWN_VALUES:
        try:
            value = Decimal(value_text.replace(",", "."))
        except InvalidOperation:
            value = Decimal("Infinity")
        if unit == "kg":
            return (0, 0, value * 1000, "g")
        if unit == "g":
            return (0, 0, value, "g")
        if unit in {"l", "liter"}:
            return (0, 1, value * 1000, "ml")
        if unit == "ml":
            return (0, 1, value, "ml")
        if unit in {"piece", "stueck", "stück"}:
            return (0, 2, value, "piece")

    legacy_value = _normalized_number(_value(row, "quantity"))
    try:
        legacy_number = Decimal(legacy_value) if legacy_value else Decimal("Infinity")
    except InvalidOperation:
        legacy_number = Decimal("Infinity")
    return (1, 9, legacy_number, normalize_text(_value(row, "unit")))


def offer_sort_key(record: OfferRecord) -> tuple[Any, ...]:
    row = record.product
    return (
        _value(row, "valid_from"),
        _value(row, "valid_to"),
        *_structured_amount_sort_key(row),
        normalize_text(_value(row, "packaging_type")),
        _value(row, "price"),
        record.offer_id,
    )
