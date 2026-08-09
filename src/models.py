from datetime import date
import json
import math

from pydantic import BaseModel, Field, field_validator


PRODUCT_FAMILIES = frozenset(
    {"fleisch", "fisch", "obst_gemuese", "mopro", "wurst", "sonstiges", "unknown"}
)
TEMPERATURE_STATES = frozenset(
    {"fresh", "chilled", "frozen", "thawed", "ambient", "unknown"}
)
PROCESSING_STATES = frozenset(
    {
        "raw_plain",
        "raw_cut",
        "raw_minced",
        "raw_formed",
        "raw_skewered",
        "raw_seasoned",
        "marinated",
        "sauced",
        "cooked",
        "fried",
        "smoked",
        "cured",
        "pickled",
        "preserved",
        "ready_to_eat",
        "unknown",
    }
)
BRAND_EVIDENCE_SOURCES = frozenset(
    {"product_name", "description", "image", "unknown"}
)
PRICE_BASES = frozenset(
    {
        "per_kg",
        "per_100g",
        "per_liter",
        "per_100ml",
        "per_piece",
        "per_package",
        "unknown",
    }
)
CONTENT_UNITS = frozenset({"g", "kg", "ml", "l", "piece", "unknown"})
PACKAGING_TYPES = frozenset(
    {
        "bag",
        "pack",
        "box",
        "crate",
        "basket",
        "tray",
        "bucket",
        "bottle",
        "can",
        "bundle",
        "piece",
        "unknown",
    }
)
QUALITY_RETRY_STATUSES = frozenset(
    {"not_needed", "selected", "kept_primary", "failed"}
)
EXTRACTION_SCHEMA_VERSION = 2

_PRICE_BASIS_ALIASES = {
    "kg": "per_kg",
    "perkg": "per_kg",
    "je_kg": "per_kg",
    "100g": "per_100g",
    "per_100_g": "per_100g",
    "liter": "per_liter",
    "per_l": "per_liter",
    "je_liter": "per_liter",
    "100ml": "per_100ml",
    "per_100_ml": "per_100ml",
    "piece": "per_piece",
    "stueck": "per_piece",
    "stück": "per_piece",
    "unit": "per_piece",
    "package": "per_package",
    "pack": "per_package",
    "packung": "per_package",
}
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


def _normalized_enum(value, allowed_values: frozenset[str]) -> str:
    if value is None:
        return "unknown"
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed_values else "unknown"


class RawProduct(BaseModel):
    supplier: str
    location: str | None = None
    product_name: str
    description: str | None = None
    category: str  # fisch, fleisch, obst_gemuese, tk, wurst, mopro, sonstiges
    origin: str | None = None
    unit: str = "kg"
    quantity: float | None = None
    price: float
    price_per_kg: float | None = None
    price_is_net: bool = False
    price_gross: float | None = None
    price_tiers: list[dict] | None = None  # [{"min_qty": 15, "price": 20.99}]
    # Structured offer/package semantics. ``unit`` and ``quantity`` remain above
    # as legacy compatibility fields and are not used as the source of truth for
    # newly extracted data.
    price_basis: str = "unknown"
    package_count: int | None = None
    package_size_value: float | None = None
    package_size_unit: str = "unknown"
    total_content_value: float | None = None
    total_content_unit: str = "unknown"
    packaging_type: str = "unknown"
    packaging_raw: str | None = None
    certifications: list[str] = Field(default_factory=list)
    valid_from: date | None = None
    valid_to: date | None = None
    calendar_week: int | None = None
    year: int | None = None
    source_file: str = ""
    source_title: str | None = None
    source_tab: str | None = None
    source_page: int | None = None
    extraction_confidence: float = 1.0
    # Additive identity attributes. They intentionally do not replace the legacy
    # category field, so historical parsed CSV files remain loadable.
    product_family: str = "unknown"
    temperature_state: str = "unknown"
    processing_state: str = "unknown"
    calibre: str | None = None
    source_brand: str | None = None
    brand_evidence: str | None = None
    brand_evidence_source: str = "unknown"
    # Source-item and quality-retry provenance. Schema version 1 denotes rows
    # loaded from historical CSVs; current extraction explicitly writes v2.
    source_item_index: int | None = None
    source_item_id: str | None = None
    source_document_sha256: str | None = None
    primary_extraction_model: str | None = None
    selected_extraction_model: str | None = None
    quality_retry_attempted: bool = False
    quality_retry_status: str = "not_needed"
    quality_retry_model: str | None = None
    quality_retry_issues: list[str] = Field(default_factory=list)
    extraction_schema_version: int = 1

    @field_validator("product_family", mode="before")
    @classmethod
    def normalize_product_family(cls, value) -> str:
        return _normalized_enum(value, PRODUCT_FAMILIES)

    @field_validator("temperature_state", mode="before")
    @classmethod
    def normalize_temperature_state(cls, value) -> str:
        return _normalized_enum(value, TEMPERATURE_STATES)

    @field_validator("processing_state", mode="before")
    @classmethod
    def normalize_processing_state(cls, value) -> str:
        return _normalized_enum(value, PROCESSING_STATES)

    @field_validator("brand_evidence_source", mode="before")
    @classmethod
    def normalize_brand_evidence_source(cls, value) -> str:
        return _normalized_enum(value, BRAND_EVIDENCE_SOURCES)

    @field_validator("price_basis", mode="before")
    @classmethod
    def normalize_price_basis(cls, value) -> str:
        normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
        normalized = _PRICE_BASIS_ALIASES.get(normalized, normalized)
        return normalized if normalized in PRICE_BASES else "unknown"

    @field_validator("package_size_unit", "total_content_unit", mode="before")
    @classmethod
    def normalize_content_unit(cls, value) -> str:
        normalized = str(value or "").strip().casefold().replace(" ", "_")
        normalized = _CONTENT_UNIT_ALIASES.get(normalized, normalized)
        return normalized if normalized in CONTENT_UNITS else "unknown"

    @field_validator("packaging_type", mode="before")
    @classmethod
    def normalize_packaging_type(cls, value) -> str:
        normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
        normalized = _PACKAGING_TYPE_ALIASES.get(normalized, normalized)
        return normalized if normalized in PACKAGING_TYPES else "unknown"

    @field_validator("quality_retry_status", mode="before")
    @classmethod
    def normalize_quality_retry_status(cls, value) -> str:
        normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
        return normalized if normalized in QUALITY_RETRY_STATUSES else "not_needed"

    @field_validator("package_count", "source_item_index", mode="before")
    @classmethod
    def normalize_positive_integer(cls, value) -> int | None:
        if value in (None, "", "None", "null"):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0 or not number.is_integer():
            return None
        return int(number)

    @field_validator("package_size_value", "total_content_value", mode="before")
    @classmethod
    def normalize_positive_number(cls, value) -> float | None:
        if value in (None, "", "None", "null"):
            return None
        try:
            number = float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    @field_validator("certifications", "quality_retry_issues", mode="before")
    @classmethod
    def normalize_string_list(cls, value) -> list[str]:
        if value in (None, "", "None", "null"):
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = [part.strip() for part in value.split(",")]
            value = parsed if isinstance(parsed, list) else [parsed]
        if not isinstance(value, (list, tuple, set)):
            value = [value]
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            normalized.append(text.upper() if key in {"asc", "msc", "qs", "bio", "halal"} else text)
        return normalized

    @field_validator("extraction_schema_version", mode="before")
    @classmethod
    def normalize_extraction_schema_version(cls, value) -> int:
        try:
            version = int(value)
        except (TypeError, ValueError):
            return 1
        return version if version >= 1 else 1

    @field_validator(
        "calibre",
        "source_brand",
        "brand_evidence",
        "packaging_raw",
        "source_item_id",
        "source_document_sha256",
        "primary_extraction_model",
        "selected_extraction_model",
        "quality_retry_model",
        mode="before",
    )
    @classmethod
    def normalize_optional_identity_text(cls, value) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class CanonicalProduct(BaseModel):
    canonical_name: str
    category: str
    unit: str = "kg"
    keywords: list[str] = []


class MatchedProduct(BaseModel):
    canonical_name: str
    category: str
    unit: str
    metro_price: str | None = None  # netto/brutto format
    metro_info: str | None = None
    edeka_price: float | None = None
    edeka_info: str | None = None
    selgros_price: float | None = None
    selgros_period: str | None = None
    handelshof_price: float | None = None
    handelshof_period: str | None = None
    match_confidence: float = 1.0


class AcquiredDocument(BaseModel):
    supplier: str
    location: str
    doc_type: str = "pdf"  # pdf | web_data
    file_path: str | None = None
    url: str | None = None
    title: str | None = None
    tab: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    calendar_week: int | None = None
    year: int | None = None
    category: str | None = None
    is_relevant: bool | None = None
    relevance_label: str | None = None
    relevance_reason: str | None = None
    relevance_confidence: float | None = None
    market_scope: str | None = None
    valid_markets: list[str] | None = None
    products: list[RawProduct] | None = None  # for web-scraped data
