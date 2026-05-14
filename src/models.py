from datetime import date
from pydantic import BaseModel


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
    valid_from: date | None = None
    valid_to: date | None = None
    calendar_week: int | None = None
    year: int | None = None
    source_file: str = ""
    source_title: str | None = None
    source_tab: str | None = None
    source_page: int | None = None
    extraction_confidence: float = 1.0


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
