import csv
import logging
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz, process

from src.models import CanonicalProduct, RawProduct, MatchedProduct

logger = logging.getLogger("birkenhof.harmonize")


def load_canonical_products(csv_path: str = "reference/canonical_products.csv") -> list[CanonicalProduct]:
    products = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keywords = [k.strip() for k in row["keywords"].split(",") if k.strip()]
            products.append(CanonicalProduct(
                canonical_name=row["canonical_name"],
                category=row["category"],
                unit=row.get("unit", "kg"),
                keywords=keywords,
            ))
    logger.info(f"Loaded {len(products)} canonical products from {csv_path}")
    return products


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("-", " ").replace("/", " ").replace(".", " ")
    # Normalize German umlauts
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = text.replace("é", "e").replace("ô", "o").replace("è", "e")
    return " ".join(text.split())


def match_product(raw_name: str, canonicals: list[CanonicalProduct],
                  threshold: int = 80) -> tuple[CanonicalProduct | None, float]:
    normalized = normalize_text(raw_name)

    # Layer 1: keyword match - prefer longest matching keyword
    matches = []
    for canon in canonicals:
        for kw in canon.keywords:
            nkw = normalize_text(kw)
            if nkw in normalized or normalized in nkw:
                matches.append((canon, len(nkw)))
    if matches:
        # Return the match with the longest keyword (most specific)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0], 1.0

    # Layer 2: fuzzy match against canonical names
    choices = {i: normalize_text(c.canonical_name) for i, c in enumerate(canonicals)}
    result = process.extractOne(
        normalized,
        choices,
        scorer=fuzz.token_sort_ratio,
    )

    if result and result[1] >= threshold:
        idx = result[2]
        return canonicals[idx], result[1] / 100.0

    return None, 0.0


def match_all_products(
    raw_products: list[RawProduct],
    canonicals: list[CanonicalProduct],
    threshold: int = 80,
) -> tuple[dict[str, dict[str, RawProduct]], list[RawProduct]]:
    """
    Returns:
        matched: {canonical_name: {supplier: best_product}}
        unmatched: list of products that couldn't be matched
    """
    matched: dict[str, dict[str, RawProduct]] = {}
    unmatched: list[RawProduct] = []

    for product in raw_products:
        canon, confidence = match_product(product.product_name, canonicals, threshold)

        if canon:
            if canon.canonical_name not in matched:
                matched[canon.canonical_name] = {}

            supplier = product.supplier
            existing = matched[canon.canonical_name].get(supplier)

            # Keep the product with higher confidence or lower price
            if existing is None or confidence > (existing.extraction_confidence or 0):
                matched[canon.canonical_name][supplier] = product
                logger.debug(
                    f"Matched '{product.product_name}' -> '{canon.canonical_name}' "
                    f"(confidence: {confidence:.0%}, supplier: {supplier})"
                )
        else:
            unmatched.append(product)
            logger.debug(f"Unmatched: '{product.product_name}' ({product.supplier})")

    logger.info(
        f"Matching complete: {len(matched)} products matched, "
        f"{len(unmatched)} unmatched from {len(raw_products)} total"
    )
    return matched, unmatched


def build_comparison(
    matched: dict[str, dict[str, RawProduct]],
    canonicals: list[CanonicalProduct],
) -> list[MatchedProduct]:
    canon_map = {c.canonical_name: c for c in canonicals}
    rows = []

    for name, suppliers in matched.items():
        canon = canon_map.get(name)
        if not canon:
            continue

        metro = suppliers.get("metro")
        edeka = suppliers.get("edeka")
        selgros = suppliers.get("selgros")
        handelshof = suppliers.get("handelshof")

        # Metro: format as netto/brutto
        metro_price = None
        if metro:
            if metro.price_tiers and len(metro.price_tiers) > 1:
                prices = [f"{t['price']:.2f}".replace(".", ",") for t in metro.price_tiers]
                metro_price = "/".join(prices)
            elif metro.price_gross:
                metro_price = f"{metro.price:.2f}/{metro.price_gross:.2f}".replace(".", ",")
            else:
                metro_price = f"{metro.price:.2f}".replace(".", ",")

        def _format_period(product: RawProduct | None) -> str | None:
            if not product or not product.valid_from or not product.valid_to:
                return None
            return f"{product.valid_from.strftime('%d.%m.')}-{product.valid_to.strftime('%d.%m.')}"

        rows.append(MatchedProduct(
            canonical_name=name,
            category=canon.category,
            unit=canon.unit,
            metro_price=metro_price,
            metro_info=metro.description if metro else None,
            edeka_price=edeka.price if edeka else None,
            edeka_info=_format_period(edeka),
            selgros_price=selgros.price if selgros else None,
            selgros_period=_format_period(selgros),
            handelshof_price=handelshof.price if handelshof else None,
            handelshof_period=_format_period(handelshof),
        ))

    return rows
