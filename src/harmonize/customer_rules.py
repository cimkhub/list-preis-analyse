from __future__ import annotations

from typing import Any


CANONICAL_CATEGORIES = frozenset(
    {"fleisch", "fisch", "obst_gemuese", "tk", "wurst", "mopro", "sonstiges"}
)
CATEGORY_ALIASES = {
    "obst & gemüse": "obst_gemuese",
    "obst & gemuese": "obst_gemuese",
    "obst/gemüse": "obst_gemuese",
    "obst/gemuese": "obst_gemuese",
}


def text_for_product_fields(*values: object) -> str:
    return " ".join(str(value or "") for value in values)


def apply_customer_category_overrides(
    category: str | None,
    *,
    product_name: object = "",
    description: object = "",
    brand: object = "",
    unit: object = "",
) -> str:
    """Normalize the supplied category without inferring it from product text.

    The keyword arguments remain part of the public API for older callers. Product
    name, description, brand, packaging and temperature must not silently rewrite
    the product family/category at this layer.
    """
    _ = product_name, description, brand, unit
    normalized = str(category or "").strip().casefold()
    normalized = CATEGORY_ALIASES.get(normalized, normalized)
    return normalized if normalized in CANONICAL_CATEGORIES else "sonstiges"


def apply_customer_category_overrides_to_row(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["category"] = apply_customer_category_overrides(
        str(updated.get("category") or "sonstiges"),
        product_name=updated.get("product_name"),
        description=updated.get("description"),
        brand=updated.get("brand"),
        unit=updated.get("unit"),
    )
    return updated


def apply_customer_category_overrides_to_product(product: Any) -> Any:
    category = apply_customer_category_overrides(
        product.category,
        product_name=product.product_name,
        description=product.description,
        unit=product.unit,
    )
    if category == product.category:
        return product
    return product.model_copy(update={"category": category})
