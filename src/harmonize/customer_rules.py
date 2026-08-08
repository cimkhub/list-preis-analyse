from __future__ import annotations

import re
from typing import Any


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
    result = (category or "sonstiges").strip().casefold() or "sonstiges"
    text = text_for_product_fields(brand, product_name, description).casefold()
    unit_text = str(unit or "").strip().casefold()

    if _contains_any_word(text, ["gefroren", "tiefgefroren", "tiefkühl", "tiefkuehl"]):
        result = "tk"

    if _contains_any_word(text, ["friesenkrone"]):
        result = "mopro"

    if any(term in text for term in ["bacon", "bratwurst", "bratwürst", "bratwuerst"]):
        result = "wurst"

    if unit_text in {"glas", "gläser", "glaeser", "dose", "dosen"} or _contains_any_word(
        text,
        ["glas", "gläser", "glaeser", "dose", "dosen"],
    ):
        result = "sonstiges"

    return result


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


def _contains_any_word(text: str, words: list[str]) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) for word in words)
