import json
import logging
import re
from datetime import date

from src.models import RawProduct
from src.extract.text_extract import classify_category

logger = logging.getLogger("birkenhof.extract.web")


def parse_json_ld_offers(json_ld_data: list[dict], supplier: str,
                         valid_from: date | None = None,
                         valid_to: date | None = None,
                         week: int | None = None,
                         year: int | None = None) -> list[RawProduct]:
    products = []

    for item in json_ld_data:
        try:
            item_type = item.get("@type", "")

            if item_type == "Product":
                name = item.get("name", "")
                offers = item.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                price = _extract_price(offers)
                description = item.get("description")
            elif item_type == "Offer":
                name = item.get("name", item.get("itemOffered", {}).get("name", ""))
                price = _extract_price(item)
                description = item.get("description")
            else:
                continue

            if not name or price <= 0:
                continue

            # Try to extract validity from offer
            offer_valid_from = valid_from
            offer_valid_to = valid_to
            if "validFrom" in item:
                try:
                    offer_valid_from = date.fromisoformat(item["validFrom"][:10])
                except (ValueError, TypeError):
                    pass
            if "validThrough" in item:
                try:
                    offer_valid_to = date.fromisoformat(item["validThrough"][:10])
                except (ValueError, TypeError):
                    pass

            unit = _detect_unit(name, description)

            products.append(RawProduct(
                supplier=supplier,
                product_name=name,
                description=description,
                category=classify_category(name + " " + (description or "")),
                unit=unit,
                price=price,
                price_is_net=False,
                valid_from=offer_valid_from,
                valid_to=offer_valid_to,
                calendar_week=week,
                year=year,
                source_file="web_json_ld",
                extraction_confidence=0.95,
            ))
        except Exception as e:
            logger.debug(f"Failed to parse JSON-LD item: {e}")

    return products


def _extract_price(data: dict) -> float:
    price_str = data.get("price", data.get("lowPrice", "0"))
    try:
        return float(str(price_str).replace(",", "."))
    except (ValueError, TypeError):
        return 0.0


def _detect_unit(name: str, description: str | None = None) -> str:
    text = (name + " " + (description or "")).lower()
    if any(u in text for u in ["pro kg", "/kg", "je kg", "per kg"]):
        return "kg"
    if any(u in text for u in ["stück", "stueck", "stk"]):
        return "stueck"
    if "bund" in text:
        return "bund"
    if "schale" in text:
        return "schale"
    if "packung" in text or "pack" in text:
        return "packung"
    if "flasche" in text:
        return "flasche"
    if "beutel" in text:
        return "beutel"
    return "kg"
