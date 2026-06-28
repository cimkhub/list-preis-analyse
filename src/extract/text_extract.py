import logging
import re
from datetime import date

from src.models import RawProduct
from src.convert.pdf_to_images import pdf_to_text, pdf_page_to_text, pdf_page_count
from src.harmonize.customer_rules import apply_customer_category_overrides

logger = logging.getLogger("birkenhof.extract.text")

CATEGORY_MAP = {
    # Fleisch
    "fleisch": "fleisch",
    "rind": "fleisch",
    "kalb": "fleisch",
    "schwein": "fleisch",
    "lamm": "fleisch",
    "schaf": "fleisch",
    "pute": "fleisch",
    "roastbeef": "fleisch",
    "filet": "fleisch",
    "hackfleisch": "fleisch",
    "gulasch": "fleisch",
    "schnitzel": "fleisch",
    "steak": "fleisch",
    "braten": "fleisch",
    "kasseler": "fleisch",
    "nacken": "fleisch",
    # Fisch
    "fisch": "fisch",
    "seafood": "fisch",
    "fischtheke": "fisch",
    "lachs": "fisch",
    "forelle": "fisch",
    "garnele": "fisch",
    "dorade": "fisch",
    "seeteufel": "fisch",
    "pulpo": "fisch",
    "shrimp": "fisch",
    "schwertfisch": "fisch",
    "seelachs": "fisch",
    "skrei": "fisch",
    "thunfisch": "fisch",
    # Obst & Gemüse
    "obst": "obst_gemuese",
    "gemüse": "obst_gemuese",
    "gemuese": "obst_gemuese",
    "o&g": "obst_gemuese",
    "spinat": "obst_gemuese",
    "salat": "obst_gemuese",
    "tomate": "obst_gemuese",
    "gurke": "obst_gemuese",
    "kartoffel": "obst_gemuese",
    "zwiebel": "obst_gemuese",
    "spargel": "obst_gemuese",
    "mango": "obst_gemuese",
    "erdbeere": "obst_gemuese",
    "ananas": "obst_gemuese",
    "melone": "obst_gemuese",
    "zitrone": "obst_gemuese",
    "rucola": "obst_gemuese",
    "lauch": "obst_gemuese",
    "möhre": "obst_gemuese",
    "traube": "obst_gemuese",
    "physalis": "obst_gemuese",
    "aubergine": "obst_gemuese",
    "kohl": "obst_gemuese",
    "frischeangebot": "obst_gemuese",
    # Milchprodukte
    "molkerei": "mopro",
    "milch": "mopro",
    "sahne": "mopro",
    "schmand": "mopro",
    "gouda": "mopro",
    "käse": "mopro",
    "kaese": "mopro",
    "joghurt": "mopro",
    "quark": "mopro",
    "butter": "mopro",
    "margarine": "mopro",
    "ayran": "mopro",
    # TK
    "tiefkühl": "tk",
    "tiefgefroren": "tk",
    "tk ": "tk",
    " tk": "tk",
    "fries": "tk",
    "broccoli": "tk",
    "himbeere": "tk",
    "brechbohne": "tk",
    # Wurst
    "wurst": "wurst",
    "bockwurst": "wurst",
    "krakauer": "wurst",
    "schinken": "wurst",
    # Sonstiges
    "feinkost": "sonstiges",
}


def classify_category(text: str) -> str:
    text_lower = text.lower()
    for keyword, cat in CATEGORY_MAP.items():
        if keyword in text_lower:
            return apply_customer_category_overrides(cat, product_name=text)
    return apply_customer_category_overrides("sonstiges", product_name=text)


def extract_metro_text(pdf_path: str, valid_from: date | None = None, valid_to: date | None = None) -> list[RawProduct]:
    full_text = pdf_to_text(pdf_path)
    if not full_text.strip():
        logger.warning(f"No text extracted from Metro PDF: {pdf_path}")
        return []

    products = []
    pages = pdf_page_count(pdf_path)

    for page_num in range(1, pages + 1):
        page_text = pdf_page_to_text(pdf_path, page_num)
        if not page_text.strip():
            continue

        page_products = _parse_metro_page(page_text, page_num, pdf_path, valid_from, valid_to)
        products.extend(page_products)

    logger.info(f"Metro text extraction: {len(products)} products from {pages} pages")
    return products


def _parse_metro_page(text: str, page_num: int, source: str,
                      valid_from: date | None, valid_to: date | None) -> list[RawProduct]:
    products = []
    # Metro price pattern: number,number* (number,number) or just number,number
    # Tier pattern: ab X kg   je kg   PRICE* (PRICE)
    price_pattern = re.compile(
        r'(\d+[,\.]\d{2})\s*\*?\s*\(?\s*(\d+[,\.]\d{2})?\s*\)?'
    )
    tier_pattern = re.compile(
        r'ab\s+(\d+[\s,]*\d*)\s*kg\s+je\s+kg\s+(\d+[,\.]\d{2})\s*\*?\s*\(?(\d+[,\.]\d{2})?\)?'
    )

    lines = text.splitlines()
    current_product = None
    current_details = []
    current_tiers = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_product and current_tiers:
                product = _build_metro_product(
                    current_product, current_details, current_tiers,
                    page_num, source, valid_from, valid_to
                )
                if product:
                    products.append(product)
            current_product = None
            current_details = []
            current_tiers = []
            continue

        tier_match = tier_pattern.search(line)
        if tier_match:
            qty = float(tier_match.group(1).replace(",", ".").replace(" ", ""))
            net_price = float(tier_match.group(2).replace(",", "."))
            gross_price = float(tier_match.group(3).replace(",", ".")) if tier_match.group(3) else None
            current_tiers.append({
                "min_qty": qty,
                "price_net": net_price,
                "price_gross": gross_price
            })
            continue

        if line.startswith("•") or line.startswith("·"):
            current_details.append(line.lstrip("•· "))
            continue

        # Check if this looks like a product name (no prices, not too short)
        if not price_pattern.search(line) and len(line) > 3 and not line.startswith("ab "):
            if current_product and current_tiers:
                product = _build_metro_product(
                    current_product, current_details, current_tiers,
                    page_num, source, valid_from, valid_to
                )
                if product:
                    products.append(product)
                current_details = []
                current_tiers = []
            current_product = line

    # Last product on page
    if current_product and current_tiers:
        product = _build_metro_product(
            current_product, current_details, current_tiers,
            page_num, source, valid_from, valid_to
        )
        if product:
            products.append(product)

    return products


def _build_metro_product(name: str, details: list[str], tiers: list[dict],
                         page_num: int, source: str,
                         valid_from: date | None, valid_to: date | None) -> RawProduct | None:
    if not tiers:
        return None

    # Use middle tier if available, else first
    if len(tiers) >= 3:
        tier = tiers[1]  # middle
    elif len(tiers) >= 2:
        tier = tiers[0]  # lower quantity = higher price, more realistic
    else:
        tier = tiers[0]

    description = "; ".join(details) if details else None
    category = classify_category(name + " " + (description or ""))

    net_price = tier["price_net"]
    gross_price = tier.get("price_gross") or round(net_price * 1.07, 2)

    return RawProduct(
        supplier="metro",
        product_name=name.strip(),
        description=description,
        category=category,
        price=net_price,
        price_is_net=True,
        price_gross=gross_price,
        price_per_kg=gross_price,  # Metro prices are per kg
        unit="kg",
        price_tiers=[{"min_qty": t["min_qty"], "price": t["price_net"]} for t in tiers],
        valid_from=valid_from,
        valid_to=valid_to,
        source_file=source,
        source_page=page_num,
        extraction_confidence=0.8,
    )


def extract_selgros_filename_meta(filename: str) -> dict:
    """Parse Selgros filename: 22041_ALLE_20260302_20260331_Fleisch_F_kl.pdf"""
    parts = filename.replace(".pdf", "").split("_")
    meta = {"article_id": parts[0] if parts else "", "scope": "", "category": ""}

    if len(parts) >= 6:
        meta["scope"] = parts[1]
        try:
            meta["valid_from"] = date(int(parts[2][:4]), int(parts[2][4:6]), int(parts[2][6:8]))
            meta["valid_to"] = date(int(parts[3][:4]), int(parts[3][4:6]), int(parts[3][6:8]))
        except (ValueError, IndexError):
            pass

        tail = parts[4:]
        if tail[-1].lower() == "kl":
            tail = tail[:-1]
        if tail and tail[-1] in {"F", "A"}:
            tail = tail[:-1]
        meta["category"] = "_".join(tail)

    return meta
