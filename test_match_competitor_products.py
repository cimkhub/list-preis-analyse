import pandas as pd

from match_competitor_products import dedupe_exact_same_supplier_offers, normalize_confidence


def test_normalize_confidence_accepts_fractional_and_percent_scales():
    assert normalize_confidence(1) == 100
    assert normalize_confidence("0.92") == 92
    assert normalize_confidence("87") == 87
    assert normalize_confidence("") == 0


def test_dedupe_exact_same_supplier_offers_ignores_source_and_unit_noise():
    rows = [
        {
            "product_id": "p1",
            "supplier": "metro",
            "supplier_norm": "Metro",
            "category": "fisch",
            "product_name": "Marinaden",
            "product": "Marinaden",
            "brand": "",
            "description": "Verschiedene Sorten",
            "origin": "",
            "unit": "schale",
            "quantity": "2.5",
            "price": "19.99",
            "price_per_kg": "",
            "price_tiers": '[{"min_qty": 4, "price": 19.99}]',
            "valid_from": "2026-06-01",
            "valid_to": "2026-06-06",
            "source_file": "a.pdf",
            "source_page": "8",
        },
        {
            "product_id": "p2",
            "supplier": "metro",
            "supplier_norm": "Metro",
            "category": "fisch",
            "product_name": "Marinaden",
            "product": "Marinaden",
            "brand": "",
            "description": "Verschiedene Sorten",
            "origin": "",
            "unit": "kg",
            "quantity": "2.5",
            "price": "19.99",
            "price_per_kg": "",
            "price_tiers": '[{"price": 19.99, "min_qty": 4}]',
            "valid_from": "2026-06-01",
            "valid_to": "2026-06-06",
            "source_file": "b.pdf",
            "source_page": "9",
        },
    ]

    deduped, removed = dedupe_exact_same_supplier_offers(pd.DataFrame(rows))

    assert removed == 1
    assert len(deduped) == 1
