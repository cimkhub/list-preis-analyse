from pathlib import Path

from openpyxl import Workbook
import pandas as pd

from match_competitor_products import (
    build_final_output_rows,
    dedupe_exact_same_supplier_offers,
    final_output_column_widths,
    format_offer_cell_short,
    normalize_confidence,
    origin_country_codes,
    write_final_output_sheet,
)


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


def test_final_output_moves_brand_into_description_and_removes_brand_column():
    rows = build_final_output_rows([
        {
            "category": "fleisch",
            "product": "Rinderfilet",
            "brand": "LIST Premium",
            "description": "Argentinisches Rinderfilet",
            "origin": "Argentinien",
        }
    ])

    assert "Marke" not in rows[0]
    assert rows[0]["Beschreibung"] == "Argentinisches Rinderfilet\nBrand: LIST Premium,"
    assert rows[0]["Herkunft"] == "AR"


def test_final_output_skips_brand_line_when_brand_is_empty():
    rows = build_final_output_rows([
        {
            "category": "fleisch",
            "product": "Rinderfilet",
            "brand": " ",
            "description": "Argentinisches Rinderfilet",
            "origin": "Argentinien",
        }
    ])

    assert rows[0]["Beschreibung"] == "Argentinisches Rinderfilet"


def test_origin_country_codes_maps_german_names_aliases_and_existing_codes():
    assert origin_country_codes("Deutschland / Niederlande") == "DE, NL"
    assert origin_country_codes("Holland") == "NL"
    assert origin_country_codes("de") == "DE"
    assert origin_country_codes("Spanien, Italien oder Frankreich") == "ES, IT, FR"
    assert origin_country_codes("Chile + Neuseeland") == "CL, NZ"


def test_short_offer_cell_uses_line_breaks_instead_of_wide_separator():
    cell = format_offer_cell_short({
        "price": "19.99",
        "quantity": "2.5",
        "unit": "kg",
        "price_tiers": '[{"min_qty": 4, "price": 18.5}]',
        "valid_from": "2026-06-01",
        "valid_to": "2026-06-06",
    })

    assert cell == "19,99 €\n2,5 kg\nab 4: 18,5 €\n01.06. bis 06.06."
    assert " | " not in cell


def test_compact_final_output_widths_are_narrower_for_editing():
    widths = final_output_column_widths(compact_rows=True)

    assert sum(widths.values()) <= 170
    assert widths["Metro"] == 17
    assert widths["Beschreibung"] == 24
    assert widths["Herkunft"] == 7


def test_final_output_formats_ek_and_vk_as_euro_values():
    wb = Workbook()
    ws = wb.active
    write_final_output_sheet(
        ws,
        [{
            "Kategorie": "Fleisch",
            "Produkt": "Rinderfilet",
            "Beschreibung": "Argentinisches Rinderfilet",
            "Herkunft": "AR",
            "Metro": "",
            "Selgros": "",
            "Handelshof": "",
            "Edeka": "",
            "EK": 12.3456,
            "VK": 19.99,
            "Notizen": "",
        }],
        Path("Artikelvergleich KW01.xlsx"),
        None,
    )
    headers = {str(cell.value): cell.column for cell in ws[7]}

    assert ws.cell(8, headers["EK"]).number_format == '#,##0.000 "€"'
    assert ws.cell(8, headers["VK"]).number_format == '#,##0.00 "€"'


def test_compact_final_output_row_height_grows_with_wrapped_content():
    wb = Workbook()
    ws = wb.active
    write_final_output_sheet(
        ws,
        [{
            "Kategorie": "Fleisch",
            "Produkt": "Rinderfilet mit sehr langem Namen",
            "Beschreibung": "Sehr lange Beschreibung mit vielen Details zur Qualität, zum Zuschnitt und zur Verarbeitung\nBrand: LIST Premium,",
            "Herkunft": "AR",
            "Metro": "19,99 €\n2,5 kg\nab 4: 18,5 €\n01.06. bis 06.06.",
            "Selgros": "20,99 €\n2,5 kg\n01.06. bis 06.06.",
            "Handelshof": "",
            "Edeka": "",
            "EK": 12.3456,
            "VK": 19.99,
            "Notizen": "Lange Bearbeitungsnotiz fuer die Disposition",
        }],
        Path("Artikelvergleich KW01.xlsx"),
        None,
        compact_rows=True,
    )

    assert ws.row_dimensions[8].height > 80
    assert ws.row_dimensions[8].height <= 360
