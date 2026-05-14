#!/usr/bin/env python3
"""Test the pipeline end-to-end using example data from the reference Excel."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import RawProduct
from src.harmonize.matcher import load_canonical_products, match_all_products, build_comparison
from src.harmonize.categories import category_label
from src.report.excel_report import generate_report
from src.utils.logging_setup import setup_logging
from datetime import date


def create_test_products() -> list[RawProduct]:
    """Create test products based on the reference Excel data."""
    products = []

    # Metro products (netto prices from reference Excel)
    metro_items = [
        ("Forellen vers. Sort.", "fisch", 8.99, 9.99),
        ("Moscardini Moschus Krake", "fisch", 12.99, 13.99),
        ("Pulpo MAP 2kg Schale", "fisch", 22.49, 23.99),
        ("Schwertfischfilet mH 2-4kg", "fisch", 20.49, 21.99),
        ("Seelachsfilet vers. Sort.", "fisch", 18.99, 19.99),
        ("Seeteufelschwänze o.K. 1-4kg", "fisch", 23.49, 24.99),
        ("Skreifilet mit/ohne Haut", "fisch", 24.99, 26.99),
        ("White Tiger Garnelen 20-40", "fisch", 1.29, 1.79),
        ("Kalbsbraten Dicker Bug hell", "fleisch", 14.99, 16.04),
        ("Kalbskugel hell", "fleisch", 15.99, 17.11),
        ("Kalbsleber hell TK", "fleisch", 9.99, None),
        ("Kasselernacken", "fleisch", 5.99, 6.99),
        ("Lamm ganz TK ca. 15kg", "fleisch", 8.99, None),
        ("Lammfilet TK", "fleisch", 27.49, 29.49),
        ("Lammhinterhaxe TK", "fleisch", 11.99, 13.99),
        ("Lammkotelett TK", "fleisch", 17.49, 19.49),
        ("Rinderfilet 3/4er Arg.", "fleisch", 32.99, 34.99),
        ("Rinderroastbeef Arg.", "fleisch", 20.99, 22.99),
        ("Rinderbrust DT", "fleisch", 12.99, 13.99),
        ("Rinder-Entrecôte Simmentaler", "fleisch", 24.99, None),
        ("Rinderhüftsteaks Arg.", "fleisch", 23.99, None),
        ("Rindertafelspitz EU", "fleisch", 12.99, 13.99),
        ("Schweinelachse TK", "fleisch", 3.49, 3.99),
        ("Schweinenacken TK o.k.", "fleisch", 4.29, 4.79),
        ("Schweineschinken schier", "fleisch", 3.99, 4.49),
        ("Ananas CR", "obst_gemuese", 2.51, 2.79),
        ("Auberginen NL/ESP 5kg", "obst_gemuese", 13.75, 15.99),
        ("Cantaloupemelone BR/HO", "obst_gemuese", 2.15, 2.39),
        ("Erdbeeren GR/ESP 500g", "obst_gemuese", 2.42, 2.69),
        ("Gurken 350/400g", "obst_gemuese", 0.62, 0.69),
        ("Rucola 1kg Styro", "obst_gemuese", 5.50, 6.39),
        ("Spargel weiß/violett 22mm 5kg", "obst_gemuese", 49.95, None),
        ("Zitronen Primofiori 5kg", "obst_gemuese", 9.89, 10.99),
        ("Zwiebeln rot NL 10kg", "obst_gemuese", 6.49, None),
        ("MC Brechbohnen 2,5kg", "tk", 3.39, 3.79),
        ("MC Himbeeren 2,5kg", "tk", 20.99, 23.99),
    ]

    for name, cat, net, gross in metro_items:
        products.append(RawProduct(
            supplier="metro",
            product_name=name,
            category=cat,
            price=net,
            price_is_net=True,
            price_gross=gross,
            unit="kg",
            valid_from=date(2026, 3, 23),
            valid_to=date(2026, 3, 28),
            calendar_week=13,
            year=2026,
            source_file="METRO DE - Wochen-Angebote.pdf",
        ))

    # EDEKA products
    edeka_items = [
        ("Blockhouse Rinderfilet 3/4 Uruguay", "fleisch", 34.99),
        ("Jungschaflachse TK", "fleisch", 19.99),
        ("Feldsalat IT/FR 125g", "obst_gemuese", 1.19),
        ("Fleischtomaten ESP", "obst_gemuese", 2.99),
        ("Lauchzwiebeln DT/EGY Bund", "obst_gemuese", 0.69),
        ("Mango PR", "obst_gemuese", 1.99),
        ("Spargel grün PR/MX 500g", "obst_gemuese", 3.99),
        ("Zitronen ESP 500g", "obst_gemuese", 3.99),
        ("Schleizer Bockwurst 10x100g", "wurst", 4.99),
        ("Henkelmann Hinterkochschinken 500g", "wurst", 4.99),
        ("EFS H Schmand", "mopro", 2.99),
    ]

    for name, cat, price in edeka_items:
        products.append(RawProduct(
            supplier="edeka",
            product_name=name,
            category=cat,
            price=price,
            price_is_net=False,
            unit="kg",
            valid_from=date(2026, 3, 26),
            valid_to=date(2026, 4, 1),
            calendar_week=13,
            year=2026,
            source_file="Aktuell [Nord].pdf",
        ))

    # Selgros products
    selgros_items = [
        ("Hackfleisch gemischt", "fleisch", 7.90, "26.03.-01.04."),
        ("Kalbsgulasch a.d. Unterschale", "fleisch", 17.90, "26.03.-01.04."),
        ("Kalbsschnitzel a.d. Unterschale", "fleisch", 26.90, "26.03.-01.04."),
        ("Lammkeule TK m.K.", "fleisch", 12.49, "23.03.-28.03"),
        ("Rindergulasch a.d. Keule", "fleisch", 13.90, "26.03.-01.04."),
        ("Rinderroastbeef Arg.", "fleisch", 22.99, "23.03.-28.03"),
        ("Rinderroastbeef Arg. TK", "fleisch", 18.88, "23.03.-28.03"),
        ("Rinderroastbeef Brasil", "fleisch", 14.99, "23.03.-28.03"),
        ("Rinderrouladen a.d. Oberschale", "fleisch", 16.90, "26.03.-01.04."),
        ("Rinderfilet 3/4 Arg.", "fleisch", 29.99, "23.03.-28.03"),
        ("Schweinenacken o.k.", "fleisch", 4.29, "23.03.-28.03"),
        ("Schweinenackensteaks gew.", "fleisch", 7.90, "26.03.-01.04."),
        ("Baby Spinat 500g", "obst_gemuese", 6.49, "26.03.-01.04."),
        ("Flugmango PR/BR", "obst_gemuese", 4.44, "26.03.-01.04."),
        ("Honigmelone BR/CR/HD", "obst_gemuese", 2.22, "26.03.-01.04."),
        ("Lauchzwiebeln Premium IT Bund", "obst_gemuese", 1.99, "23.03.-28.03"),
        ("Mix Salat 9er IT", "obst_gemuese", 9.49, "23.03.-28.03"),
        ("Speisefrühkartoffeln festk. 1,5kg", "obst_gemuese", 2.99, "26.03.-01.04."),
        ("Broccoli 20/40 TK", "tk", 5.975, "23.03.-28.03"),
        ("Economy Weizenbrötchen 100x70g", "tk", 0.15, "23.03.-28.03"),
        ("Schleizer Krakauer 10x120g", "wurst", 9.99, "23.03.-28.03"),
        ("Schleizer Bockwurst 10x100g", "wurst", 5.99, "26.03.-01.04."),
        ("Dairystar H-Schlagsahne 30%", "mopro", 2.49, "23.03.-28.03"),
        ("Gouda gerieben 48% Delina", "mopro", 4.79, "23.03.-28.03"),
    ]

    for name, cat, price, period in selgros_items:
        products.append(RawProduct(
            supplier="selgros",
            product_name=name,
            category=cat,
            price=price,
            price_is_net=False,
            unit="kg",
            valid_from=date(2026, 3, 23),
            valid_to=date(2026, 3, 28),
            calendar_week=13,
            year=2026,
            source_file="Selgros Flyer",
        ))

    # Handelshof products
    handelshof_items = [
        ("Dorade Royal", "fisch", 11.99, "02.04.-08.04"),
        ("Seelachsfilet vers. Sort.", "fisch", 10.99, "02.04.-08.04"),
        ("Jungschaflachse TK", "fleisch", 19.99, "02.04.-08.04"),
        ("Kalbssemerrolle rosé", "fleisch", 18.79, "02.04.-08.04"),
        ("Putenschnitzel", "fleisch", 11.49, "02.04.-08.04"),
        ("Rind Falsches Filet", "fleisch", 13.49, "02.04.-08.04"),
        ("Auberginen NL/ESP 5kg", "obst_gemuese", 2.79, "02.04.-08.04"),
        ("Fleischtomaten ESP", "obst_gemuese", 2.99, "02.04.-08.04"),
        ("Lauchzwiebeln DT/EGY Bund", "obst_gemuese", 0.69, "02.04.-08.04"),
        ("Mango ready to eat PR", "obst_gemuese", 1.99, "02.04.-08.04"),
        ("Möhren Bund IT", "obst_gemuese", 1.49, "02.04.-08.04"),
        ("Roscof Zwiebeln FR 350g", "obst_gemuese", 3.99, "02.04.-08.04"),
        ("Spargel grün PR/MX 500g", "obst_gemuese", 3.99, "02.04.-08.04"),
        ("Tafeltrauben rot ZA kernlos", "obst_gemuese", 3.99, "02.04.-08.04"),
        ("Aviko Sweet Potatoe Fries 2,27kg", "tk", 8.49, "02.04.-08.04"),
        ("Broccoli 20/40 TK", "tk", 5.49, "02.04.-08.04"),
        ("Henkelmann Hinterkochschinken 500g", "wurst", 4.99, "02.04.-08.04"),
        ("Schleizer Bockwurst 10x100g", "wurst", 4.99, "02.04.-08.04"),
        ("EFS H Schmand", "mopro", 2.99, "02.04.-08.04"),
    ]

    for name, cat, price, period in handelshof_items:
        products.append(RawProduct(
            supplier="handelshof",
            product_name=name,
            category=cat,
            price=price,
            price_is_net=False,
            unit="kg",
            valid_from=date(2026, 4, 2),
            valid_to=date(2026, 4, 8),
            calendar_week=14,
            year=2026,
            source_file="Handelshof Web",
        ))

    return products


def main():
    logger = setup_logging()
    logger.info("=== END-TO-END TEST WITH EXAMPLE DATA ===")

    # Create test products (simulating extraction results)
    all_products = create_test_products()
    logger.info(f"Created {len(all_products)} test products")

    supplier_counts = {}
    for p in all_products:
        supplier_counts[p.supplier] = supplier_counts.get(p.supplier, 0) + 1
    for s, c in sorted(supplier_counts.items()):
        logger.info(f"  {s}: {c} products")

    # Load canonical products
    canonicals = load_canonical_products()

    # Match products
    matched, unmatched = match_all_products(all_products, canonicals)
    logger.info(f"Matched: {len(matched)} canonical products")
    logger.info(f"Unmatched: {len(unmatched)} products")

    if unmatched:
        logger.info("Unmatched products:")
        for p in unmatched:
            logger.info(f"  [{p.supplier}] {p.product_name}")

    # Build comparison
    comparison = build_comparison(matched, canonicals)

    # Generate report
    report_path = generate_report(
        comparison, unmatched, all_products,
        week=13, year=2026,
        output_dir="reports",
    )

    logger.info(f"\nReport generated: {report_path}")
    logger.info("Test complete!")


if __name__ == "__main__":
    main()
