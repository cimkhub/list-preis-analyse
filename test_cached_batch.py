import csv
import json
from datetime import date

from src.extract.cached_batch import (
    CachedExtractionTarget,
    dedupe_documents,
    discover_cached_targets,
    load_downloaded_documents,
)
from src.models import AcquiredDocument, RawProduct
from src.report.parsed_csv import save_combined_csv, save_parsed_csv


def test_discover_cached_targets_finds_supplier_week_dirs(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "edeka" / "2026" / "15").mkdir(parents=True)
    (data_root / "edeka" / "2026" / "15" / "flyer.pdf").write_bytes(b"%PDF-1.4 test")
    (data_root / "metro" / "2026" / "16").mkdir(parents=True)
    (data_root / "metro" / "2026" / "16" / "raw_brochures.json").write_text("[]", encoding="utf-8")
    (data_root / "metro" / "misc").mkdir(parents=True)

    targets = discover_cached_targets(data_root, ["edeka", "metro"])

    assert [(target.supplier, target.year, target.week) for target in targets] == [
        ("edeka", 2026, 15),
        ("metro", 2026, 16),
    ]


def test_load_downloaded_documents_keeps_all_downloaded_pdfs(tmp_path):
    data_dir = tmp_path / "data" / "edeka" / "2026" / "15"
    data_dir.mkdir(parents=True)
    pdf_a = data_dir / "a.pdf"
    pdf_b = data_dir / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 a")
    pdf_b.write_bytes(b"%PDF-1.4 b")

    raw_brochures = [
        {
            "selected": False,
            "title": "Hidden Flyer",
            "tab": "Archiv",
            "valid_from": "2026-04-02",
            "valid_to": "2026-04-08",
            "catalog_category_name": "Sonderwerbung",
            "pdf_local_path": str(pdf_a),
        },
        {
            "selected": True,
            "title": "Visible Flyer",
            "tab": "Aktuell",
            "valid_from": "2026-04-09",
            "valid_to": "2026-04-15",
            "catalog_category_name": "Aktuell",
            "pdf_local_path": str(pdf_b),
        },
    ]
    (data_dir / "raw_brochures.json").write_text(
        json.dumps(raw_brochures),
        encoding="utf-8",
    )

    target = CachedExtractionTarget(
        supplier="edeka",
        week=15,
        year=2026,
        data_dir=data_dir,
    )
    documents = load_downloaded_documents(target, {"location": "wernigerode"})

    assert len(documents) == 2
    assert {document.title for document in documents} == {"Hidden Flyer", "Visible Flyer"}
    assert {document.tab for document in documents} == {"Archiv", "Aktuell"}
    assert {document.valid_from for document in documents} == {
        date(2026, 4, 2),
        date(2026, 4, 9),
    }
    assert all(document.location == "wernigerode" for document in documents)


def test_dedupe_documents_uses_file_content_hash(tmp_path):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 same-content")
    pdf_b.write_bytes(b"%PDF-1.4 same-content")

    documents = [
        AcquiredDocument(
            supplier="handelshof",
            location="hannover",
            file_path=str(pdf_a),
            calendar_week=15,
            year=2026,
        ),
        AcquiredDocument(
            supplier="handelshof",
            location="hannover",
            file_path=str(pdf_b),
            calendar_week=17,
            year=2026,
        ),
    ]

    unique_documents = dedupe_documents(documents)

    assert len(unique_documents) == 1
    assert unique_documents[0].file_path == str(pdf_a)


def test_save_parsed_csv_writes_provenance_columns(tmp_path):
    output_dir = tmp_path / "parsed"
    products = [
        RawProduct(
            supplier="selgros",
            location="braunschweig",
            product_name="Rinderfilet",
            category="fleisch",
            price=19.99,
            valid_from=date(2026, 4, 2),
            valid_to=date(2026, 4, 8),
            calendar_week=15,
            year=2026,
            source_file="data/selgros/2026/15/file.pdf",
            source_title="Bestes Fleisch",
            source_tab="Aktuelle Angebote",
            source_page=3,
        )
    ]

    save_parsed_csv(products, "selgros", 15, 2026, str(output_dir))

    csv_path = output_dir / "KW15_2026" / "selgros.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["location"] == "braunschweig"
    assert rows[0]["source_title"] == "Bestes Fleisch"
    assert rows[0]["source_tab"] == "Aktuelle Angebote"
    assert rows[0]["source_file"] == "data/selgros/2026/15/file.pdf"


def test_save_combined_csv_writes_batch_level_file(tmp_path):
    output_dir = tmp_path / "parsed"
    products = [
        RawProduct(
            supplier="metro",
            location="goslar",
            product_name="Rinderfilet",
            category="fleisch",
            price=19.99,
            valid_from=date(2026, 4, 27),
            valid_to=date(2026, 5, 2),
            calendar_week=18,
            year=2026,
            source_file="data/metro/2026/18/wochen-angebote-270426-020526.pdf",
            source_page=1,
        ),
        RawProduct(
            supplier="selgros",
            location="braunschweig",
            product_name="Lachsfilet",
            category="fisch",
            price=12.49,
            valid_from=date(2026, 4, 27),
            valid_to=date(2026, 5, 2),
            calendar_week=18,
            year=2026,
            source_file="data/selgros/2026/18/file.pdf",
            source_page=2,
        ),
    ]

    save_combined_csv(products, 18, 2026, str(output_dir))

    csv_path = output_dir / "KW18_2026" / "all_suppliers.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert {row["supplier"] for row in rows} == {"metro", "selgros"}
