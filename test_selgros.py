import json

from src.acquire.selgros import SelgrosScraper
from src.extract.text_extract import extract_selgros_filename_meta


def make_scraper() -> SelgrosScraper:
    return SelgrosScraper({
        "location": "braunschweig",
        "url": "https://www.selgros.de/markt/braunschweig",
    })


def test_parse_flipbook_url_reads_dates_and_category():
    scraper = make_scraper()

    meta = scraper._parse_flipbook_url(
        "https://www.selgros.de/sites/default/files/offers/"
        "22182_ALLE_20260402_20260408_Bestes_Fleisch_F/index.html"
    )

    assert meta is not None
    assert meta["valid_from"].isoformat() == "2026-04-02"
    assert meta["valid_to"].isoformat() == "2026-04-08"
    assert meta["category_name"] == "Bestes_Fleisch"
    assert meta["pdf_filename"] == "22182_ALLE_20260402_20260408_Bestes_Fleisch_F_kl.pdf"


def test_extract_pdf_url_from_html_prefers_viewer_download_pattern():
    scraper = make_scraper()
    html = """
    <html>
      <body>
        <button title="Download" onclick="FLOWPAPER.REFLOW.downloadPDF('22182_ALLE_20260402_20260408_Food_F_kl.pdf')">
          Download
        </button>
      </body>
    </html>
    """

    pdf_url = scraper._extract_pdf_url_from_html(
        html,
        "https://www.selgros.de/sites/default/files/offers/22182_ALLE_20260402_20260408_Food_F/index.html",
    )

    assert pdf_url == (
        "https://www.selgros.de/sites/default/files/offers/"
        "22182_ALLE_20260402_20260408_Food_F/docs/"
        "22182_ALLE_20260402_20260408_Food_F_kl.pdf"
    )


def test_cached_brochures_are_loaded_with_viewer_metadata(tmp_path):
    scraper = make_scraper()
    pdf_path = tmp_path / "22182_ALLE_20260402_20260408_Food_F_kl.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    raw = [
        {
            "tab": "Aktuelle Angebote",
            "title": "Food",
            "selected": True,
            "skip_reason": None,
            "viewer_url": (
                "https://www.selgros.de/sites/default/files/offers/"
                "22182_ALLE_20260402_20260408_Food_F/index.html"
            ),
            "pdf_url": (
                "https://www.selgros.de/sites/default/files/offers/"
                "22182_ALLE_20260402_20260408_Food_F/docs/"
                "22182_ALLE_20260402_20260408_Food_F_kl.pdf"
            ),
            "valid_from": "2026-04-02",
            "valid_to": "2026-04-08",
            "category_name": "Food",
            "pdf_status": "downloaded",
            "pdf_local_path": str(pdf_path),
        }
    ]
    (tmp_path / "raw_brochures.json").write_text(json.dumps(raw), encoding="utf-8")

    docs = scraper._load_cached_brochures(tmp_path, week=15, year=2026)

    assert len(docs) == 1
    assert docs[0].url == raw[0]["viewer_url"]
    assert docs[0].title == "Food"
    assert docs[0].tab == "Aktuelle Angebote"
    assert docs[0].category == "Food"


def test_extract_selgros_filename_meta_handles_underscored_categories():
    meta = extract_selgros_filename_meta("22182_ALLE_20260402_20260408_Bestes_Fleisch_F_kl.pdf")

    assert meta["category"] == "Bestes_Fleisch"
