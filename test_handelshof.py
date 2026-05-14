import json
from pathlib import Path

import src.acquire.handelshof as handelshof_module
from src.acquire.handelshof import HandelshofScraper


FIXTURE_CONFIG = {
    "headline": "Aktuelle Werbung",
    "tabs": [
        {
            "title": "Aktuell",
            "entries": [
                {
                    "title": "Living & Lifestyle",
                    "validFrom": "2026-03-19 00:01",
                    "validTo": "2026-06-24 23:59",
                    "downloadLink": "/dam/jcr:646b4bc3-e118-44c8-86d1-93eaa8c30f79/hh-2026-kw12-living-und-lifestyle-sonderwerbung.pdf",
                    "articleListDownloadLink": "/dam/jcr:f698d4da-c3a4-4262-9899-53c0f7ec0487/kundeninfo-living-lifestyle-sonderwerbung-kw12.xlsx",
                    "pdfUrl": "https://werbung.handelshof.de/html5/hh_2026_kw12_living_und_lifestyle_sonderwerbung/index.html",
                },
                {
                    "title": "Aktuell [Groß]",
                    "validFrom": "2026-04-02 00:01",
                    "validTo": "2026-04-08 23:59",
                    "downloadLink": "/dam/jcr:c9c02eb1-8900-4d62-ab8a-5795d9908268/hh-2026-kw14-aktuell-gross.pdf",
                    "articleListDownloadLink": "/dam/jcr:cf50f10a-bfc3-4942-bbb3-0722e5ca6cad/kundeninfo-aktuell-gross-kw14.xlsx",
                    "pdfUrl": "https://werbung.handelshof.de/html5/hh_2026_kw14_aktuell_gross/index.html",
                },
                {
                    "title": "Aktuell [Medium]",
                    "validFrom": "2026-04-02 00:01",
                    "validTo": "2026-04-08 23:59",
                    "downloadLink": "/dam/jcr:cd1a9a26-91a3-41b2-a386-3c5f6f372fec/hh-2026-kw14-aktuell-medium.pdf",
                    "articleListDownloadLink": "/dam/jcr:1299cb69-dee6-4bd1-b7fe-ce6dc4299b50/kundeninfo-aktuell-medium-kw14.xlsx",
                    "pdfUrl": "https://werbung.handelshof.de/html5/hh_2026_kw14_aktuell_medium/index.html",
                },
                {
                    "title": "Gastro & GV",
                    "validFrom": "2026-04-01 00:01",
                    "validTo": "2026-04-30 23:59",
                    "downloadLink": "/dam/jcr:f500626c-6487-427b-a75e-eaba96ddfc38/hh-2026-kw14-gastro-gv.pdf",
                    "articleListDownloadLink": "",
                    "pdfUrl": "https://werbung.handelshof.de/html5/hh_2026_kw14_gastro_gv/index.html",
                },
            ],
        },
        {
            "title": "Demnächst",
            "entries": [
                {
                    "title": "Aktuell [Groß]",
                    "validFrom": "2026-04-09 00:01",
                    "validTo": "2026-04-15 23:59",
                    "downloadLink": "/dam/jcr:b12704fa-335e-4414-8603-3f8da3c7cc9e/hh-2026-kw15-aktuell-gross.pdf",
                    "articleListDownloadLink": "/dam/jcr:1a33460b-adb7-4449-897f-9a17f9449697/kundeninfo-aktuell-gross-kw15.xlsx",
                    "pdfUrl": "https://werbung.handelshof.de/html5/hh_2026_kw15_aktuell_gross/index.html",
                },
                {
                    "title": "Aktuell [Medium]",
                    "validFrom": "2026-04-09 00:01",
                    "validTo": "2026-04-15 23:59",
                    "downloadLink": "/dam/jcr:1c0562ed-630b-41f0-84e1-ee6b7336f050/hh-2026-kw15-aktuell-medium.pdf",
                    "articleListDownloadLink": "/dam/jcr:532967e9-29fb-4597-82d8-73b545471e80/kundeninfo-aktuell-medium-kw15.xlsx",
                    "pdfUrl": "https://werbung.handelshof.de/html5/hh_2026_kw15_aktuell_medium/index.html",
                },
            ],
        },
        {
            "title": "Bestellkataloge",
            "entries": [
                {
                    "title": "Eiskonzept 2026",
                    "validFrom": "2026-01-01 11:11",
                    "validTo": "2026-12-31 23:59",
                    "downloadLink": "/dam/jcr:09c85712-7efa-4326-967d-e71dfae583ad/hh-eiskonzept-2026.pdf",
                    "articleListDownloadLink": "",
                    "pdfUrl": "https://werbung.handelshof.de/html5/hh_eiskonzept_2026/index.html",
                },
            ],
        },
    ],
}

FIXTURE_HTML = (
    "<html><body><cmp-brochure-preview data-config='"
    + json.dumps(FIXTURE_CONFIG, ensure_ascii=False)
    + "'></cmp-brochure-preview></body></html>"
)


def make_scraper() -> HandelshofScraper:
    return HandelshofScraper({
        "location": "hannover",
        "url": "https://www.handelshof.de/angebote/hannover",
    })


def test_parse_brochure_preview_selects_current_and_next_tabs():
    scraper = make_scraper()

    brochures = scraper._parse_brochure_preview(
        FIXTURE_HTML,
        "https://www.handelshof.de/angebote/hannover",
    )

    assert len(brochures) == 7
    assert sum(1 for item in brochures if item["tab"] == "Aktuell") == 4
    assert sum(1 for item in brochures if item["tab"] == "Demnächst") == 2
    assert sum(1 for item in brochures if item["tab"] == "Bestellkataloge") == 1
    assert sum(1 for item in brochures if item["selected"]) == 6
    assert sum(1 for item in brochures if item["article_list_url"]) == 5

    first = brochures[0]
    assert first["download_url"] == (
        "https://www.handelshof.de/dam/jcr:646b4bc3-e118-44c8-86d1-93eaa8c30f79/"
        "hh-2026-kw12-living-und-lifestyle-sonderwerbung.pdf"
    )
    assert first["viewer_url"] == (
        "https://werbung.handelshof.de/html5/"
        "hh_2026_kw12_living_und_lifestyle_sonderwerbung/index.html"
    )
    assert first["valid_from"].isoformat() == "2026-03-19"
    assert first["valid_to"].isoformat() == "2026-06-24"

    bestell = next(item for item in brochures if item["tab"] == "Bestellkataloge")
    assert bestell["selected"] is False
    assert bestell["skip_reason"] == "ignored_tab"


def test_download_pdf_uses_viewer_fallback_when_direct_pdf_is_invalid(tmp_path, monkeypatch):
    scraper = make_scraper()
    brochure = {
        "title": "Aktuell [Medium]",
        "download_url": "https://www.handelshof.de/dam/jcr:test/hh-2026-kw14-aktuell-medium.pdf",
        "viewer_url": "https://werbung.handelshof.de/html5/hh_2026_kw14_aktuell_medium/index.html",
    }

    class FakeResponse:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url: str, timeout: int = 60):
            assert url == brochure["download_url"]
            return FakeResponse(b"not-a-pdf")

    calls: list[Path] = []

    def fake_validate_pdf(pdf_path: str, min_pages: int = 1):
        calls.append(Path(pdf_path))
        if len(calls) == 1:
            raise RuntimeError("Invalid PDF header")
        return 2

    def fake_viewer_fallback(self, viewer_url: str, dest: Path):
        assert viewer_url == brochure["viewer_url"]
        dest.write_bytes(b"%PDF-1.4\n% fallback\n")
        handelshof_module.validate_pdf(str(dest))
        return dest

    monkeypatch.setattr(handelshof_module, "validate_pdf", fake_validate_pdf)
    monkeypatch.setattr(
        HandelshofScraper,
        "_download_pdf_from_viewer",
        fake_viewer_fallback,
    )

    result = scraper._download_pdf_entry(FakeSession(), brochure, tmp_path)

    assert result == tmp_path / "hh-2026-kw14-aktuell-medium.pdf"
    assert result.exists()
    assert len(calls) == 2


def test_all_tabs_are_downloaded_but_only_selected_tabs_are_processed(tmp_path, monkeypatch):
    scraper = make_scraper()

    monkeypatch.setattr(scraper, "_fetch_offers_page", lambda url: FIXTURE_HTML)

    def fake_download_article_list(self, session, brochure, data_dir):
        if not brochure.get("article_list_url"):
            return None
        path = data_dir / f"{brochure['title']}.xlsx"
        path.write_text("xlsx", encoding="utf-8")
        return path

    def fake_download_pdf_entry(self, session, brochure, data_dir):
        path = data_dir / f"{brochure['title']}.pdf"
        path.write_bytes(b"%PDF-1.4\n")
        return path

    monkeypatch.setattr(
        HandelshofScraper,
        "_download_article_list",
        fake_download_article_list,
    )
    monkeypatch.setattr(
        HandelshofScraper,
        "_download_pdf_entry",
        fake_download_pdf_entry,
    )

    docs = scraper._scrape_and_download(week=15, year=2026, data_dir=tmp_path)
    raw = json.loads((tmp_path / "raw_brochures.json").read_text(encoding="utf-8"))

    assert len(raw) == 7
    assert sum(1 for item in raw if item["pdf_status"] == "downloaded") == 7
    assert sum(1 for item in raw if item["xlsx_status"] == "downloaded") == 5
    assert len(docs) == 6
    assert {doc.tab for doc in docs} == {"Aktuell", "Demnächst"}
