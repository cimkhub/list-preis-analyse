import json

import src.acquire.edeka as edeka_module
from src.acquire.edeka import EdekaScraper


FIXTURE_CONFIG = {
    "headline": "Aktuelle Werbung",
    "tabs": [
        {
            "title": "Aktuell",
            "entries": [
                {
                    "title": "Aktuell [Nord]",
                    "validFrom": "2026-04-02 00:01",
                    "validTo": "2026-04-08 23:59",
                    "downloadLink": "/dam/jcr:08e6f531-319b-4933-9834-483f837565fb/efs-2026-kw14-aktuell-basis-nord.pdf",
                    "articleListDownloadLink": "/dam/jcr:411e3fdd-6d8c-4299-bcac-5bbe6e728a48/kundeninfo-aktuell-basis-kw14.xlsx",
                    "pdfUrl": "https://werbung.edeka-foodservice.de/html5/efs_2026_kw14_aktuell_basis_nord/index.html",
                    "catalogCategoryName": "Aktuell",
                    "regionCategoriesNames": "basis-nord",
                },
                {
                    "title": "Fisch",
                    "validFrom": "2026-04-02 00:01",
                    "validTo": "2026-04-08 23:59",
                    "downloadLink": "/dam/jcr:9ec2f0a0-a96b-49cc-80ef-faad9b068889/efs-2026-kw14-fisch-sonderwerbung.pdf",
                    "articleListDownloadLink": "",
                    "pdfUrl": "https://werbung.edeka-foodservice.de/html5/efs_2026_kw14_fisch_sonderwerbung/index.html",
                    "catalogCategoryName": "Fisch-Sonderwerbung",
                    "regionCategoriesNames": "Fisch-Sonderwerbung",
                },
            ],
        },
        {
            "title": "Demnächst",
            "entries": [
                {
                    "title": "Aktuell [Nord]",
                    "validFrom": "2026-04-09 00:01",
                    "validTo": "2026-04-15 23:59",
                    "downloadLink": "/dam/jcr:5d11d4c7-2fc5-4098-ba73-0cb07d476647/efs-2026-kw15-aktuell-basis-nord.pdf",
                    "articleListDownloadLink": "/dam/jcr:6589db52-914f-4281-a1a5-a72bd67a9a22/kundeninfo-aktuell-basis-kw15.xlsx",
                    "pdfUrl": "https://werbung.edeka-foodservice.de/html5/efs_2026_kw15_aktuell_basis_nord/index.html",
                    "catalogCategoryName": "Aktuell",
                    "regionCategoriesNames": "basis-nord",
                },
            ],
        },
        {
            "title": "Bestellkataloge",
            "entries": [
                {
                    "title": "Eiskonzept 2026",
                    "validFrom": "2026-01-01 11:13",
                    "validTo": "2026-12-31 23:59",
                    "downloadLink": "/dam/jcr:985d11e6-a013-4afd-8af3-611be32da7ce/efs-eiskonzept-2026.pdf",
                    "articleListDownloadLink": "",
                    "pdfUrl": "https://werbung.edeka-foodservice.de/html5/efs_eiskonzept_2026/index.html",
                    "catalogCategoryName": "Bestellkataloge",
                    "regionCategoriesNames": "Alle Märkte",
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


def make_scraper() -> EdekaScraper:
    return EdekaScraper({
        "location": "wernigerode",
        "region": "nord",
        "url": "https://www.edeka-foodservice.de/angebote/wernigerode",
    })


def test_parse_brochure_preview_selects_aktuell_and_demnaechst():
    scraper = make_scraper()

    brochures = scraper._parse_brochure_preview(
        FIXTURE_HTML,
        "https://www.edeka-foodservice.de/angebote/wernigerode",
    )

    assert len(brochures) == 4
    assert sum(1 for item in brochures if item["tab"] == "Aktuell") == 2
    assert sum(1 for item in brochures if item["tab"] == "Demnächst") == 1
    assert sum(1 for item in brochures if item["tab"] == "Bestellkataloge") == 1
    assert sum(1 for item in brochures if item["selected"]) == 3

    first = brochures[0]
    assert first["download_url"] == (
        "https://www.edeka-foodservice.de/dam/jcr:08e6f531-319b-4933-9834-483f837565fb/"
        "efs-2026-kw14-aktuell-basis-nord.pdf"
    )
    assert first["viewer_url"] == (
        "https://werbung.edeka-foodservice.de/html5/efs_2026_kw14_aktuell_basis_nord/index.html"
    )
    assert first["valid_from"].isoformat() == "2026-04-02"
    assert first["valid_to"].isoformat() == "2026-04-08"

    bestell = next(item for item in brochures if item["tab"] == "Bestellkataloge")
    assert bestell["selected"] is False
    assert bestell["skip_reason"] == "ignored_tab"


def test_download_pdf_falls_back_to_direct_link_when_viewer_has_no_result(tmp_path, monkeypatch):
    scraper = make_scraper()
    brochure = {
        "title": "Aktuell [Nord]",
        "download_url": "https://www.edeka-foodservice.de/dam/jcr:test/efs-2026-kw14-aktuell-basis-nord.pdf",
        "viewer_url": "https://werbung.edeka-foodservice.de/html5/efs_2026_kw14_aktuell_basis_nord/index.html",
    }

    class FakeResponse:
        content = b"%PDF-1.4\n% pdf\n"

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url: str, timeout: int = 60):
            assert url == brochure["download_url"]
            return FakeResponse()

    calls = []

    def fake_validate_pdf(pdf_path: str, min_pages: int = 1):
        calls.append(pdf_path)
        return 2

    monkeypatch.setattr(EdekaScraper, "_download_pdf_from_viewer", lambda self, viewer_url, dest: None)
    monkeypatch.setattr(edeka_module, "validate_pdf", fake_validate_pdf)

    result = scraper._download_pdf_entry(FakeSession(), brochure, tmp_path)

    assert result == tmp_path / "efs-2026-kw14-aktuell-basis-nord.pdf"
    assert result.exists()
    assert len(calls) == 1


def test_cached_brochures_are_loaded_with_viewer_metadata(tmp_path):
    scraper = make_scraper()
    pdf_path = tmp_path / "efs-2026-kw14-aktuell-basis-nord.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    raw = [
        {
            "tab": "Aktuell",
            "title": "Aktuell [Nord]",
            "selected": True,
            "skip_reason": None,
            "valid_from": "2026-04-02",
            "valid_to": "2026-04-08",
            "download_url": "https://www.edeka-foodservice.de/dam/jcr:test/efs-2026-kw14-aktuell-basis-nord.pdf",
            "viewer_url": "https://werbung.edeka-foodservice.de/html5/efs_2026_kw14_aktuell_basis_nord/index.html",
            "article_list_url": "https://www.edeka-foodservice.de/dam/jcr:test/kundeninfo.xlsx",
            "catalog_category_name": "Aktuell",
            "region_categories_names": "basis-nord",
            "pdf_status": "downloaded",
            "pdf_local_path": str(pdf_path),
        }
    ]
    (tmp_path / "raw_brochures.json").write_text(json.dumps(raw), encoding="utf-8")

    docs = scraper._load_cached_brochures(tmp_path, week=15, year=2026)

    assert len(docs) == 1
    assert docs[0].url == raw[0]["viewer_url"]
    assert docs[0].title == "Aktuell [Nord]"
    assert docs[0].tab == "Aktuell"
    assert docs[0].category == "Aktuell"


def test_parse_card_text_reads_visible_card_metadata():
    scraper = make_scraper()

    meta = scraper._parse_card_text(
        "Aktuell [Nord]\nGültigkeit:\n02.04. - 08.04.2026\nArtikelliste\nDownload\nBlättern",
        "Aktuell",
    )

    assert meta is not None
    assert meta["title"] == "Aktuell [Nord]"
    assert meta["valid_from"].isoformat() == "2026-04-02"
    assert meta["valid_to"].isoformat() == "2026-04-08"


def test_match_visible_brochures_reduces_to_visible_main_page_cards():
    scraper = make_scraper()
    brochures = scraper._parse_brochure_preview(
        FIXTURE_HTML,
        "https://www.edeka-foodservice.de/angebote/wernigerode",
    )
    visible_cards = [
        {
            "tab": "Aktuell",
            "title": "Aktuell [Nord]",
            "valid_from": brochures[0]["valid_from"],
            "valid_to": brochures[0]["valid_to"],
            "download_url": None,
            "viewer_url": None,
        },
        {
            "tab": "Demnächst",
            "title": "Aktuell [Nord]",
            "valid_from": brochures[2]["valid_from"],
            "valid_to": brochures[2]["valid_to"],
            "download_url": None,
            "viewer_url": None,
        },
    ]

    matched = scraper._match_visible_brochures(brochures, visible_cards)

    assert len(matched) == 2
    assert {item["tab"] for item in matched} == {"Aktuell", "Demnächst"}
    assert {item["title"] for item in matched} == {"Aktuell [Nord]"}
