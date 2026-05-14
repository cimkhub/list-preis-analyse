from pathlib import Path
from datetime import date

from src.extract.pdf_relevance import (
    _filename_hints,
    _normalize_decision,
    filter_relevant_documents,
)
from src.models import AcquiredDocument


def make_document(tmp_path, filename: str, title: str | None = None) -> AcquiredDocument:
    pdf_path = tmp_path / filename
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return AcquiredDocument(
        supplier="metro",
        location="goslar",
        doc_type="pdf",
        file_path=str(pdf_path),
        title=title or pdf_path.stem,
    )


def test_filename_hints_detect_irrelevant_non_food_terms():
    hints = _filename_hints(
        "metro-professional-non-food-flyer.pdf",
        "Metro Professional Non Food Flyer",
    )

    assert "non food" in hints["irrelevant_hits"] or "non-food" in hints["irrelevant_hits"]


def test_filter_relevant_documents_saves_decisions_and_skips_irrelevant(tmp_path, monkeypatch):
    relevant_doc = make_document(tmp_path, "wochen-angebote-060426-110426.pdf")
    irrelevant_doc = make_document(tmp_path, "metro-professional-non-food-flyer.pdf")

    def fake_classify(document, supplier, images_base_dir="images"):
        if "non-food" in Path(document.file_path).name:
            return {
                "is_relevant": False,
                "relevance_label": "irrelevant_non_food_only",
                "relevance_reason": "Nur Non-Food auf der Titelseite.",
                "relevance_confidence": 0.95,
            }
        return {
            "is_relevant": True,
            "relevance_label": "relevant_food_offer",
            "relevance_reason": "Normales Angebotsheft mit Lebensmittelpreisen.",
            "relevance_confidence": 0.97,
        }

    monkeypatch.setattr(
        "src.extract.pdf_relevance.classify_document_relevance",
        fake_classify,
    )

    relevant, skipped = filter_relevant_documents(
        [relevant_doc, irrelevant_doc],
        supplier="metro",
        images_base_dir="images",
    )

    assert len(relevant) == 1
    assert len(skipped) == 1
    assert relevant[0].relevance_label == "relevant_food_offer"
    assert skipped[0].relevance_label == "irrelevant_non_food_only"

    decisions_file = tmp_path / "relevance_decisions.json"
    assert decisions_file.exists()
    saved = decisions_file.read_text(encoding="utf-8")
    assert "relevant_food_offer" in saved
    assert "irrelevant_non_food_only" in saved


def test_filter_relevant_documents_reuses_cached_decisions(tmp_path, monkeypatch):
    document = make_document(tmp_path, "wochen-angebote-060426-110426.pdf")
    calls = {"count": 0}

    def fake_classify(document, supplier, images_base_dir="images"):
        calls["count"] += 1
        return {
            "is_relevant": True,
            "relevance_label": "relevant_food_offer",
            "relevance_reason": "Normales Angebotsheft.",
            "relevance_confidence": 0.9,
        }

    monkeypatch.setattr(
        "src.extract.pdf_relevance.classify_document_relevance",
        fake_classify,
    )

    filter_relevant_documents([document], supplier="metro", images_base_dir="images")
    assert calls["count"] == 1

    document_again = make_document(tmp_path, "wochen-angebote-060426-110426.pdf")
    filter_relevant_documents([document_again], supplier="metro", images_base_dir="images")
    assert calls["count"] == 1
    assert document_again.relevance_label == "relevant_food_offer"


def test_relevance_decision_applies_visible_validity_period(tmp_path):
    document = make_document(tmp_path, "wochen-angebote-060426-110426.pdf")

    decision = _normalize_decision(
        {
            "is_relevant": True,
            "relevance_label": "relevant_food_offer",
            "reason": "Lebensmittel auf erster Seite.",
            "valid_from": "2026-04-06",
            "valid_to": "2026-04-11",
            "confidence": 0.96,
        },
        document,
        {"relevant_hits": [], "irrelevant_hits": []},
    )

    assert decision["valid_from"] == "2026-04-06"
    assert decision["valid_to"] == "2026-04-11"
    assert document.valid_from == date(2026, 4, 6)
    assert document.valid_to == date(2026, 4, 11)


def test_relevance_decision_keeps_target_market_scope(tmp_path):
    document = make_document(tmp_path, "basis-nord.pdf")

    decision = _normalize_decision(
        {
            "is_relevant": True,
            "relevance_label": "relevant_food_offer",
            "reason": "Lebensmittelangebote sichtbar.",
            "market_scope": "specific",
            "valid_markets": ["Aurich", "Hildesheim", "Wernigerode"],
            "confidence": 0.95,
        },
        document,
        {"relevant_hits": [], "irrelevant_hits": []},
    )

    assert decision["is_relevant"] is True
    assert decision["market_scope"] == "specific"
    assert "Hildesheim" in decision["valid_markets"]


def test_relevance_decision_skips_specific_non_target_markets(tmp_path):
    document = make_document(tmp_path, "regionaler-flyer.pdf")

    decision = _normalize_decision(
        {
            "is_relevant": True,
            "relevance_label": "relevant_food_offer",
            "reason": "Lebensmittelangebote sichtbar.",
            "market_scope": "specific",
            "valid_markets": ["Arnsberg", "Bielefeld", "Bocholt"],
            "confidence": 0.95,
        },
        document,
        {"relevant_hits": [], "irrelevant_hits": []},
    )

    assert decision["is_relevant"] is False
    assert decision["relevance_label"] == "irrelevant_market_scope"
    assert "keinen Zielmarkt" in decision["relevance_reason"]


def test_relevance_decision_keeps_when_no_market_scope_visible(tmp_path):
    document = make_document(tmp_path, "wochen-angebote.pdf")

    decision = _normalize_decision(
        {
            "is_relevant": True,
            "relevance_label": "relevant_food_offer",
            "reason": "Lebensmittelangebote sichtbar.",
            "market_scope": "all",
            "valid_markets": [],
            "confidence": 0.95,
        },
        document,
        {"relevant_hits": [], "irrelevant_hits": []},
    )

    assert decision["is_relevant"] is True
    assert decision["market_scope"] == "all"
    assert decision["valid_markets"] == []


def test_filter_relevant_documents_skips_previous_week_duplicate(tmp_path, monkeypatch):
    previous_dir = tmp_path / "data" / "metro" / "2026" / "14"
    current_dir = tmp_path / "data" / "metro" / "2026" / "15"
    previous_dir.mkdir(parents=True)
    current_dir.mkdir(parents=True)
    pdf_bytes = b"%PDF-1.4\nsame-content\n" * 100
    previous_pdf = previous_dir / "wochen-angebote-alt.pdf"
    current_pdf = current_dir / "wochen-angebote-neu.pdf"
    previous_pdf.write_bytes(pdf_bytes)
    current_pdf.write_bytes(pdf_bytes)

    document = AcquiredDocument(
        supplier="metro",
        location="goslar",
        doc_type="pdf",
        file_path=str(current_pdf),
        title="Wochen Angebote",
        calendar_week=15,
        year=2026,
    )
    calls = {"count": 0}

    def fake_classify(document, supplier, images_base_dir="images"):
        calls["count"] += 1
        return {
            "is_relevant": True,
            "relevance_label": "relevant_food_offer",
            "relevance_reason": "Sollte wegen Duplikat nicht laufen.",
            "relevance_confidence": 0.9,
        }

    monkeypatch.setattr(
        "src.extract.pdf_relevance.classify_document_relevance",
        fake_classify,
    )

    relevant, skipped = filter_relevant_documents(
        [document],
        supplier="metro",
        images_base_dir="images",
    )

    assert calls["count"] == 0
    assert relevant == []
    assert skipped == [document]
    assert document.relevance_label == "duplicate_previous_week"
    assert previous_pdf.name in document.relevance_reason
