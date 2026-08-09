import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import extract_single_cached_pdf as single_pdf
import recover_failed_cached_extraction as recovery
from src.extract import vision


def _valid_item(name="Rinderfilet", price=19.99):
    return {
        "product_name": name,
        "description": None,
        "category": "fleisch",
        "unit": "kg",
        "quantity": 1,
        "price": price,
        "confidence": 0.95,
    }


def _document():
    return SimpleNamespace(
        supplier="metro",
        file_path="data/metro/2026/32/offer.pdf",
        calendar_week=32,
        year=2026,
        location="goslar",
        title="Offer",
        tab="Aktuell",
        valid_from=None,
        valid_to=None,
    )


def test_clear_primary_does_not_call_central_retry_executor():
    calls = []

    outcome = vision.apply_quality_retry_once(
        [_valid_item()],
        image_path="page-1.png",
        supplier="metro",
        primary_model="gemini-2.5-flash",
        retry_executor=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert calls == []
    assert outcome.quality_retry_attempted is False
    assert outcome.quality_retry_status == "not_needed"
    assert outcome.selected_model == "gemini-2.5-flash"


@pytest.mark.parametrize(
    "primary",
    [
        [{"product_name": "Rinderfilet", "price": None, "confidence": 0.5}],
        [{"product_name": "@@@###", "price": 19.99, "confidence": 0.9}],
        [
            {
                "product_name": "TK Gemüse",
                "description": "800 g Beutel",
                "unit": "beutel",
                "quantity": 800,
                "price": 3.49,
                "confidence": 0.9,
            }
        ],
    ],
)
def test_unclear_primary_calls_hard_locked_retry_exactly_once(primary):
    calls = []

    def retry(*args, **kwargs):
        calls.append((args, kwargs))
        return [_valid_item()]

    outcome = vision.apply_quality_retry_once(
        primary,
        image_path="page-1.png",
        supplier="metro",
        primary_model="gemini-2.5-flash",
        retry_executor=retry,
    )

    assert len(calls) == 1
    assert calls[0][1]["model_name"] == "gemini-3.6-flash"
    assert calls[0][1]["max_retries"] == 1
    assert calls[0][1]["operation"] == "product_extraction_quality_retry"
    assert outcome.quality_retry_attempted is True
    assert outcome.quality_retry_model == "gemini-3.6-flash"


def test_empty_primary_is_unclear_and_gets_one_retry():
    calls = []

    def retry(*args, **kwargs):
        calls.append(kwargs)
        return [_valid_item()]

    outcome = vision.apply_quality_retry_once(
        [],
        image_path="page-1.png",
        supplier="metro",
        primary_model="gemini-2.5-flash",
        retry_executor=retry,
    )

    assert len(calls) == 1
    assert outcome.quality_retry_status == "selected"


def test_packaging_quality_requires_count_total_and_keeps_multipack_valid():
    simple_incomplete = {
        **_valid_item("TK Gemüse", 3.49),
        "description": "2500 g Beutel",
        "package_size_value": 2500,
        "package_size_unit": "g",
        "packaging_type": "bag",
        "price_basis": "per_package",
    }
    multipack = {
        **_valid_item("Burger Patties", 8.99),
        "description": "10 Stück à 80 g, Gesamt 800 g pro Packung",
        "price_basis": "per_package",
        "package_count": 10,
        "package_size_value": 80,
        "package_size_unit": "g",
        "total_content_value": 800,
        "total_content_unit": "g",
        "packaging_type": "pack",
    }

    simple_issues = vision.extraction_quality_issues([simple_incomplete])
    assert any("package_count 1" in issue for issue in simple_issues)
    assert any("Gesamtinhalt" in issue for issue in simple_issues)
    assert vision.extraction_quality_issues([multipack]) == []


def test_calibre_count_and_certification_as_brand_trigger_quality_retry():
    item = {
        **_valid_item("ASC Garnelen", 17.99),
        "description": "100/200 Stück per lb, 800 g Beutel",
        "calibre": "100/200 Stück/lb",
        "source_brand": "ASC",
        "price_basis": "per_package",
        "package_count": 100,
        "package_size_value": 800,
        "package_size_unit": "g",
        "total_content_value": 800,
        "total_content_unit": "g",
        "packaging_type": "bag",
    }

    issues = vision.extraction_quality_issues([item])
    assert any("Kaliber Stück/lb" in issue for issue in issues)
    assert any("Zertifizierung wurde als Marke" in issue for issue in issues)


def test_retry_with_lower_recall_never_replaces_primary():
    primary = [
        _valid_item("Rinderfilet"),
        {"product_name": "Rinderhüfte", "price": None, "confidence": 0.4},
    ]
    outcome = vision.apply_quality_retry_once(
        primary,
        image_path="page-1.png",
        supplier="metro",
        primary_model="gemini-2.5-flash",
        retry_executor=lambda *args, **kwargs: [_valid_item("Rinderfilet")],
    )

    assert outcome.quality_retry_status == "kept_primary"
    assert outcome.selected_items == primary
    assert outcome.selected_model == "gemini-2.5-flash"


@pytest.mark.parametrize("retry_result", [[], [{"product_name": "@@@", "price": None}]])
def test_bad_or_empty_retry_keeps_nonempty_primary_without_recursion(retry_result):
    calls = []
    primary = [{**_valid_item(), "confidence": 0.5}]

    def retry(*args, **kwargs):
        calls.append(kwargs)
        return retry_result

    outcome = vision.apply_quality_retry_once(
        primary,
        image_path="page-1.png",
        supplier="metro",
        primary_model="gemini-2.5-flash",
        retry_executor=retry,
    )

    assert len(calls) == 1
    assert outcome.quality_retry_status == "kept_primary"
    assert outcome.selected_items == primary


def test_retry_exception_marks_failed_and_keeps_primary():
    primary = [{**_valid_item(), "confidence": 0.5}]

    def fail(*args, **kwargs):
        raise TimeoutError("quality timeout")

    outcome = vision.apply_quality_retry_once(
        primary,
        image_path="page-1.png",
        supplier="metro",
        primary_model="gemini-2.5-flash",
        retry_executor=fail,
    )

    assert outcome.quality_retry_status == "failed"
    assert outcome.selected_items == primary
    assert outcome.retry_items is None


def test_selected_kept_and_failed_outcomes_flow_into_product_provenance():
    primary = [{**_valid_item(), "confidence": 0.5}]
    outcomes = [
        vision.apply_quality_retry_once(
            primary,
            image_path="page.png",
            supplier="metro",
            primary_model="gemini-2.5-flash",
            retry_executor=lambda *args, **kwargs: [_valid_item(price=20.99)],
        ),
        vision.apply_quality_retry_once(
            primary,
            image_path="page.png",
            supplier="metro",
            primary_model="gemini-2.5-flash",
            retry_executor=lambda *args, **kwargs: [],
        ),
        vision.apply_quality_retry_once(
            primary,
            image_path="page.png",
            supplier="metro",
            primary_model="gemini-2.5-flash",
            retry_executor=lambda *args, **kwargs: None,
        ),
    ]

    assert [outcome.quality_retry_status for outcome in outcomes] == [
        "selected",
        "kept_primary",
        "failed",
    ]
    for outcome in outcomes:
        products = vision._raw_items_to_products(
            outcome.selected_items,
            supplier="metro",
            source_page=1,
            source_document_sha256="a" * 64,
            extraction_outcome=outcome,
        )
        assert products[0].quality_retry_status == outcome.quality_retry_status
        assert products[0].quality_retry_attempted is True
        assert products[0].quality_retry_model == "gemini-3.6-flash"
        assert products[0].selected_extraction_model == outcome.selected_model


def test_three_primary_process_attempts_still_allow_only_one_quality_process(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        mode = command[command.index("--worker-mode") + 1]
        calls.append(mode)
        if mode == "primary" and calls.count("primary") < 3:
            raise subprocess.CalledProcessError(1, command)
        if mode == "primary":
            items = [{"product_name": "Rinderfilet", "price": None, "confidence": 0.4}]
        else:
            items = [_valid_item()]
        return SimpleNamespace(
            stdout=json.dumps(
                {"status": "ok", "items": items, "model_name": "test", "worker_mode": mode}
            )
        )

    monkeypatch.setattr(single_pdf.subprocess, "run", fake_run)
    page_image = Path("page-1.png")
    document = _document()
    primary = single_pdf.run_page_subprocess(
        Path("extract_single_cached_pdf.py"),
        "config.yaml",
        page_image,
        document,
        None,
        30,
        3,
        worker_mode="primary",
    )
    outcome = single_pdf.apply_single_page_quality_retry(
        primary_items=primary,
        primary_model="gemini-2.5-flash",
        script_path=Path("extract_single_cached_pdf.py"),
        config_path="config.yaml",
        page_image=page_image,
        document=document,
        fallback_category=None,
        timeout_seconds=30,
    )

    assert calls.count("primary") == 3
    assert calls.count("quality_retry") == 1
    assert outcome.quality_retry_status == "selected"


def test_single_quality_timeout_is_not_retried(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        mode = command[command.index("--worker-mode") + 1]
        calls.append(mode)
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr(single_pdf.subprocess, "run", fake_run)
    outcome = single_pdf.apply_single_page_quality_retry(
        primary_items=[{**_valid_item(), "confidence": 0.5}],
        primary_model="gemini-2.5-flash",
        script_path=Path("extract_single_cached_pdf.py"),
        config_path="config.yaml",
        page_image=Path("page-1.png"),
        document=_document(),
        fallback_category=None,
        timeout_seconds=30,
    )

    assert calls == ["quality_retry"]
    assert outcome.quality_retry_status == "failed"


def test_page_manifest_keeps_zero_product_outcome_auditable(tmp_path):
    outcome = vision.PageExtractionOutcome(
        primary_items=[],
        selected_items=[],
        primary_model="gemini-2.5-flash",
        selected_model="gemini-2.5-flash",
        primary_failed=False,
        primary_quality_issues=("keine Produkte erkannt",),
        quality_retry_attempted=True,
        quality_retry_status="kept_primary",
        quality_retry_model="gemini-3.6-flash",
        retry_items=[],
        retry_quality_issues=("keine Produkte erkannt",),
    )
    record = outcome.to_manifest_record(source_page=4)
    record["accepted_product_count"] = 0

    path = single_pdf.write_page_outcome_manifest(tmp_path / "result.csv", [record])
    persisted = json.loads(path.read_text(encoding="utf-8").strip())

    assert persisted["source_page"] == 4
    assert persisted["accepted_product_count"] == 0
    assert persisted["quality_retry_status"] == "kept_primary"


def test_batch_persists_page_outcome_hashes_and_actual_document_sha(tmp_path, monkeypatch):
    source_pdf = tmp_path / "offer.pdf"
    source_pdf.write_bytes(b"%PDF source")
    image_dir = tmp_path / "images" / "offer"
    image_dir.mkdir(parents=True)
    page_image = image_dir / "page-1.png"
    page_image.write_bytes(b"png")
    monkeypatch.setattr(
        vision,
        "extract_products_from_image",
        lambda *args, **kwargs: [_valid_item()],
    )

    products = vision.extract_products_from_pdf_images(
        [str(page_image)],
        supplier="metro",
        source_file=str(source_pdf),
        model_name="gemini-2.5-flash",
        max_concurrent_requests=1,
    )
    manifest = image_dir / "extraction_outcomes.jsonl"
    record = json.loads(manifest.read_text(encoding="utf-8").strip())

    assert products[0].source_document_sha256 == vision.file_sha256(source_pdf)
    assert products[0].source_item_id
    assert record["page_complete"] is True
    assert record["primary_items_sha256"]
    assert record["selected_items_sha256"] == record["primary_items_sha256"]
    assert record["retry_items_sha256"] is None


def test_batch_manifest_marks_unresolved_kept_primary_quality_issue_incomplete(
    tmp_path,
    monkeypatch,
):
    source_pdf = tmp_path / "offer.pdf"
    source_pdf.write_bytes(b"%PDF source")
    image_dir = tmp_path / "images" / "offer"
    image_dir.mkdir(parents=True)
    page_image = image_dir / "page-1.png"
    page_image.write_bytes(b"png")

    def fake_extract(*args, **kwargs):
        if kwargs.get("operation") == "product_extraction_quality_retry":
            return []
        return [{**_valid_item(), "confidence": 0.5}]

    monkeypatch.setattr(vision, "extract_products_from_image", fake_extract)
    vision.extract_products_from_pdf_images(
        [str(page_image)],
        supplier="metro",
        source_file=str(source_pdf),
        model_name="gemini-2.5-flash",
        max_concurrent_requests=1,
    )
    record = json.loads(
        (image_dir / "extraction_outcomes.jsonl").read_text(encoding="utf-8").strip()
    )

    assert record["quality_retry_status"] == "kept_primary"
    assert record["page_complete"] is False


def test_recovery_failure_regex_accepts_configured_attempt_counts(tmp_path):
    log_file = tmp_path / "run.log"
    log_file.write_text(
        "Failed to analyze images/metro/2026/KW32/offer/page-4.png after 8 attempts\n",
        encoding="utf-8",
    )

    assert recovery.parse_failed_document_keys(log_file, 32, 2026) == [
        ("metro", 32, 2026, "offer")
    ]


def test_recovery_refuses_incomplete_manifest_before_source_replacement(tmp_path):
    source = tmp_path / "offer.pdf"
    source.write_bytes(b"%PDF source")
    document = _document()
    document.file_path = str(source)
    images_dir = tmp_path / "images"
    manifest = recovery.extraction_outcome_manifest_path(document, str(images_dir))
    manifest.parent.mkdir(parents=True)
    source_hash = vision.file_sha256(source)

    incomplete = {
        "source_page": 1,
        "source_document_sha256": source_hash,
        "page_complete": False,
        "selected_item_count": 1,
        "accepted_product_count": 0,
    }
    manifest.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
    assert recovery.document_extraction_is_complete(document, str(images_dir)) is False

    complete = {**incomplete, "page_complete": True, "accepted_product_count": 1}
    manifest.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    assert recovery.document_extraction_is_complete(document, str(images_dir)) is True
