import csv
import json

import pytest

from golden_set_validator import (
    DEFAULT_FIXTURE,
    FixtureValidationError,
    load_fixture,
    load_records,
    main,
    validate_fixture,
    validate_records,
)


def fixture_with_cases(*case_ids: str) -> dict:
    fixture = load_fixture(DEFAULT_FIXTURE)
    wanted = set(case_ids)
    fixture["cases"] = [case for case in fixture["cases"] if case["case_id"] in wanted]
    assert {case["case_id"] for case in fixture["cases"]} == wanted
    return fixture


def case_result(report: dict, case_id: str) -> dict:
    return next(case for case in report["cases"] if case["case_id"] == case_id)


def test_kw32_fixture_is_versioned_policy_only_and_schema_valid():
    fixture = load_fixture(DEFAULT_FIXTURE)

    assert fixture["fixture_id"] == "kw32_2026_customer_feedback"
    assert fixture["fixture_version"] == 1
    assert fixture["calendar_week"] == 32
    assert fixture["year"] == 2026
    assert fixture["fixture_mode"] == "expected_policy"
    assert fixture["raw_snapshot_available"] is False
    assert len(fixture["cases"]) >= 20
    assert validate_fixture(fixture) == []
    assert all(case["source_reference"]["raw_row_available"] is False for case in fixture["cases"])


def test_kw32_fixture_preserves_confirmed_brand_and_size_policy():
    fixture = load_fixture(DEFAULT_FIXTURE)
    policy = fixture["confirmed_policy"]

    assert policy["packaged_exception_brands"] == [
        "ARO",
        "Chef",
        "Metro",
        "Milram",
        "Schleiz",
        "Quality",
        "economy",
        "Edeka",
        "Foodservice",
        "Henkelmann",
        "Meemken",
        "Aviko",
    ]
    assert policy["minimum_package_grams"] == {"Cheese": 500, "Sausage": 500}


def test_fixture_validation_rejects_missing_raw_source_status():
    fixture = fixture_with_cases("relevance_negative_toilet_paper")
    fixture["cases"][0]["source_reference"].pop("raw_row_available")

    errors = validate_fixture(fixture)

    assert any("raw_row_available" in error for error in errors)
    with pytest.raises(FixtureValidationError, match="raw_row_available"):
        validate_records(fixture, [])


def test_load_records_accepts_json_jsonl_and_csv(tmp_path):
    expected = [
        {"product_name": "Toilettenpapier", "Relevant": "Nein"},
        {"product_name": "Cherry-Strauchtomaten", "Relevant": "Ja"},
    ]
    json_path = tmp_path / "results.json"
    json_path.write_text(json.dumps({"records": expected}, ensure_ascii=False), encoding="utf-8")
    jsonl_path = tmp_path / "results.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in expected) + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_name", "Relevant"])
        writer.writeheader()
        writer.writerows(expected)

    assert load_records(json_path) == expected
    assert load_records(jsonl_path) == expected
    assert load_records(csv_path) == expected


def test_relevance_case_accepts_csv_style_field_and_german_value_aliases():
    fixture = fixture_with_cases("relevance_negative_toilet_paper")

    report = validate_records(
        fixture,
        [{"product_name": "Premium Toilettenpapier", "Relevant": "Nein"}],
    )

    assert report["passed"] is True
    assert case_result(report, "relevance_negative_toilet_paper")["status"] == "passed"


def test_relevance_case_reports_wrong_decision():
    fixture = fixture_with_cases("relevance_negative_guacamole")

    report = validate_records(
        fixture,
        [{"product_name": "Avocado Guacamole mild", "Relevant": "Ja"}],
    )

    result = case_result(report, "relevance_negative_guacamole")
    assert report["passed"] is False
    assert result["status"] == "failed"
    assert "field_equals" in result["errors"][0]


def test_positive_plain_meat_selector_does_not_claim_marinated_record():
    fixture = fixture_with_cases("relevance_positive_pork_strips_when_plain")

    report = validate_records(
        fixture,
        [
            {
                "product_name": "Schweinegeschnetzeltes",
                "description": "mariniert in Kräutersauce",
                "Relevant": "Nein",
            }
        ],
        allow_missing=True,
    )

    result = case_result(report, "relevance_positive_pork_strips_when_plain")
    assert report["passed"] is True
    assert result["status"] == "not_evaluable"
    assert result["matched_records"] == 0


def test_grouping_case_requires_same_family_variant_and_distinct_offers():
    fixture = fixture_with_cases("group_pfifferlinge_offer_variants")
    records = [
        {
            "product_name": "Pfifferlinge",
            "supplier": "Selgros",
            "description": "mittelfallend",
            "product_family_id": "family:pfifferlinge",
            "variant_id": "variant:pfifferlinge:mittelfallend:ua-ro",
            "offer_id": "offer:selgros:pfifferlinge:28.99",
        },
        {
            "product_name": "Pfifferlinge",
            "supplier": "Selgros",
            "description": "mittelfallend",
            "product_family_id": "family:pfifferlinge",
            "variant_id": "variant:pfifferlinge:mittelfallend:ua-ro",
            "offer_id": "offer:selgros:pfifferlinge:11.99",
        },
    ]

    report = validate_records(fixture, records)

    assert report["passed"] is True
    assert case_result(report, "group_pfifferlinge_offer_variants")["matched_records"] == 2


def test_grouping_case_fails_when_offer_identity_is_reused():
    fixture = fixture_with_cases("group_pfifferlinge_offer_variants")
    records = [
        {
            "product_name": "Pfifferlinge",
            "supplier": "Selgros",
            "description": "mittelfallend",
            "family_id": "family:pfifferlinge",
            "variant_id": "variant:pfifferlinge",
            "offer_id": "offer:same",
        },
        {
            "product_name": "Pfifferlinge",
            "supplier": "Selgros",
            "description": "mittelfallend",
            "family_id": "family:pfifferlinge",
            "variant_id": "variant:pfifferlinge",
            "offer_id": "offer:same",
        },
    ]

    report = validate_records(fixture, records)

    result = case_result(report, "group_pfifferlinge_offer_variants")
    assert report["passed"] is False
    assert result["status"] == "failed"
    assert any("expected distinct values" in error for error in result["errors"])


def test_roastbeef_case_keeps_family_but_separates_temperature_variants():
    fixture = fixture_with_cases("variant_roastbeef_fresh_vs_frozen")
    records = [
        {
            "product": "Rinder-Roastbeef",
            "product_family_id": "family:rinder-roastbeef",
            "variant_id": "variant:fresh:uy",
            "offer_id": "offer:fresh",
            "temperature_state": "fresh",
        },
        {
            "product": "Rinder Roastbeef",
            "product_family_id": "family:rinder-roastbeef",
            "variant_id": "variant:frozen:ar",
            "offer_id": "offer:frozen",
            "temperature_state": "frozen",
        },
    ]

    report = validate_records(fixture, records)

    assert report["passed"] is True


def test_red_snapper_guard_rejects_rotbarsch_mapping():
    fixture = fixture_with_cases("normalization_block_red_snapper_to_rotbarsch")

    report = validate_records(
        fixture,
        [
            {
                "original_product_name": "Red Snapper Filet",
                "canonical_product_name": "Rotbarsch Filet",
            }
        ],
    )

    result = case_result(report, "normalization_block_red_snapper_to_rotbarsch")
    assert report["passed"] is False
    assert result["status"] == "failed"


def test_translation_case_accepts_normalized_punctuation_and_umlauts():
    fixture = fixture_with_cases("normalization_translate_sweet_potato_fries")

    report = validate_records(
        fixture,
        [
            {
                "original_product_name": "Sweet Potato Fries",
                "canonical_product_name": "Süßkartoffel-Pommes Frites",
            }
        ],
    )

    assert report["passed"] is True


def test_missing_cases_can_be_reported_without_failing_partial_outputs():
    fixture = fixture_with_cases("relevance_negative_toilet_paper")

    report = validate_records(fixture, [], allow_missing=True)

    assert report["passed"] is True
    assert report["summary"] == {"not_evaluable": 1}


def test_confirmed_forelle_calibre_policy_requires_one_variant_and_two_offers():
    fixture = fixture_with_cases("variant_forelle_calibre_is_non_blocking")
    report = validate_records(
        fixture,
        [
            {
                "product": "Forelle",
                "product_family_id": "family:forelle",
                "variant_id": "variant:forelle",
                "offer_id": "offer:200-400",
                "calibre": "200-400 g",
            },
            {
                "product": "Forelle",
                "product_family_id": "family:forelle",
                "variant_id": "variant:forelle",
                "offer_id": "offer:300-400",
                "calibre": "300-400 g",
            },
        ],
    )

    assert report["passed"] is True


def test_cli_can_validate_fixture_without_local_kw32_results(capsys):
    exit_code = main(["--fixture", str(DEFAULT_FIXTURE)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["fixture_valid"] is True
    assert output["results_validated"] is False
    assert output["case_count"] >= 20
