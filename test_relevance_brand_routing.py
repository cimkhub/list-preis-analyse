import json

import pytest

import classify_fresh_food_relevance as relevance


def identity(group: str, *, exclusion_signal: str = "none") -> dict:
    family = {
        "Fleisch": "fleisch",
        "Fisch": "fisch",
        "Obst Gemüse": "obst_gemuese",
        "Sausage": "wurst",
        "Milk": "mopro",
        "Cream": "mopro",
        "Quark": "mopro",
        "Cheese": "mopro",
    }.get(group, "sonstiges")
    return {
        "product_type": group,
        "product_family": family,
        "policy_group": group,
        "temperature_state": "unknown",
        "processing_state": "unknown",
        "source_brand": None,
        "brand_evidence": None,
        "brand_evidence_source": "unknown",
        "exclusion_signal": exclusion_signal,
        "hard_exclusion": exclusion_signal == "explicit_exclusion",
        "exclusion_reason": "Prepared" if exclusion_signal == "explicit_exclusion" else "",
        "confidence": 0.9,
        "evidence": [group],
    }


def policy_payload(
    group: str,
    *,
    decision: str = "include",
    route: str = "packaged_exception",
    required_brand_found=True,
    package_size_grams=None,
) -> str:
    return json.dumps(
        {
            "policy_decision": decision,
            "eligibility_route": route,
            "product_group": group,
            "required_brand_found": required_brand_found,
            "package_size_grams": package_size_grams,
            "rule_id": "ADDITIONAL_PRODUCT" if decision == "include" else "INSUFFICIENT_EVIDENCE",
            "reason": group,
            "confidence": 0.9,
            "review_needed": decision == "uncertain",
            "evidence": [group],
        }
    )


def final_payload(
    decision: str,
    *,
    overrode_stage: str = "none",
    review_needed: bool = False,
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "reason": "final",
            "rule_id": "FINAL_REVIEW",
            "confidence": 0.9,
            "review_needed": review_needed,
            "overrode_stage": overrode_stage,
            "evidence": ["product evidence"],
        }
    )


def approved_brand_proof() -> dict:
    return {
        "brand": "ARO",
        "matched_text": "ARO",
        "source": "product_name",
        "evidence": "ARO Milch",
        "legacy_fallback": False,
    }


def issue_codes(result: dict) -> set[str]:
    return {
        str(issue.get("code"))
        for issue in result.get("contract_issues", [])
        if isinstance(issue, dict)
    }


def blocking_issue_codes(result: dict) -> set[str]:
    return {
        str(issue.get("code"))
        for issue in result.get("contract_issues", [])
        if isinstance(issue, dict) and issue.get("severity") == "blocking"
    }


def test_product_evidence_is_an_exact_allowlist_and_prompts_do_not_leak_metadata():
    row = {
        **{field: f"allowed-{field}" for field in relevance.PRODUCT_EVIDENCE_FIELDS},
        "supplier": "FORBIDDEN_SUPPLIER_9182",
        "location": "FORBIDDEN_LOCATION_9182",
        "source_file": "FORBIDDEN_FILE_9182",
        "source_title": "FORBIDDEN_TITLE_9182",
        "source_tab": "FORBIDDEN_TAB_9182",
        "source_page": "FORBIDDEN_PAGE_9182",
        "price": "FORBIDDEN_PRICE_9182",
        "valid_from": "FORBIDDEN_DATE_9182",
        "brand": "FORBIDDEN_LATER_BRAND_9182",
        "product": "FORBIDDEN_LATER_PRODUCT_9182",
        "arbitrary": "FORBIDDEN_ARBITRARY_9182",
    }
    sanitized = relevance.build_product_evidence(row)
    assert set(sanitized) == set(relevance.PRODUCT_EVIDENCE_FIELDS)

    facts = identity("Milk")
    policy = {
        "policy_decision": "uncertain",
        "eligibility_route": "none",
        "product_group": "Milk",
        "required_brand_found": None,
        "package_size_grams": None,
        "requirements_met": False,
        "rule_id": "INSUFFICIENT_EVIDENCE",
        "reason": "unclear",
        "failure_reason": "unclear",
        "confidence": 0.5,
        "review_needed": True,
        "evidence": ["unclear"],
    }
    prompts = [
        relevance.build_identity_prompt(row),
        relevance.build_eligibility_prompt(row, facts, None),
        relevance.build_final_decision_prompt(row, facts, policy, None),
    ]
    for prompt in prompts:
        assert "FORBIDDEN_" not in prompt


def test_supplier_filename_and_title_never_create_brand_proof():
    row = {
        "product_name": "Milch 1 l",
        "description": "haltbar",
        "supplier": "Metro",
        "source_file": "ARO_Milch.pdf",
        "source_title": "Milram Wochenangebot",
        "source_tab": "Aviko",
    }
    assert relevance.resolve_approved_brand(row) is None


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {
                "product_name": "ARO Milch",
                "source_brand": "ARO",
                "brand_evidence": "ARO Milch",
                "brand_evidence_source": "product_name",
            },
            "ARO",
        ),
        (
            {
                "product_name": "Haltbare Milch",
                "description": "Marke Milram",
                "source_brand": "Milram",
                "brand_evidence": "Milram",
                "brand_evidence_source": "description",
            },
            "Milram",
        ),
        (
            {
                "product_name": "Pommes Frites",
                "source_brand": "Aviko",
                "brand_evidence": "Aviko",
                "brand_evidence_source": "image",
            },
            "Aviko",
        ),
        ({"product_name": "Schleizer Bockwürste"}, "Schleiz"),
    ],
)
def test_approved_brand_requires_product_local_evidence(row, expected):
    assert relevance.resolve_approved_brand(row)["brand"] == expected


@pytest.mark.parametrize(
    "row",
    [
        {"product_name": "Milch", "source_brand": "ARO"},
        {
            "product_name": "Milch",
            "source_brand": "ARO",
            "brand_evidence": "ARO",
            "brand_evidence_source": "unknown",
        },
        {
            "product_name": "Milch",
            "description": "ohne Markenangabe",
            "source_brand": "ARO",
            "brand_evidence": "ARO",
            "brand_evidence_source": "description",
        },
        {"product_name": "Metropolitan Milch"},
    ],
)
def test_unproven_or_substring_brand_is_rejected(row):
    assert relevance.resolve_approved_brand(row) is None


@pytest.mark.parametrize("group", sorted(relevance.PACKAGED_EXCEPTION_POLICY_GROUPS))
def test_additional_groups_using_core_fresh_are_safely_downgraded(group):
    parsed = relevance.normalize_eligibility_analysis(
        policy_payload(group, route="core_fresh"),
        identity_analysis=identity(group),
        brand_proof=approved_brand_proof(),
    )

    assert parsed["policy_decision"] == "uncertain"
    assert parsed["eligibility_route"] == "none"
    assert parsed["requirements_met"] is False
    assert parsed["review_needed"] is True
    assert parsed["reported_policy_decision"] == "include"
    assert parsed["reported_eligibility_route"] == "core_fresh"
    assert "PACKAGED_ROUTE_MISMATCH" in blocking_issue_codes(parsed)


@pytest.mark.parametrize("group", sorted(relevance.PACKAGED_EXCEPTION_POLICY_GROUPS))
def test_additional_groups_without_verified_brand_are_safely_downgraded(group):
    size = 500 if group in {"Cheese", "Sausage"} else None
    parsed = relevance.normalize_eligibility_analysis(
        policy_payload(group, package_size_grams=size),
        identity_analysis=identity(group),
        brand_proof=None,
    )

    assert parsed["policy_decision"] == "uncertain"
    assert parsed["eligibility_route"] == "none"
    assert parsed["requirements_met"] is False
    assert parsed["review_needed"] is True
    assert parsed["verified_brand_found"] is False
    assert parsed["reported_policy_decision"] == "include"
    assert "PACKAGED_BRAND_UNVERIFIED" in blocking_issue_codes(parsed)


@pytest.mark.parametrize("group", sorted(relevance.PACKAGED_EXCEPTION_POLICY_GROUPS))
def test_additional_groups_accept_verified_brand_only_on_packaged_route(group):
    size = 500 if group in {"Cheese", "Sausage"} else None
    parsed = relevance.normalize_eligibility_analysis(
        policy_payload(group, package_size_grams=size),
        identity_analysis=identity(group),
        brand_proof=approved_brand_proof(),
    )
    assert parsed["requirements_met"] is True
    assert parsed["eligibility_route"] == "packaged_exception"
    assert parsed["verified_brand"] == "ARO"


@pytest.mark.parametrize("group", ["Cheese", "Sausage"])
def test_cheese_and_sausage_still_require_at_least_500_grams(group):
    parsed = relevance.normalize_eligibility_analysis(
        policy_payload(group, package_size_grams=499),
        identity_analysis=identity(group),
        brand_proof=approved_brand_proof(),
    )

    assert parsed["policy_decision"] == "uncertain"
    assert parsed["eligibility_route"] == "none"
    assert parsed["requirements_met"] is False
    assert parsed["review_needed"] is True
    assert parsed["package_size_grams"] == 499
    assert "PACKAGE_SIZE_REQUIREMENT_UNPROVEN" in blocking_issue_codes(parsed)


@pytest.mark.parametrize("group", sorted(relevance.CORE_FRESH_POLICY_GROUPS))
def test_core_fresh_groups_do_not_require_a_brand(group):
    parsed = relevance.normalize_eligibility_analysis(
        policy_payload(
            group,
            route="core_fresh",
            required_brand_found=None,
        ),
        identity_analysis=identity(group),
        brand_proof=None,
    )
    assert parsed["requirements_met"] is True
    assert parsed["eligibility_route"] == "core_fresh"


def test_pickled_cucumbers_group_correction_is_nonfatal_for_negative_route():
    facts = identity("Obst Gemüse", exclusion_signal="explicit_exclusion")
    facts["processing_state"] = "pickled"
    policy = relevance.normalize_eligibility_analysis(
        policy_payload(
            "Other",
            decision="exclude",
            route="none",
            required_brand_found=None,
        ),
        identity_analysis=facts,
        brand_proof=None,
    )

    assert policy["policy_decision"] == "exclude"
    assert policy["eligibility_route"] == "none"
    assert policy["product_group"] == "Obst Gemüse"
    assert policy["reported_product_group"] == "Other"
    assert policy["stage_1_product_group"] == "Obst Gemüse"
    assert policy["product_group_changed"] is True
    assert "STAGE_2_PRODUCT_GROUP_CHANGED" in issue_codes(policy)
    final = relevance.normalize_final_review(
        final_payload("Nein"),
        facts,
        policy,
        brand_proof=None,
    )
    assert final["decision"] == "Nein"


def test_stage_2_group_change_cannot_create_a_positive_route():
    parsed = relevance.normalize_eligibility_analysis(
        policy_payload(
            "Fleisch",
            decision="include",
            route="core_fresh",
            required_brand_found=None,
        ),
        identity_analysis=identity("Other"),
        brand_proof=None,
    )

    assert parsed["product_group"] == "Other"
    assert parsed["stage_1_product_group"] == "Other"
    assert parsed["reported_product_group"] == "Fleisch"
    assert parsed["product_group_changed"] is True
    assert parsed["policy_decision"] == "uncertain"
    assert parsed["eligibility_route"] == "none"
    assert parsed["review_needed"] is True
    assert "STAGE_2_PRODUCT_GROUP_CHANGED" in issue_codes(parsed)
    assert "OTHER_CANNOT_BE_POSITIVE" in blocking_issue_codes(parsed)


def test_final_reviewer_cannot_bypass_brand_or_other_contract():
    milk_identity = identity("Milk")
    milk_policy = relevance.normalize_eligibility_analysis(
        policy_payload(
            "Milk",
            decision="uncertain",
            route="none",
            required_brand_found=None,
        ),
        identity_analysis=milk_identity,
        brand_proof=None,
    )
    milk_final = relevance.normalize_final_review(
        final_payload("Ja"),
        milk_identity,
        milk_policy,
        brand_proof=None,
    )
    assert milk_final["reported_decision"] == "Ja"
    assert milk_final["decision"] == "Nein"
    assert milk_final["rule_id"] == "FINAL_POLICY_GUARDRAIL"
    assert milk_final["review_needed"] is True
    assert "FINAL_PACKAGED_BRAND_BLOCKED" in blocking_issue_codes(milk_final)

    other_identity = identity("Other", exclusion_signal="explicit_exclusion")
    other_policy = relevance.normalize_eligibility_analysis(
        policy_payload(
            "Other",
            decision="exclude",
            route="none",
            required_brand_found=None,
        ),
        identity_analysis=other_identity,
        brand_proof=approved_brand_proof(),
    )
    other_final = relevance.normalize_final_review(
        final_payload("Ja", overrode_stage="both", review_needed=True),
        other_identity,
        other_policy,
        brand_proof=approved_brand_proof(),
    )
    assert other_final["reported_decision"] == "Ja"
    assert other_final["decision"] == "Nein"
    assert other_final["rule_id"] == "FINAL_POLICY_GUARDRAIL"
    assert other_final["review_needed"] is True
    assert "FINAL_OTHER_BLOCKED" in blocking_issue_codes(other_final)


def test_prompts_explicitly_protect_reported_product_group_edge_cases():
    prompt = relevance.build_identity_prompt({"product_name": "Milram Milchreis"})
    policy_prompt = relevance.build_eligibility_prompt(
        {"product_name": "Aviko TK Gemüse"},
        identity("Frozen vegetables"),
        approved_brand_proof(),
    )
    assert "Milram Milchreis or ARO Sahnejoghurt => policy_group Other" in prompt
    assert "Unmarked frozen vegetables are not core fresh" in policy_prompt
    assert "guacamole and creamy dips remain excluded" in policy_prompt
