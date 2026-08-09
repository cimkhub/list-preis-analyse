import csv
import json

import pytest

import classify_fresh_food_relevance as relevance
from main import resolve_deepseek_model


def api_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def stage_1_facts(*, meat: bool) -> dict:
    if meat:
        return {
            "product_type": "raw beef cut",
            "product_family": "fleisch",
            "policy_group": "Fleisch",
            "temperature_state": "fresh",
            "processing_state": "raw_cut",
            "source_brand": None,
            "brand_evidence": None,
            "brand_evidence_source": "unknown",
            "exclusion_signal": "none",
            "exclusion_reason": "",
            "confidence": 0.97,
            "evidence": ["Rinderfilet", "frisch"],
        }
    return {
        "product_type": "plain milk",
        "product_family": "mopro",
        "policy_group": "Milk",
        "temperature_state": "chilled",
        "processing_state": "unknown",
        "source_brand": "ARO",
        "brand_evidence": "ARO Milch",
        "brand_evidence_source": "product_name",
        "exclusion_signal": "none",
        "exclusion_reason": "",
        "confidence": 0.96,
        "evidence": ["ARO Milch"],
    }


def stage_2_policy(*, meat: bool) -> dict:
    return {
        "policy_decision": "include",
        "eligibility_route": "core_fresh" if meat else "packaged_exception",
        "product_group": "Fleisch" if meat else "Milk",
        "required_brand_found": None if meat else True,
        "package_size_grams": None,
        "rule_id": "CORE_FRESH" if meat else "ADDITIONAL_PRODUCT",
        "reason": "Fleisch" if meat else "Milk",
        "confidence": 0.96,
        "review_needed": False,
        "evidence": ["raw beef cut" if meat else "ARO Milch"],
    }


def stage_3_review(*, meat: bool) -> dict:
    return {
        "decision": "Ja",
        "reason": "Fleisch" if meat else "Milk",
        "rule_id": "CORE_FRESH" if meat else "ADDITIONAL_PRODUCT",
        "confidence": 0.95,
        "review_needed": False,
        "overrode_stage": "none",
        "evidence": ["Rinderfilet frisch" if meat else "ARO Milch"],
    }


def workflow_stage_responses(prompt: str) -> dict:
    is_meat = "Rinderfilet" in prompt
    if prompt.startswith("STAGE 1 OF 3"):
        return api_response(json.dumps(stage_1_facts(meat=is_meat)))
    if prompt.startswith("STAGE 2 OF 3"):
        return api_response(json.dumps(stage_2_policy(meat=is_meat)))
    if prompt.startswith("STAGE 3 OF 3"):
        return api_response(json.dumps(stage_3_review(meat=is_meat)))
    raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


def test_three_prompts_have_separate_responsibilities():
    row = {
        "category": "mopro",
        "source_brand": "ARO",
        "brand_evidence": "ARO Milch",
        "brand_evidence_source": "product_name",
        "product_name": "ARO Milch",
    }
    facts = relevance.normalize_identity_analysis(json.dumps(stage_1_facts(meat=False)))
    policy = relevance.normalize_eligibility_analysis(json.dumps(stage_2_policy(meat=False)))

    facts_prompt = relevance.build_identity_prompt(row)
    policy_prompt = relevance.build_eligibility_prompt(row, facts)
    final_prompt = relevance.build_final_decision_prompt(row, facts, policy)

    assert "FACT EXTRACTION" in facts_prompt
    assert "Never decide include/exclude" in facts_prompt
    assert "Required brands:" not in facts_prompt
    assert "POLICY DECISION" in policy_prompt
    assert "include, exclude, or uncertain" in policy_prompt
    assert "Required brands: ARO" in policy_prompt
    assert "INDEPENDENT FINAL REVIEW" in final_prompt
    assert "may override stage 1, stage 2, or both" in final_prompt
    assert '"decision":"Ja|Nein"' in final_prompt


def test_prompts_protect_recall_contrast_cases():
    facts_prompt = relevance.build_identity_prompt(
        {"product_name": "Short Ribs BBQ", "category": "fleisch"}
    )
    policy_prompt = relevance.build_eligibility_prompt(
        {"product_name": "Cevapcici", "category": "fleisch"},
        relevance.normalize_identity_analysis(json.dumps(stage_1_facts(meat=True))),
    )

    assert "Raw cut, chopped/geschnitten, minced/gehackt, formed/geformt" in facts_prompt
    assert "The word BBQ alone does not prove" in facts_prompt
    assert "Ready to eat' on ripe/mature avocado" in facts_prompt
    assert "rohe Cevapcici => fleisch, raw_formed, none" in facts_prompt
    assert "Unknown facts never justify exclude" in policy_prompt
    assert "raw_minced, raw_formed, raw_skewered, and raw_seasoned all qualify" in policy_prompt
    assert "processing_state=unknown does not block an established core family" in policy_prompt


def test_json_stage_parsers_accept_fenced_json_and_preserve_unknown():
    facts = stage_1_facts(meat=True)
    facts["temperature_state"] = "unknown"
    parsed = relevance.normalize_identity_analysis(f"```json\n{json.dumps(facts)}\n```")

    assert parsed["product_family"] == "fleisch"
    assert parsed["temperature_state"] == "unknown"
    assert parsed["hard_exclusion"] is False
    assert parsed["evidence"] == ["Rinderfilet", "frisch"]

    uncertain = relevance.normalize_eligibility_analysis(
        json.dumps(
            {
                "policy_decision": "uncertain",
                "eligibility_route": "none",
                "product_group": "Other",
                "required_brand_found": None,
                "package_size_grams": None,
                "rule_id": "INSUFFICIENT_EVIDENCE",
                "reason": "Produktidentität unklar",
                "confidence": 0.45,
                "review_needed": True,
                "evidence": ["unvollständige Bezeichnung"],
            }
        )
    )
    assert uncertain["policy_decision"] == "uncertain"
    assert uncertain["requirements_met"] is False
    assert uncertain["review_needed"] is True


def test_unknown_brand_cannot_be_turned_into_brand_missing_exclusion():
    with pytest.raises(RuntimeError, match="Unknown brand evidence"):
        relevance.normalize_eligibility_analysis(
            json.dumps(
                {
                    "policy_decision": "exclude",
                    "eligibility_route": "packaged_exception",
                    "product_group": "Milk",
                    "required_brand_found": None,
                    "package_size_grams": None,
                    "rule_id": "BRAND_MISSING",
                    "reason": "Brand missing",
                    "confidence": 0.6,
                    "review_needed": True,
                    "evidence": ["keine lesbare Marke"],
                }
            )
        )


def test_v4_model_disables_thinking_for_small_classifier_stages():
    assert relevance.thinking_config_for_model("deepseek-v4-flash") == {"type": "disabled"}
    assert relevance.thinking_config_for_model("classic-chat") is None
    assert relevance.max_tokens_for_model("deepseek-v4-flash") == relevance.DEFAULT_MAX_TOKENS
    assert relevance.max_tokens_for_model("deepseek-v4-pro") == relevance.DEFAULT_MAX_TOKENS
    assert relevance.max_tokens_for_model("classic-chat") == relevance.DEFAULT_MAX_TOKENS


def test_pipeline_and_direct_stage_calls_are_locked_to_deepseek_v4_pro(monkeypatch):
    assert relevance.DEFAULT_DEEPSEEK_MODEL == "deepseek-v4-pro"
    assert resolve_deepseek_model("pro") == ("pro", "deepseek-v4-pro")
    assert resolve_deepseek_model("deepseek-v4-pro") == ("pro", "deepseek-v4-pro")
    with pytest.raises(ValueError, match="locked to the DeepSeek Pro"):
        resolve_deepseek_model("flash")

    monkeypatch.setattr(
        relevance,
        "call_deepseek",
        lambda **_kwargs: pytest.fail("A non-Pro model must fail before any model call"),
    )
    with pytest.raises(RuntimeError, match="must use deepseek-v4-pro"):
        relevance.classify_row_with_trace(
            0,
            {"product_name": "Rinderfilet"},
            api_key="unused",
            model="deepseek-v4-flash",
            base_url="https://unused.invalid",
            timeout_seconds=1,
            max_retries=1,
        )


def test_classify_row_runs_all_three_pro_model_stages_in_order(monkeypatch):
    calls: list[dict] = []

    def fake_call_deepseek(**kwargs):
        calls.append(kwargs)
        return workflow_stage_responses(kwargs["prompt"])

    monkeypatch.setattr(relevance, "call_deepseek", fake_call_deepseek)

    index, label, reason, trace = relevance.classify_row_with_trace(
        7,
        {
            "category": "mopro",
            "source_brand": "ARO",
            "brand_evidence": "ARO Milch",
            "brand_evidence_source": "product_name",
            "product_name": "ARO Milch",
        },
        api_key="unused",
        model="deepseek-v4-pro",
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=1,
    )

    assert [call["prompt"].splitlines()[0] for call in calls] == [
        "STAGE 1 OF 3: FACT EXTRACTION",
        "STAGE 2 OF 3: POLICY DECISION",
        "STAGE 3 OF 3: INDEPENDENT FINAL REVIEW",
    ]
    assert all(call["model"] == "deepseek-v4-pro" for call in calls)
    assert (index, label, reason) == (7, "Ja", "Milk")
    assert trace["schema_version"] == 3
    assert trace["model"] == "deepseek-v4-pro"
    assert trace["decision_source"] == "three_stage_model_v3"
    assert trace["stage_1_facts"]["product_family"] == "mopro"
    assert trace["stage_2_policy"]["policy_decision"] == "include"
    assert trace["stage_1_identity"] == trace["stage_1_facts"]
    assert trace["stage_2_eligibility"] == trace["stage_2_policy"]
    assert trace["stage_3_final"]["rule_id"] == "ADDITIONAL_PRODUCT"


def test_pickled_cucumber_stage_group_correction_completes_all_three_stages(
    monkeypatch,
):
    calls: list[str] = []

    def fake_call_deepseek(**kwargs):
        prompt = kwargs["prompt"]
        calls.append(prompt.splitlines()[0])
        if prompt.startswith("STAGE 1 OF 3"):
            payload = {
                "product_type": "pickled sandwich cucumbers",
                "product_family": "obst_gemuese",
                "policy_group": "Obst Gemüse",
                "temperature_state": "ambient",
                "processing_state": "pickled",
                "source_brand": "Hengstenberg",
                "brand_evidence": "Hengstenberg Sandwich-Gurken",
                "brand_evidence_source": "product_name",
                "exclusion_signal": "explicit_exclusion",
                "exclusion_reason": "pickled preserved product",
                "confidence": 0.98,
                "evidence": ["Sandwich-Gurken"],
            }
        elif prompt.startswith("STAGE 2 OF 3"):
            payload = {
                "policy_decision": "exclude",
                "eligibility_route": "none",
                "product_group": "Other",
                "required_brand_found": None,
                "package_size_grams": None,
                "rule_id": "EXPLICIT_EXCLUSION",
                "reason": "Pickled cucumbers are preserved",
                "confidence": 0.99,
                "review_needed": False,
                "evidence": ["processing_state pickled"],
            }
        elif prompt.startswith("STAGE 3 OF 3"):
            payload = {
                "decision": "Nein",
                "reason": "Pickled preserved cucumber product",
                "rule_id": "EXPLICIT_EXCLUSION",
                "confidence": 0.99,
                "review_needed": False,
                "overrode_stage": "none",
                "evidence": ["Sandwich-Gurken", "pickled"],
            }
        else:  # pragma: no cover - protects the stage contract in this regression.
            raise AssertionError(prompt[:80])
        return api_response(json.dumps(payload))

    monkeypatch.setattr(relevance, "call_deepseek", fake_call_deepseek)
    index, label, reason, trace = relevance.classify_row_with_trace(
        158,
        {
            "supplier": "metro",
            "category": "obst_gemuese",
            "product_name": "Hengstenberg Sandwich-Gurken",
            "description": "eingelegt",
        },
        api_key="unused",
        model="deepseek-v4-pro",
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=1,
    )

    assert calls == [
        "STAGE 1 OF 3: FACT EXTRACTION",
        "STAGE 2 OF 3: POLICY DECISION",
        "STAGE 3 OF 3: INDEPENDENT FINAL REVIEW",
    ]
    assert (index, label, reason) == (
        158,
        "Nein",
        "Pickled preserved cucumber product",
    )
    assert trace["stage_2_policy"]["stage_1_product_group"] == "Obst Gemüse"
    assert trace["stage_2_policy"]["product_group"] == "Other"
    assert trace["stage_2_policy"]["product_group_changed"] is True


def test_final_reviewer_can_override_both_prior_stages_when_declared():
    facts = relevance.normalize_identity_analysis(
        json.dumps(
            {
                "product_type": "raw BBQ ribs",
                "product_family": "fleisch",
                "policy_group": "Fleisch",
                "temperature_state": "fresh",
                "processing_state": "raw_seasoned",
                "source_brand": None,
                "brand_evidence": None,
                "brand_evidence_source": "unknown",
                "exclusion_signal": "explicit_exclusion",
                "exclusion_reason": "Prepared",
                "confidence": 0.55,
                "evidence": ["Short Ribs BBQ"],
            }
        )
    )
    policy = relevance.normalize_eligibility_analysis(
        json.dumps(
            {
                "policy_decision": "exclude",
                "eligibility_route": "none",
                "product_group": "Fleisch",
                "required_brand_found": None,
                "package_size_grams": None,
                "rule_id": "EXPLICIT_EXCLUSION",
                "reason": "Prepared",
                "confidence": 0.55,
                "review_needed": True,
                "evidence": ["stage 1 exclusion"],
            }
        )
    )
    response = json.dumps(
        {
            "decision": "Ja",
            "reason": "Fleisch",
            "rule_id": "CORE_FRESH_REVIEW",
            "confidence": 0.9,
            "review_needed": True,
            "overrode_stage": "both",
            "evidence": ["BBQ alone does not prove cooked or sauced"],
        }
    )

    review = relevance.normalize_final_review(response, facts, policy)
    assert review["decision"] == "Ja"
    assert review["overrode_stage"] == "both"
    assert relevance.validate_final_decision(response, facts, policy) == ("Ja", "Fleisch")

    invalid = json.loads(response)
    invalid["overrode_stage"] = "none"
    with pytest.raises(RuntimeError, match="decision requires 'both'"):
        relevance.normalize_final_review(json.dumps(invalid), facts, policy)


def test_final_stage_requires_structured_json():
    facts = relevance.normalize_identity_analysis(json.dumps(stage_1_facts(meat=True)))
    policy = relevance.normalize_eligibility_analysis(json.dumps(stage_2_policy(meat=True)))
    with pytest.raises(RuntimeError, match="Expected a JSON object"):
        relevance.normalize_final_review("Ja|Fleisch", facts, policy)


def test_weekly_workflow_writes_compatible_csv_v2_audit_and_trace(tmp_path, monkeypatch):
    input_path = tmp_path / "all_suppliers.csv"
    output_path = tmp_path / "all_suppliers_relevant.csv"
    fieldnames = [
        "supplier",
        "category",
        "product_family",
        "temperature_state",
        "processing_state",
        "source_brand",
        "brand_evidence",
        "brand_evidence_source",
        "product_name",
        "description",
        "valid_to",
        "calendar_week",
        "year",
    ]
    with input_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "supplier": "test",
                "category": "fleisch",
                "product_family": "fleisch",
                "temperature_state": "fresh",
                "processing_state": "raw_cut",
                "product_name": "Rinderfilet",
                "valid_to": "2026-05-29",
                "calendar_week": "22",
                "year": "2026",
            }
        )
        writer.writerow(
            {
                "supplier": "test",
                "category": "mopro",
                "product_family": "mopro",
                "temperature_state": "chilled",
                "processing_state": "unknown",
                "source_brand": "ARO",
                "brand_evidence": "ARO Milch",
                "brand_evidence_source": "product_name",
                "product_name": "ARO Milch",
                "valid_to": "2026-05-29",
                "calendar_week": "22",
                "year": "2026",
            }
        )

    calls: list[dict] = []

    def fake_call_deepseek(**kwargs):
        calls.append(kwargs)
        return workflow_stage_responses(kwargs["prompt"])

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    monkeypatch.setattr(relevance, "call_deepseek", fake_call_deepseek)

    result_path, yes_count, no_count = relevance.run_relevance_classification(
        input_path=input_path,
        output_path=output_path,
        workers=1,
        deepseek_model="deepseek-v4-pro",
        max_retries=1,
    )

    with result_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    trace_path = tmp_path / "all_suppliers_relevant_trace.jsonl"
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert (yes_count, no_count) == (2, 0)
    assert len(calls) == 6
    assert all(call["model"] == "deepseek-v4-pro" for call in calls)
    assert rows[0]["Relevant"] == "Ja"
    assert rows[0]["Reason"] == "Fleisch"
    assert rows[0]["Relevant Time"] == "Ja"
    assert rows[0]["relevance_decision"] == "Ja"
    assert rows[0]["relevance_reason"] == "Fleisch"
    assert rows[0]["relevance_rule_id"] == "CORE_FRESH"
    assert rows[0]["relevance_confidence"] == "0.95"
    assert rows[0]["relevance_review_needed"] == "false"
    assert rows[0]["relevance_overrode_stage"] == "none"
    assert json.loads(rows[0]["relevance_evidence"]) == ["Rinderfilet frisch"]
    assert rows[0]["relevance_policy_group"] == "Fleisch"
    assert rows[0]["relevance_policy_decision"] == "include"
    assert rows[0]["relevance_route"] == "core_fresh"
    assert rows[0]["relevance_brand"] == ""
    assert rows[1]["relevance_policy_group"] == "Milk"
    assert rows[1]["relevance_route"] == "packaged_exception"
    assert rows[1]["relevance_brand"] == "ARO"
    assert rows[1]["relevance_brand_source"] == "product_name"
    assert rows[0]["relevance_trace_schema_version"] == "3"
    assert traces[0]["schema_version"] == 3
    assert traces[0]["decision_source"] == "three_stage_model_v3"
    assert traces[1]["decision_source"] == "three_stage_model_v3"
    assert traces[1]["verified_brand_proof"]["brand"] == "ARO"
    assert all(trace["stage_1_facts"] is not None for trace in traces)
    assert all(trace["stage_2_policy"] is not None for trace in traces)


def test_save_rows_keeps_old_call_signature_and_leaves_audit_blank(tmp_path):
    output_path = tmp_path / "compatible.csv"
    relevance.save_rows(
        output_path,
        ["product_name"],
        [{"product_name": "Rinderfilet"}],
        [("Ja", "Fleisch")],
    )

    with output_path.open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["Relevant"] == "Ja"
    assert row["relevance_rule_id"] == ""
    assert row["relevance_trace_schema_version"] == ""
