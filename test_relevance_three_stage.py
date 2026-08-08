import csv
import json

import pytest

import classify_fresh_food_relevance as relevance
from main import resolve_deepseek_model


def api_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def positive_stage_responses(prompt: str) -> dict:
    if prompt.startswith("STAGE 1 OF 3"):
        return api_response(
            json.dumps(
                {
                    "product_type": "plain milk",
                    "hard_exclusion": False,
                    "exclusion_reason": "",
                    "evidence": "Milch",
                }
            )
        )
    if prompt.startswith("STAGE 2 OF 3"):
        return api_response(
            json.dumps(
                {
                    "eligibility_route": "packaged_exception",
                    "product_group": "Milk",
                    "required_brand_found": True,
                    "package_size_grams": None,
                    "requirements_met": True,
                    "failure_reason": "",
                    "evidence": "required brand and milk",
                }
            )
        )
    if prompt.startswith("STAGE 3 OF 3"):
        return api_response("Ja|Milk")
    raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


def workflow_stage_responses(prompt: str) -> dict:
    is_meat = "Rinderfilet" in prompt
    if prompt.startswith("STAGE 1 OF 3"):
        return api_response(
            json.dumps(
                {
                    "product_type": "raw beef" if is_meat else "plain milk",
                    "hard_exclusion": False,
                    "exclusion_reason": "",
                    "evidence": "Rinderfilet" if is_meat else "Milch",
                }
            )
        )
    if prompt.startswith("STAGE 2 OF 3"):
        return api_response(
            json.dumps(
                {
                    "eligibility_route": "core_fresh" if is_meat else "packaged_exception",
                    "product_group": "Fleisch" if is_meat else "Milk",
                    "required_brand_found": None if is_meat else True,
                    "package_size_grams": None,
                    "requirements_met": True,
                    "failure_reason": "",
                    "evidence": "raw beef" if is_meat else "required brand and milk",
                }
            )
        )
    if prompt.startswith("STAGE 3 OF 3"):
        return api_response("Ja|Fleisch" if is_meat else "Ja|Milk")
    raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


def test_three_prompts_have_separate_responsibilities():
    row = {"category": "mopro", "brand": "ARO", "product_name": "Milch"}
    identity = {
        "product_type": "plain milk",
        "hard_exclusion": False,
        "exclusion_reason": "",
        "evidence": "Milch",
    }
    eligibility = {
        "eligibility_route": "packaged_exception",
        "product_group": "Milk",
        "required_brand_found": True,
        "package_size_grams": None,
        "requirements_met": True,
        "failure_reason": "",
        "evidence": "ARO Milch",
    }

    identity_prompt = relevance.build_identity_prompt(row)
    eligibility_prompt = relevance.build_eligibility_prompt(row, identity)
    final_prompt = relevance.build_final_decision_prompt(row, identity, eligibility)

    assert "HARD EXCLUSIONS" in identity_prompt
    assert "Required brands:" not in identity_prompt
    assert "POSITIVE ELIGIBILITY" in eligibility_prompt
    assert "Required brands: ARO" in eligibility_prompt
    assert "FINAL RELEVANCE DECISION" in final_prompt
    assert "Apply this priority exactly" in final_prompt


def test_json_stage_parser_accepts_fenced_json():
    parsed = relevance.normalize_identity_analysis(
        '```json\n{"product_type":"fish","hard_exclusion":false,'
        '"exclusion_reason":"","evidence":"Lachsfilet"}\n```'
    )

    assert parsed["product_type"] == "fish"
    assert parsed["hard_exclusion"] is False


def test_v4_model_disables_thinking_for_small_classifier_stages():
    assert relevance.thinking_config_for_model("deepseek-v4-flash") == {"type": "disabled"}
    assert relevance.thinking_config_for_model("classic-chat") is None
    assert relevance.max_tokens_for_model("deepseek-v4-flash") == relevance.DEFAULT_MAX_TOKENS
    assert relevance.max_tokens_for_model("deepseek-v4-pro") == relevance.DEFAULT_MAX_TOKENS
    assert relevance.max_tokens_for_model("classic-chat") == relevance.DEFAULT_MAX_TOKENS


def test_pipeline_is_locked_to_deepseek_v4_pro():
    assert relevance.DEFAULT_DEEPSEEK_MODEL == "deepseek-v4-pro"
    assert resolve_deepseek_model("pro") == ("pro", "deepseek-v4-pro")
    assert resolve_deepseek_model("deepseek-v4-pro") == ("pro", "deepseek-v4-pro")
    with pytest.raises(ValueError, match="locked to the DeepSeek Pro"):
        resolve_deepseek_model("flash")


def test_classify_row_runs_all_three_model_stages_in_order(monkeypatch):
    prompts: list[str] = []

    def fake_call_deepseek(**kwargs):
        prompts.append(kwargs["prompt"])
        return positive_stage_responses(kwargs["prompt"])

    monkeypatch.setattr(relevance, "call_deepseek", fake_call_deepseek)

    index, label, reason, trace = relevance.classify_row_with_trace(
        7,
        {"category": "mopro", "brand": "", "product_name": "Unklares Produkt"},
        api_key="unused",
        model="deepseek-v4-pro",
        base_url="https://unused.invalid",
        timeout_seconds=1,
        max_retries=1,
    )

    assert [prompt.splitlines()[0] for prompt in prompts] == [
        "STAGE 1 OF 3: PRODUCT IDENTITY AND HARD EXCLUSIONS",
        "STAGE 2 OF 3: POSITIVE ELIGIBILITY",
        "STAGE 3 OF 3: FINAL RELEVANCE DECISION",
    ]
    assert (index, label, reason) == (7, "Ja", "Milk")
    assert trace["decision_source"] == "three_stage_model"
    assert trace["stage_1_identity"]["product_type"] == "plain milk"
    assert trace["stage_2_eligibility"]["eligibility_route"] == "packaged_exception"


def test_final_stage_cannot_contradict_prior_stages():
    identity = {
        "product_type": "non-food",
        "hard_exclusion": True,
        "exclusion_reason": "Non Food",
        "evidence": "Toilettenpapier",
    }
    eligibility = {
        "eligibility_route": "none",
        "product_group": "",
        "required_brand_found": None,
        "package_size_grams": None,
        "requirements_met": False,
        "failure_reason": "Not Relevant",
        "evidence": "",
    }

    try:
        relevance.validate_final_decision("Ja|Fresh Product", identity, eligibility)
    except RuntimeError as exc:
        assert "contradicted prior stages" in str(exc)
    else:
        raise AssertionError("Contradictory final decision was accepted")


def test_weekly_workflow_writes_compatible_csv_and_trace(tmp_path, monkeypatch):
    input_path = tmp_path / "all_suppliers.csv"
    output_path = tmp_path / "all_suppliers_relevant.csv"
    fieldnames = [
        "supplier",
        "category",
        "brand",
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
                "brand": "",
                "product_name": "Unklares Produkt",
                "valid_to": "2026-05-29",
                "calendar_week": "22",
                "year": "2026",
            }
        )

    calls: list[str] = []

    def fake_call_deepseek(**kwargs):
        calls.append(kwargs["prompt"])
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
    assert rows[0]["Relevant"] == "Ja"
    assert rows[0]["Reason"] == "Fleisch"
    assert rows[0]["Relevant Time"] == "Ja"
    assert rows[1]["Relevant"] == "Ja"
    assert traces[0]["decision_source"] == "three_stage_model"
    assert traces[1]["decision_source"] == "three_stage_model"
    assert all(trace["stage_1_identity"] is not None for trace in traces)
    assert all(trace["stage_2_eligibility"] is not None for trace in traces)
