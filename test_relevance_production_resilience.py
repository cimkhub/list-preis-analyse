"""Production-regression tests for row-local relevance failures.

These tests deliberately return responses that are syntactically valid but
semantically disagree with the route contract, as well as responses that are
not parseable at all.  Semantic disagreements must reach the final reviewer;
exhausted transport/schema failures must fail closed for that row without
aborting the weekly batch.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

import classify_fresh_food_relevance as relevance


PRO_MODEL = "deepseek-v4-pro"


def api_response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def stage_name(prompt: str) -> str:
    return prompt.splitlines()[0]


def frozen_mushroom_facts() -> dict[str, Any]:
    return {
        "product_type": "frozen chanterelles",
        "product_family": "obst_gemuese",
        "policy_group": "Frozen vegetables",
        "temperature_state": "frozen",
        "processing_state": "raw_plain",
        "source_brand": "Metro",
        "brand_evidence": "Metro Chef Pfifferlinge",
        "brand_evidence_source": "product_name",
        "exclusion_signal": "none",
        "exclusion_reason": "",
        "confidence": 0.98,
        "evidence": ["Metro Chef Pfifferlinge", "tiefgefroren"],
    }


def meat_facts(product_name: str) -> dict[str, Any]:
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
        "evidence": [product_name, "frisch"],
    }


def milk_facts() -> dict[str, Any]:
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
        "confidence": 0.98,
        "evidence": ["ARO Milch"],
    }


def policy(
    *,
    decision: str,
    route: str,
    group: str,
    required_brand_found: bool | None,
    review_needed: bool,
    rule_id: str,
) -> dict[str, Any]:
    return {
        "policy_decision": decision,
        "eligibility_route": route,
        "product_group": group,
        "required_brand_found": required_brand_found,
        "package_size_grams": None,
        "rule_id": rule_id,
        "reason": f"model policy: {decision} via {route}",
        "confidence": 0.72,
        "review_needed": review_needed,
        "evidence": [group, route],
    }


def final_review(
    *,
    decision: str,
    review_needed: bool,
    overrode_stage: str = "none",
) -> dict[str, Any]:
    return {
        "decision": decision,
        "reason": "independent final review",
        "rule_id": "FINAL_REVIEW",
        "confidence": 0.8,
        "review_needed": review_needed,
        "overrode_stage": overrode_stage,
        "evidence": ["source row", "stage policy"],
    }


def write_input(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "supplier",
        "category",
        "product_name",
        "description",
        "source_brand",
        "brand_evidence",
        "brand_evidence_source",
        "valid_to",
        "calendar_week",
        "year",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "valid_to": "2026-08-15",
                    "calendar_week": "32",
                    "year": "2026",
                    **row,
                }
            )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_traces(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, str]],
    api: Callable[..., dict[str, Any]],
    *,
    max_retries: int = 1,
    workers: int = 2,
    expect_technical_error: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], tuple[int, int]]:
    input_path = tmp_path / "all_suppliers.csv"
    output_path = tmp_path / "all_suppliers_relevant.csv"
    trace_path = tmp_path / "all_suppliers_relevant_trace.jsonl"
    write_input(input_path, rows)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    monkeypatch.setattr(relevance, "call_deepseek", api)

    kwargs = {
        "input_path": input_path,
        "output_path": output_path,
        "trace_output_path": trace_path,
        "workers": workers,
        "deepseek_model": PRO_MODEL,
        "max_retries": max_retries,
    }
    if expect_technical_error:
        with pytest.raises(relevance.RelevanceBatchTechnicalError) as captured:
            relevance.run_relevance_classification(**kwargs)
        result_path = captured.value.output_path
        yes_count = captured.value.yes_count
        no_count = captured.value.no_count
    else:
        result_path, yes_count, no_count = relevance.run_relevance_classification(
            **kwargs
        )
    return read_rows(result_path), read_traces(trace_path), (yes_count, no_count)


@pytest.mark.parametrize("stage_2_decision", ["exclude", "uncertain"])
def test_metro_chef_pfifferlinge_negative_or_uncertain_policy_reaches_stage_3_once(
    monkeypatch: pytest.MonkeyPatch,
    stage_2_decision: str,
) -> None:
    """The exact VM regression is a reviewable disagreement, not a fatal schema error."""
    calls: list[str] = []

    def fake_call_deepseek(**kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["prompt"]
        calls.append(stage_name(prompt))
        if prompt.startswith("STAGE 1 OF 3"):
            payload = frozen_mushroom_facts()
        elif prompt.startswith("STAGE 2 OF 3"):
            payload = policy(
                decision=stage_2_decision,
                route="none",
                group="Frozen vegetables",
                required_brand_found=True,
                review_needed=True,
                rule_id=(
                    "OUT_OF_SCOPE"
                    if stage_2_decision == "exclude"
                    else "INSUFFICIENT_EVIDENCE"
                ),
            )
        elif prompt.startswith("STAGE 3 OF 3"):
            # The reviewer corrects the negative/uncertain policy.  Its audit
            # flags are intentionally wrong: they are derived locally from
            # the decisions and must not trigger another fatal parser error.
            payload = final_review(
                decision="Ja",
                review_needed=False,
                overrode_stage="none",
            )
        else:  # pragma: no cover - guards the prompt protocol.
            raise AssertionError(prompt[:100])
        return api_response(json.dumps(payload))

    monkeypatch.setattr(relevance, "call_deepseek", fake_call_deepseek)
    index, label, _reason, trace = relevance.classify_row_with_trace(
        142,
        {
            "supplier": "metro",
            "category": "tk",
            "product_name": "Metro Chef Pfifferlinge",
            "description": "Ganz, küchenfertig, tiefgefroren",
            "source_brand": "Metro",
            "brand_evidence": "Metro Chef Pfifferlinge",
            "brand_evidence_source": "product_name",
        },
        api_key="unused",
        model=PRO_MODEL,
        base_url="https://unused.invalid",
        timeout_seconds=1,
        # A semantic disagreement must not burn retries.
        max_retries=3,
    )

    assert calls == [
        "STAGE 1 OF 3: FACT EXTRACTION",
        "STAGE 2 OF 3: POLICY DECISION",
        "STAGE 3 OF 3: INDEPENDENT FINAL REVIEW",
    ]
    assert (index, label) == (142, "Ja")
    assert trace["stage_3_final"]["review_needed"] is True
    assert trace["stage_3_final"]["overrode_stage"] == "stage_2"
    assert trace["classification_status"] == "review"
    assert trace["reported_stage_2_route"] == "none"
    assert trace["effective_final_route"] == "packaged_exception"
    assert not trace["stage_errors"]
    issues = trace["stage_2_policy"]["contract_issues"]
    assert isinstance(issues, list)
    assert all(issue["kind"].startswith("semantic") for issue in issues)
    assert all(
        set(issue) >= {"code", "kind", "severity", "message", "fields"}
        for issue in issues
    )
    assert any(issue["code"] == "VERIFIED_BRAND_ROUTE_MISMATCH" for issue in issues)


def test_stage_1_policy_group_is_authoritative_while_stage_2_report_is_audited() -> None:
    facts = relevance.normalize_identity_analysis(json.dumps(frozen_mushroom_facts()))
    brand_proof = {
        "brand": "Metro",
        "matched_text": "Metro",
        "source": "product_name",
        "evidence": "Metro Chef Pfifferlinge",
        "legacy_fallback": False,
    }
    parsed = relevance.normalize_eligibility_analysis(
        json.dumps(
            policy(
                decision="exclude",
                route="none",
                group="Other",
                required_brand_found=True,
                review_needed=True,
                rule_id="OUT_OF_SCOPE",
            )
        ),
        identity_analysis=facts,
        brand_proof=brand_proof,
    )

    assert parsed["product_group"] == "Frozen vegetables"
    assert parsed["reported_product_group"] == "Other"
    assert parsed["stage_1_product_group"] == "Frozen vegetables"
    assert parsed["product_group_changed"] is True
    assert any(
        issue["kind"].startswith("semantic") and "product_group" in issue["fields"]
        for issue in parsed["contract_issues"]
    )


def test_final_stage_can_resolve_non_authoritative_exclusion_disagreement() -> None:
    facts_payload = meat_facts("Short Ribs BBQ")
    facts_payload.update(
        {
            "processing_state": "raw_seasoned",
            "exclusion_signal": "explicit_exclusion",
            "exclusion_reason": "BBQ interpreted as prepared",
        }
    )
    facts = relevance.normalize_identity_analysis(json.dumps(facts_payload))
    stage_2 = relevance.normalize_eligibility_analysis(
        json.dumps(
            policy(
                decision="include",
                route="core_fresh",
                group="Fleisch",
                required_brand_found=None,
                review_needed=False,
                rule_id="CORE_FRESH",
            )
        ),
        identity_analysis=facts,
        brand_proof=None,
    )
    result = relevance.normalize_final_review(
        json.dumps(final_review(decision="Ja", review_needed=False)),
        facts,
        stage_2,
        brand_proof=None,
    )

    assert stage_2["policy_decision"] == "uncertain"
    assert any(
        issue["code"] == "EXCLUSION_SIGNAL_CONFLICT"
        and issue["severity"] == "warning"
        for issue in stage_2["contract_issues"]
    )
    assert result["decision"] == "Ja"
    assert result["overrode_stage"] == "both"
    assert result["review_needed"] is True


def test_final_stage_can_correct_stage_1_policy_group() -> None:
    facts_payload = meat_facts("Rinder Thin Skirt")
    facts_payload.update(
        {
            "product_type": "unknown product",
            "product_family": "sonstiges",
            "policy_group": "Other",
            "temperature_state": "unknown",
            "processing_state": "unknown",
        }
    )
    facts = relevance.normalize_identity_analysis(json.dumps(facts_payload))
    stage_2 = relevance.normalize_eligibility_analysis(
        json.dumps(
            policy(
                decision="exclude",
                route="none",
                group="Other",
                required_brand_found=None,
                review_needed=False,
                rule_id="OUT_OF_SCOPE",
            )
        ),
        identity_analysis=facts,
        brand_proof=None,
    )
    final_payload = final_review(decision="Ja", review_needed=True)
    final_payload["final_policy_group"] = "Fleisch"
    result = relevance.normalize_final_review(
        json.dumps(final_payload),
        facts,
        stage_2,
        brand_proof=None,
    )

    assert result["decision"] == "Ja"
    assert result["stage_1_policy_group"] == "Other"
    assert result["final_policy_group"] == "Fleisch"
    assert result["overrode_stage"] == "both"
    assert result["review_needed"] is True


def _good_meat_api_payload(prompt: str, product_name: str) -> dict[str, Any]:
    if prompt.startswith("STAGE 1 OF 3"):
        return meat_facts(product_name)
    if prompt.startswith("STAGE 2 OF 3"):
        return policy(
            decision="include",
            route="core_fresh",
            group="Fleisch",
            required_brand_found=None,
            review_needed=False,
            rule_id="CORE_FRESH",
        )
    if prompt.startswith("STAGE 3 OF 3"):
        return final_review(decision="Ja", review_needed=False)
    raise AssertionError(prompt[:100])


def test_one_exhausted_schema_failure_does_not_abort_or_drop_other_batch_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: Counter[str] = Counter()

    def fake_call_deepseek(**kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["prompt"]
        if "BROKEN-JSON-ROW" in prompt and prompt.startswith("STAGE 1 OF 3"):
            attempts["broken"] += 1
            return api_response("this is deliberately not JSON")
        if "BROKEN-JSON-ROW" in prompt and prompt.startswith("STAGE 2 OF 3"):
            return api_response(
                json.dumps(
                    policy(
                        decision="exclude",
                        route="none",
                        group="Other",
                        required_brand_found=None,
                        review_needed=True,
                        rule_id="INSUFFICIENT_EVIDENCE",
                    )
                )
            )
        if "BROKEN-JSON-ROW" in prompt and prompt.startswith("STAGE 3 OF 3"):
            return api_response(json.dumps(final_review(decision="Nein", review_needed=True)))
        product_name = "Rinderfilet A" if "Rinderfilet A" in prompt else "Rinderfilet B"
        return api_response(json.dumps(_good_meat_api_payload(prompt, product_name)))

    rows, traces, counts = run_batch(
        tmp_path,
        monkeypatch,
        [
            {"supplier": "test", "category": "fleisch", "product_name": "Rinderfilet A"},
            {"supplier": "test", "category": "sonstiges", "product_name": "BROKEN-JSON-ROW"},
            {"supplier": "test", "category": "fleisch", "product_name": "Rinderfilet B"},
        ],
        fake_call_deepseek,
        max_retries=2,
        expect_technical_error=True,
    )

    assert attempts["broken"] == 2
    assert counts == (2, 1)
    assert [row["Relevant"] for row in rows] == ["Ja", "Nein", "Ja"]
    assert len(rows) == len(traces) == 3
    failed_row = rows[1]
    failed_trace = traces[1]
    assert failed_row["relevance_review_needed"] == "true"
    assert failed_trace["classification_status"] == "technical_error"
    assert failed_trace["stage_3_final"]["decision"] == "Nein"
    assert failed_trace["stage_3_final"]["review_needed"] is True
    # A later successful final review remains the visible decision, while the
    # earlier exhausted stage keeps the row in technical_error/review status.
    assert failed_trace["stage_3_final"]["rule_id"] == "FINAL_REVIEW"
    assert len(failed_trace["stage_errors"]) == 1
    stage_error = failed_trace["stage_errors"][0]
    assert set(stage_error) >= {"stage", "type", "message", "attempts"}
    assert stage_error["attempts"] == 2
    assert "json" in stage_error["message"].casefold()
    assert all(trace["classification_status"] == "ok" for trace in (traces[0], traces[2]))


@pytest.mark.parametrize("violation_stage", ["stage_2", "stage_3"])
def test_final_review_repairs_stage_2_route_but_never_bypasses_final_brand_guardrail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation_stage: str,
) -> None:
    calls: list[str] = []
    has_brand_proof = violation_stage == "stage_2"

    def fake_call_deepseek(**kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["prompt"]
        calls.append(stage_name(prompt))
        if prompt.startswith("STAGE 1 OF 3"):
            payload = milk_facts()
            if not has_brand_proof:
                payload.update(
                    {
                        "source_brand": None,
                        "brand_evidence": None,
                        "brand_evidence_source": "unknown",
                    }
                )
        elif prompt.startswith("STAGE 2 OF 3"):
            payload = policy(
                decision="include" if violation_stage == "stage_2" else "uncertain",
                route="core_fresh" if violation_stage == "stage_2" else "none",
                group="Milk",
                required_brand_found=True if has_brand_proof else None,
                review_needed=violation_stage != "stage_2",
                rule_id=(
                    "ADDITIONAL_PRODUCT"
                    if violation_stage == "stage_2"
                    else "INSUFFICIENT_EVIDENCE"
                ),
            )
        elif prompt.startswith("STAGE 3 OF 3"):
            # Deliberately tries to bypass packaged_exception as well.
            payload = final_review(
                decision="Ja",
                review_needed=violation_stage == "stage_3",
                overrode_stage="none",
            )
        else:  # pragma: no cover
            raise AssertionError(prompt[:100])
        return api_response(json.dumps(payload))

    row = {
        "supplier": "metro",
        "category": "mopro",
        "product_name": "ARO Milch" if has_brand_proof else "Haltbare Milch",
    }
    if has_brand_proof:
        row.update(
            {
                "source_brand": "ARO",
                "brand_evidence": "ARO Milch",
                "brand_evidence_source": "product_name",
            }
        )

    rows, traces, counts = run_batch(
        tmp_path,
        monkeypatch,
        [row],
        fake_call_deepseek,
        max_retries=3,
        workers=1,
    )

    expected_label = "Ja" if violation_stage == "stage_2" else "Nein"
    assert counts == ((1, 0) if violation_stage == "stage_2" else (0, 1))
    assert rows[0]["Relevant"] == expected_label
    assert rows[0]["relevance_review_needed"] == "true"
    assert traces[0]["classification_status"] == "review"
    assert not traces[0]["stage_errors"]
    issues = traces[0]["contract_issues"]
    assert any(issue["severity"] == "blocking" for issue in issues)
    assert all(issue["kind"].startswith("semantic") for issue in issues)
    if violation_stage == "stage_2":
        assert traces[0]["reported_stage_2_route"] == "core_fresh"
        assert traces[0]["effective_final_route"] == "packaged_exception"
        assert traces[0]["stage_3_final"]["overrode_stage"] == "stage_2"
    else:
        assert traces[0]["stage_3_final"]["rule_id"] == "FINAL_POLICY_GUARDRAIL"
    # Semantic contract violations are reviewed once, not retried as API failures.
    assert calls == [
        "STAGE 1 OF 3: FACT EXTRACTION",
        "STAGE 2 OF 3: POLICY DECISION",
        "STAGE 3 OF 3: INDEPENDENT FINAL REVIEW",
    ]


def test_transport_exception_saves_all_rows_then_stops_downstream_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_deepseek(**kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["prompt"]
        if "TIMEOUT-ROW" in prompt and prompt.startswith("STAGE 1 OF 3"):
            raise TimeoutError("simulated upstream timeout")
        if "TIMEOUT-ROW" in prompt and prompt.startswith("STAGE 2 OF 3"):
            return api_response(
                json.dumps(
                    policy(
                        decision="exclude",
                        route="none",
                        group="Other",
                        required_brand_found=None,
                        review_needed=True,
                        rule_id="INSUFFICIENT_EVIDENCE",
                    )
                )
            )
        if "TIMEOUT-ROW" in prompt and prompt.startswith("STAGE 3 OF 3"):
            return api_response(json.dumps(final_review(decision="Nein", review_needed=True)))
        return api_response(json.dumps(_good_meat_api_payload(prompt, "Rinderfilet")))

    # The row is fully materialized, but a technical batch status prevents the
    # incomplete classification from flowing into the customer workbook.
    rows, traces, counts = run_batch(
        tmp_path,
        monkeypatch,
        [
            {"supplier": "test", "category": "fleisch", "product_name": "TIMEOUT-ROW"},
            {"supplier": "test", "category": "fleisch", "product_name": "Rinderfilet"},
        ],
        fake_call_deepseek,
        max_retries=1,
        expect_technical_error=True,
    )

    assert counts == (1, 1)
    assert [row["Relevant"] for row in rows] == ["Nein", "Ja"]
    assert rows[0]["relevance_review_needed"] == "true"
    assert traces[0]["classification_status"] == "technical_error"
    assert traces[0]["stage_3_final"]["rule_id"] == "CLASSIFICATION_ERROR"
    assert len(traces[0]["stage_errors"]) == 1
    stage_error = traces[0]["stage_errors"][0]
    assert stage_error["type"]
    assert stage_error["attempts"] == 1
    assert "simulated upstream timeout" in stage_error["message"]
    assert traces[1]["classification_status"] == "ok"


def test_exhausted_final_stage_emits_classification_error_instead_of_escaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_call_deepseek(**kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        prompt = kwargs["prompt"]
        if prompt.startswith("STAGE 3 OF 3"):
            attempts += 1
            return api_response("malformed final response")
        return api_response(json.dumps(_good_meat_api_payload(prompt, "Rinderfilet")))

    rows, traces, counts = run_batch(
        tmp_path,
        monkeypatch,
        [{"supplier": "test", "category": "fleisch", "product_name": "Rinderfilet"}],
        fake_call_deepseek,
        max_retries=2,
        workers=1,
        expect_technical_error=True,
    )

    assert attempts == 2
    assert counts == (0, 1)
    assert rows[0]["Relevant"] == "Nein"
    assert rows[0]["relevance_review_needed"] == "true"
    assert traces[0]["classification_status"] == "technical_error"
    assert traces[0]["stage_3_final"]["decision"] == "Nein"
    assert traces[0]["stage_3_final"]["rule_id"] == "CLASSIFICATION_ERROR"
    assert traces[0]["stage_3_final"]["review_needed"] is True
    assert len(traces[0]["stage_errors"]) == 1
    assert traces[0]["stage_errors"][0]["stage"] == "stage_3"
    assert traces[0]["stage_errors"][0]["attempts"] == 2


def test_checkpoint_resumes_good_rows_and_retries_only_technical_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_calls: list[str] = []

    def first_api(**kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["prompt"]
        first_calls.append(prompt)
        product_name = "BROKEN-FINAL" if "BROKEN-FINAL" in prompt else "Rinderfilet"
        if product_name == "BROKEN-FINAL" and prompt.startswith("STAGE 3 OF 3"):
            return api_response("not valid JSON")
        return api_response(json.dumps(_good_meat_api_payload(prompt, product_name)))

    rows_input = [
        {"supplier": "test", "category": "fleisch", "product_name": "Rinderfilet"},
        {"supplier": "test", "category": "fleisch", "product_name": "BROKEN-FINAL"},
    ]
    rows, traces, counts = run_batch(
        tmp_path,
        monkeypatch,
        rows_input,
        first_api,
        max_retries=1,
        workers=2,
        expect_technical_error=True,
    )
    checkpoint_path = tmp_path / "all_suppliers_relevant_trace_checkpoint.jsonl"
    assert counts == (1, 1)
    assert traces[1]["classification_status"] == "technical_error"
    assert checkpoint_path.exists()

    second_calls: list[str] = []

    def repaired_api(**kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["prompt"]
        second_calls.append(prompt)
        assert "BROKEN-FINAL" in prompt
        return api_response(json.dumps(_good_meat_api_payload(prompt, "BROKEN-FINAL")))

    rows, traces, counts = run_batch(
        tmp_path,
        monkeypatch,
        rows_input,
        repaired_api,
        max_retries=1,
        workers=2,
    )
    assert counts == (2, 0)
    assert [row["Relevant"] for row in rows] == ["Ja", "Ja"]
    assert len(second_calls) == 1
    assert second_calls[0].startswith("STAGE 3 OF 3")
    assert all(trace["classification_status"] == "ok" for trace in traces)
    assert traces[1]["resumed_stages"] == ["stage_1", "stage_2"]
    assert not checkpoint_path.exists()


def test_unexpected_worker_exception_is_materialized_at_batch_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "all_suppliers.csv"
    output_path = tmp_path / "all_suppliers_relevant.csv"
    trace_path = tmp_path / "all_suppliers_relevant_trace.jsonl"
    write_input(
        input_path,
        [{"supplier": "test", "category": "fleisch", "product_name": "Rinderfilet"}],
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    monkeypatch.setattr(
        relevance,
        "classify_row_with_trace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected worker bug")
        ),
    )

    with pytest.raises(relevance.RelevanceBatchTechnicalError) as captured:
        relevance.run_relevance_classification(
            input_path=input_path,
            output_path=output_path,
            trace_output_path=trace_path,
            workers=1,
            deepseek_model=PRO_MODEL,
            max_retries=1,
        )
    result_path = captured.value.output_path
    yes_count = captured.value.yes_count
    no_count = captured.value.no_count

    rows = read_rows(result_path)
    traces = read_traces(trace_path)
    assert (yes_count, no_count) == (0, 1)
    assert rows[0]["Relevant"] == "Nein"
    assert rows[0]["relevance_processing_status"] == "technical_error"
    assert traces[0]["classification_status"] == "technical_error"
    assert traces[0]["stage_3_final"]["rule_id"] == "CLASSIFICATION_ERROR"
    assert "unexpected worker bug" in traces[0]["row_error"]["message"]


def test_missing_trailing_csv_cells_do_not_escape_row_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "all_suppliers.csv"
    output_path = tmp_path / "all_suppliers_relevant.csv"
    trace_path = tmp_path / "all_suppliers_relevant_trace.jsonl"
    input_path.write_text(
        "supplier,category,product_name,description\nonly-supplier\n",
        encoding="utf-8-sig",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    monkeypatch.setattr(
        relevance,
        "call_deepseek",
        lambda **kwargs: api_response(
            json.dumps(_good_meat_api_payload(kwargs["prompt"], "unknown product"))
        ),
    )

    result_path, yes_count, no_count = relevance.run_relevance_classification(
        input_path=input_path,
        output_path=output_path,
        trace_output_path=trace_path,
        workers=1,
        deepseek_model=PRO_MODEL,
        max_retries=1,
    )

    assert (yes_count, no_count) == (1, 0)
    assert read_rows(result_path)[0]["Relevant"] == "Ja"
    assert read_traces(trace_path)[0]["classification_status"] == "ok"


def test_shared_circuit_breaker_stops_request_storm_during_api_outage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def unavailable_api(**kwargs: Any) -> dict[str, Any]:
        prompts.append(kwargs["prompt"])
        raise TimeoutError("simulated global DeepSeek outage")

    input_rows = [
        {
            "supplier": "test",
            "category": "fleisch",
            "product_name": f"Rinderfilet {index}",
        }
        for index in range(20)
    ]
    rows, traces, counts = run_batch(
        tmp_path,
        monkeypatch,
        input_rows,
        unavailable_api,
        max_retries=3,
        workers=2,
        expect_technical_error=True,
    )

    assert counts == (0, 20)
    assert len(rows) == len(traces) == 20
    assert all(trace["classification_status"] == "technical_error" for trace in traces)
    assert prompts
    assert len(prompts) <= 6
    assert all(prompt.startswith("STAGE 1 OF 3") for prompt in prompts)


def test_concurrent_run_for_same_output_is_rejected_before_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "all_suppliers.csv"
    output_path = tmp_path / "all_suppliers_relevant.csv"
    trace_path = tmp_path / "all_suppliers_relevant_trace.jsonl"
    write_input(
        input_path,
        [{"supplier": "test", "category": "fleisch", "product_name": "Rinderfilet"}],
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    monkeypatch.setattr(
        relevance,
        "call_deepseek",
        lambda **_kwargs: pytest.fail("model must not be called while output is locked"),
    )
    lock = relevance.acquire_relevance_run_lock(output_path)
    try:
        with pytest.raises(RuntimeError, match="Another relevance run"):
            relevance.run_relevance_classification(
                input_path=input_path,
                output_path=output_path,
                trace_output_path=trace_path,
                workers=1,
                deepseek_model=PRO_MODEL,
                max_retries=1,
            )
    finally:
        relevance.release_relevance_run_lock(lock)


def test_run_lock_is_released_when_internal_pipeline_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    monkeypatch.setattr(
        relevance,
        "_run_relevance_classification_impl",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("internal boom")),
    )

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="internal boom"):
            relevance.run_relevance_classification(
                input_path=input_path,
                output_path=output_path,
            )


def test_checkpoint_from_different_implementation_is_never_reused(
    tmp_path: Path,
) -> None:
    row = {"product_name": "Rinderfilet", "category": "fleisch"}
    checkpoint_path = tmp_path / "trace_checkpoint.jsonl"
    stale_trace = {
        "schema_version": relevance.TRACE_SCHEMA_VERSION,
        "model": PRO_MODEL,
        "implementation_digest": "stale-deployment",
        "product_evidence": relevance.build_product_evidence(row),
        "classification_status": "ok",
    }
    record = {
        "checkpoint_schema_version": relevance.CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": relevance._row_fingerprint(0, row, PRO_MODEL),
        "index": 0,
        "label": "Ja",
        "reason": "stale",
        "trace": stale_trace,
    }
    checkpoint_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert relevance.load_relevance_checkpoint(
        checkpoint_path,
        [row],
        PRO_MODEL,
    ) == {}
