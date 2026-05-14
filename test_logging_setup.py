import json

from src.utils.logging_setup import get_run_id, log_event, log_stage, setup_logging


def test_setup_logging_writes_structured_json_events(tmp_path):
    logger = setup_logging(str(tmp_path / "logs"))
    run_id = get_run_id()

    log_event(
        logger,
        "Structured test event",
        event="test_event",
        status="ok",
        supplier="metro",
        document_count=3,
    )

    json_files = sorted((tmp_path / "logs" / "runs").glob("*.jsonl"))
    assert len(json_files) == 1

    lines = json_files[0].read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    assert any(record["message"] == "Structured test event" for record in records)
    event_record = next(record for record in records if record["message"] == "Structured test event")
    assert event_record["run_id"] == run_id
    assert event_record["event"] == "test_event"
    assert event_record["status"] == "ok"
    assert event_record["supplier"] == "metro"
    assert event_record["document_count"] == 3


def test_log_stage_records_duration(tmp_path):
    logger = setup_logging(str(tmp_path / "logs"))

    with log_stage(logger, "Stage timing test", event="stage_test", supplier="edeka"):
        pass

    json_file = next((tmp_path / "logs" / "runs").glob("*.jsonl"))
    records = [
        json.loads(line)
        for line in json_file.read_text(encoding="utf-8").strip().splitlines()
    ]

    completed = next(record for record in records if record["message"] == "Stage timing test completed")
    assert completed["event"] == "stage_test"
    assert completed["status"] == "ok"
    assert completed["supplier"] == "edeka"
    assert "duration_ms" in completed
