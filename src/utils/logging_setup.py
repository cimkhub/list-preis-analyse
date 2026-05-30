import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from uuid import uuid4

_RUN_ID = ContextVar("birkenhof_run_id", default="-")
_STANDARD_LOG_FIELDS = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}


class RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID.get()
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", _RUN_ID.get()),
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_FIELDS or key.startswith("_"):
                continue
            if value is None:
                continue
            data[key] = value

        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data, ensure_ascii=False)


def setup_logging(
    logs_dir: str = "logs",
    level: int = logging.INFO,
    week: int | None = None,
    year: int | None = None,
) -> logging.Logger:
    logs_path = Path(logs_dir)
    runs_path = logs_path / "runs"
    runs_path.mkdir(parents=True, exist_ok=True)
    if week and year:
        week_runs_path = logs_path / f"KW{week:02d}_{year}"
        weekly_file = logs_path / f"{year}_KW{week:02d}.log"
    else:
        iso_year, iso_week, _ = datetime.now().isocalendar()
        week_runs_path = logs_path / f"KW{iso_week:02d}_{iso_year}"
        weekly_file = logs_path / f"{iso_year}_KW{iso_week:02d}.log"
    week_runs_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    _RUN_ID.set(run_id)

    logger = logging.getLogger("birkenhof")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    run_filter = RunContextFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.addFilter(run_filter)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(run_id)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    weekly_handler = logging.FileHandler(weekly_file, encoding="utf-8")
    weekly_handler.setLevel(logging.DEBUG)
    weekly_handler.addFilter(run_filter)
    weekly_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(run_id)s %(name)s - %(message)s"
    ))
    logger.addHandler(weekly_handler)

    run_log_path = runs_path / f"{run_id}.log"
    run_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    run_handler.setLevel(logging.DEBUG)
    run_handler.addFilter(run_filter)
    run_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(run_id)s %(name)s - %(message)s"
    ))
    logger.addHandler(run_handler)

    week_run_log_path = week_runs_path / f"{run_id}.log"
    week_run_handler = logging.FileHandler(week_run_log_path, encoding="utf-8")
    week_run_handler.setLevel(logging.DEBUG)
    week_run_handler.addFilter(run_filter)
    week_run_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(run_id)s %(name)s - %(message)s"
    ))
    logger.addHandler(week_run_handler)

    json_path = runs_path / f"{run_id}.jsonl"
    json_handler = logging.FileHandler(json_path, encoding="utf-8")
    json_handler.setLevel(logging.DEBUG)
    json_handler.addFilter(run_filter)
    json_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(json_handler)

    week_json_path = week_runs_path / f"{run_id}.jsonl"
    week_json_handler = logging.FileHandler(week_json_path, encoding="utf-8")
    week_json_handler.setLevel(logging.DEBUG)
    week_json_handler.addFilter(run_filter)
    week_json_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(week_json_handler)

    log_event(
        logger,
        "Logger initialized",
        event="logging_setup",
        status="ready",
        run_log_path=str(run_log_path),
        week_run_log_path=str(week_run_log_path),
        json_log_path=str(json_path),
        week_json_log_path=str(week_json_path),
        weekly_log_path=str(weekly_file),
    )

    return logger


def get_run_id() -> str:
    return _RUN_ID.get()


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    exc_info=False,
    **fields,
):
    safe_fields = {
        key: value
        for key, value in fields.items()
        if value is not None and key not in _STANDARD_LOG_FIELDS
    }
    logger.log(level, message, extra=safe_fields, exc_info=exc_info)


@contextmanager
def log_stage(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    **fields,
):
    start = time.perf_counter()
    log_event(logger, f"{message} started", level=level, status="start", **fields)
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            f"{message} failed: {exc}",
            level=logging.ERROR,
            status="error",
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
            exc_info=True,
            **fields,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            f"{message} completed",
            level=level,
            status="ok",
            duration_ms=duration_ms,
            **fields,
        )
