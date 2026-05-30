#!/usr/bin/env python3
"""Create a review-dashboard data bundle for deployment volumes."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Export parsed/data/images files needed by the customer review dashboard.")
    parser.add_argument("--week", type=int)
    parser.add_argument("--year", type=int)
    parser.add_argument("--all-review-data", action="store_true", help="Export all historic review data currently present.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.all_review_data:
        output = Path(args.output) if args.output else ROOT / "dashboard_review_data.zip"
        candidates = all_review_candidates()
    else:
        if not args.week or not args.year:
            parser.error("--week and --year are required unless --all-review-data is used")
        output = Path(args.output) if args.output else ROOT / f"dashboard_review_bundle_KW{args.week:02d}_{args.year}.zip"
        candidates = week_candidates(args.week, args.year)

    output.parent.mkdir(parents=True, exist_ok=True)
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = candidate.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique_candidates:
            add_path(archive, path)

    print(f"Wrote {output}")
    print("Extract this ZIP into the Railway volume mount, for example /app/runtime.")
    print("For R2 startup sync, upload this ZIP and set DASHBOARD_DATA_ZIP_URL to its URL.")


def all_review_candidates() -> list[Path]:
    candidates = []
    weeks = []
    for parsed_dir in sorted((ROOT / "parsed").glob("KW*_*/")):
        match = parsed_dir.name.removeprefix("KW").split("_", 1)
        if len(match) != 2:
            continue
        try:
            week, year = int(match[0]), int(match[1])
        except ValueError:
            continue
        weeks.append((week, year))
        candidates.append(parsed_dir)
    for week, year in weeks:
        candidates.extend(data_week_dirs(week, year))
    candidates.extend((ROOT / "data").glob("qa_marks_KW*.json"))
    candidates.append(ROOT / "images" / "_relevance")
    return candidates


def week_candidates(week: int, year: int) -> list[Path]:
    candidates = [ROOT / "parsed" / f"KW{week:02d}_{year}"]
    candidates.extend(data_week_dirs(week, year))
    for supplier_dir in data_week_dirs(week, year):
        candidates.extend(relevance_preview_dirs(supplier_dir))
    candidates.extend((ROOT / "data").glob(f"qa_marks_KW{week:02d}_{year}.json"))
    return candidates


def data_week_dirs(week: int, year: int) -> list[Path]:
    dirs = []
    for supplier_dir in (ROOT / "data").glob(f"*/{year}/{week:02d}"):
        dirs.append(supplier_dir)
    for supplier_dir in (ROOT / "data").glob(f"*/{year}/{week}"):
        if supplier_dir not in dirs:
            dirs.append(supplier_dir)
    return dirs


def add_path(archive: zipfile.ZipFile, path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        archive.write(path, path.relative_to(ROOT))
        return
    for file_path in path.rglob("*"):
        if file_path.is_file():
            archive.write(file_path, file_path.relative_to(ROOT))


def relevance_preview_dirs(supplier_dir: Path) -> list[Path]:
    decision_path = supplier_dir / "relevance_decisions.json"
    if not decision_path.exists():
        return []
    try:
        decisions = json.loads(decision_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    supplier = supplier_dir.parts[-3]
    dirs = []
    for decision in decisions if isinstance(decisions, list) else []:
        file_path = Path(str(decision.get("file_path") or decision.get("filename") or ""))
        if file_path.stem:
            dirs.append(ROOT / "images" / "_relevance" / supplier / file_path.stem)
    return dirs


if __name__ == "__main__":
    main()
