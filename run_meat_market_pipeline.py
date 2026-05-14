#!/usr/bin/env python3
"""Run pork news + beef dashboard pipeline and synthesize final price signals with DeepSeek."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_BRAVE_SCRIPT = ROOT / "brave_answers_pork.py"
DEFAULT_BEEF_SCRIPT = ROOT / "agridata_beef_dashboard_dump.py"
DEFAULT_BEEF_JSON = ROOT / "beef_dashboard_v2.json"
DEFAULT_BEEF_SUMMARY = ROOT / "beef_dashboard_v2_summary.txt"
DEFAULT_FINAL_OUTPUT = ROOT / "meat_market_price_signals.txt"
DEFAULT_FINAL_RAW_OUTPUT = ROOT / "meat_market_price_signals_raw.json"
DEFAULT_PROMPT_SNAPSHOT = ROOT / "meat_market_price_signals_prompt.txt"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hostname_from_item(item: dict) -> str:
    meta = item.get("meta_url", {}) if isinstance(item, dict) else {}
    host = (meta.get("hostname") or "").lower()
    if host:
        return host
    url = item.get("url", "") if isinstance(item, dict) else ""
    return urlparse(url).netloc.lower()


def run_child_script(args: list[str], cwd: Path) -> None:
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(args))
    print("=" * 80)
    subprocess.run(args, cwd=cwd, check=True)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_brave_context(summary_path: Path, deduped_json_path: Path, max_items: int = 15) -> str:
    summary_text = read_text(summary_path)
    deduped_items = read_json(deduped_json_path)
    if not isinstance(deduped_items, list):
        deduped_items = []

    lines = [
        "PORK NEWS SUMMARY",
        summary_text.strip(),
        "",
        "PORK NEWS SOURCE ITEMS",
    ]

    for idx, item in enumerate(deduped_items[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        lines.append(f"{idx}. Title: {item.get('title', '')}")
        lines.append(f"   Source label: Brave / {hostname_from_item(item)}")
        lines.append(f"   Age: {item.get('age', '')}")
        lines.append(f"   URL: {item.get('url', '')}")
        lines.append(f"   Description: {item.get('description', '')}")
        extra_snippets = item.get("extra_snippets") or []
        for snippet in extra_snippets[:2]:
            lines.append(f"   Snippet: {snippet}")
        lines.append("")

    return "\n".join(lines).strip()


def build_beef_context(summary_path: Path) -> str:
    summary_text = read_text(summary_path)
    return "\n".join(
        [
            "BEEF DASHBOARD SUMMARY",
            summary_text.strip(),
        ]
    ).strip()


def build_final_prompt(brave_context: str, beef_context: str) -> str:
    return f"""
You are a senior meat market analyst for Germany and Europe.

Use ONLY the source material provided below.
Do not invent facts.
Do not use outside knowledge.

Task:
1. Identify the 3 to 5 most important current developments across pork news and beef market dashboard signals.
2. Focus only on developments that could materially affect near-term meat prices or meat market expectations.
3. Combine overlapping signals where appropriate.
4. Output only bullet lines.
5. Start each line with:
   + if the fact indicates upward price pressure
   - if the fact indicates downward price pressure
6. End every line with source references in square brackets, for example:
   [Brave / reuters.com; Agridata / PricesC table]
7. Prefer concrete source labels that are present in the provided material.
8. No intro.
9. No conclusion.
10. No numbering.
11. No neutral bullets.
12. Keep each line compact but specific.

Source material:

{brave_context}

{beef_context}
""".strip()


def extract_message_content(message: object) -> str:
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None:
            return extract_message_content(content)
    return ""


def call_deepseek(
    api_key: str,
    model: str,
    prompt: str,
    base_url: str,
    timeout_seconds: int,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You produce concise market bullets from supplied evidence.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def extract_final_text(response_json: dict) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek response does not contain choices.")

    message = choices[0].get("message", {})
    content = extract_message_content(message.get("content"))
    if not content:
        raise RuntimeError("DeepSeek response did not contain final content.")
    return content.strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pork news + beef dashboard pipeline and synthesize final signals with DeepSeek."
    )
    parser.add_argument("--brave-script", default=str(DEFAULT_BRAVE_SCRIPT), help="Path to brave_answers_pork.py")
    parser.add_argument("--beef-script", default=str(DEFAULT_BEEF_SCRIPT), help="Path to agridata_beef_dashboard_dump.py")
    parser.add_argument("--beef-output", default=str(DEFAULT_BEEF_JSON), help="Path for beef dashboard JSON output")
    parser.add_argument(
        "--beef-summary-output",
        default=str(DEFAULT_BEEF_SUMMARY),
        help="Path for beef dashboard summary text output",
    )
    parser.add_argument("--final-output", default=str(DEFAULT_FINAL_OUTPUT), help="Path for final DeepSeek bullet output")
    parser.add_argument(
        "--final-raw-output",
        default=str(DEFAULT_FINAL_RAW_OUTPUT),
        help="Path for raw DeepSeek response JSON",
    )
    parser.add_argument(
        "--prompt-snapshot",
        default=str(DEFAULT_PROMPT_SNAPSHOT),
        help="Path for saved DeepSeek prompt snapshot",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="DeepSeek API base URL",
    )
    parser.add_argument(
        "--deepseek-model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner"),
        help="DeepSeek model name",
    )
    parser.add_argument(
        "--deepseek-timeout",
        type=int,
        default=300,
        help="Timeout for DeepSeek API call in seconds",
    )
    parser.add_argument(
        "--dashboard-headed",
        action="store_true",
        help="Run the beef dashboard scraper in headed mode",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Skip Brave and Agridata runs and send the existing prompt snapshot directly to DeepSeek",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    load_env_file(ROOT / ".env")

    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY fehlt in .env oder in der Umgebung.")

    brave_script = Path(args.brave_script).expanduser().resolve()
    beef_script = Path(args.beef_script).expanduser().resolve()
    beef_output = Path(args.beef_output).expanduser().resolve()
    beef_summary_output = Path(args.beef_summary_output).expanduser().resolve()
    final_output = Path(args.final_output).expanduser().resolve()
    final_raw_output = Path(args.final_raw_output).expanduser().resolve()
    prompt_snapshot = Path(args.prompt_snapshot).expanduser().resolve()

    for path in [beef_output, beef_summary_output, final_output, final_raw_output, prompt_snapshot]:
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.prompt_only:
        prompt = read_text(prompt_snapshot)
    else:
        run_child_script([sys.executable, str(brave_script)], ROOT)

        beef_cmd = [
            sys.executable,
            str(beef_script),
            "--output",
            str(beef_output),
            "--summary-output",
            str(beef_summary_output),
        ]
        if args.dashboard_headed:
            beef_cmd.append("--headed")
        run_child_script(beef_cmd, ROOT)

        brave_context = build_brave_context(
            summary_path=ROOT / "schweinefleisch_summary.txt",
            deduped_json_path=ROOT / "brave_news_pork_deduped.json",
        )
        beef_context = build_beef_context(beef_summary_output)
        prompt = build_final_prompt(brave_context, beef_context)
        prompt_snapshot.write_text(prompt, encoding="utf-8")

    print("\n" + "=" * 80)
    print("CALLING DEEPSEEK")
    print("=" * 80)
    print("Model:", args.deepseek_model)
    print("Base URL:", args.deepseek_base_url)

    response_json = call_deepseek(
        api_key=deepseek_api_key,
        model=args.deepseek_model,
        prompt=prompt,
        base_url=args.deepseek_base_url,
        timeout_seconds=args.deepseek_timeout,
    )
    final_text = extract_final_text(response_json)

    final_output.write_text(final_text, encoding="utf-8")
    final_raw_output.write_text(json.dumps(response_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL MARKET SIGNALS")
    print("=" * 80)
    print(final_text)
    print(f"Saved final bullets to: {final_output}")
    print(f"Saved raw DeepSeek response to: {final_raw_output}")
    print(f"Saved prompt snapshot to: {prompt_snapshot}")


if __name__ == "__main__":
    main()
