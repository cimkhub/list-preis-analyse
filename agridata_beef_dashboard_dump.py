"""
Improved structured downloader for the EU Beef & Veal dashboard.

What is improved vs v1:
- Accepts the cookie banner when present
- Extracts app name, tabs, app id and all sheet ids from the dashboard page
- Visits every sheet directly via /single/?appid=...&sheet=...
- Captures JSON responses AND WebSocket frames (Qlik often moves data over WS)
- Tries to keep only data-bearing payloads in a normalized section
- Writes one structured JSON file with dashboard meta + per-sheet data

Usage:
    pip install playwright
    playwright install chromium
    python agridata_beef_dashboard_dump_v2.py --output beef_dashboard_v2.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Request,
    Response,
    WebSocket,
    async_playwright,
)

DEFAULT_URL = "https://agridata.ec.europa.eu/extensions/DashboardBeef/Dashboard.html"
APP_MAPPING_URL = "https://agridata.ec.europa.eu/files/app-id-mapping.json"
INSTRUMENT_WEBSOCKET_SCRIPT = r"""
(() => {
  const existing = window.__qlikCapturedWebSocketFrames;
  if (Array.isArray(existing)) {
    return;
  }

  const frames = [];
  const OriginalWebSocket = window.WebSocket;
  if (!OriginalWebSocket) {
    window.__qlikCapturedWebSocketFrames = frames;
    return;
  }

  function serializePayload(payload) {
    if (typeof payload === "string") {
      return payload;
    }
    if (payload instanceof ArrayBuffer) {
      return `[arraybuffer:${payload.byteLength}]`;
    }
    if (typeof ArrayBuffer !== "undefined" && ArrayBuffer.isView && ArrayBuffer.isView(payload)) {
      return `[typedarray:${payload.byteLength ?? payload.length ?? 0}]`;
    }
    if (typeof Blob !== "undefined" && payload instanceof Blob) {
      return `[blob:${payload.size}]`;
    }
    try {
      return JSON.stringify(payload);
    } catch (error) {
      try {
        return String(payload);
      } catch (innerError) {
        return "[unserializable]";
      }
    }
  }

  function record(direction, websocketUrl, payload) {
    try {
      frames.push({
        direction,
        websocket_url: websocketUrl || "",
        payload: serializePayload(payload),
        captured_at: new Date().toISOString(),
      });
    } catch (error) {
      frames.push({
        direction,
        websocket_url: websocketUrl || "",
        payload: "[capture-error]",
        captured_at: new Date().toISOString(),
      });
    }
  }

  function WrappedWebSocket(url, protocols) {
    const ws = protocols === undefined ? new OriginalWebSocket(url) : new OriginalWebSocket(url, protocols);
    const originalSend = ws.send.bind(ws);
    ws.send = function(data) {
      record("sent", url, data);
      return originalSend(data);
    };
    ws.addEventListener("message", (event) => {
      record("received", url, event.data);
    });
    return ws;
  }

  WrappedWebSocket.prototype = OriginalWebSocket.prototype;
  for (const name of ["CONNECTING", "OPEN", "CLOSING", "CLOSED"]) {
    try {
      WrappedWebSocket[name] = OriginalWebSocket[name];
    } catch (error) {
      // Ignore readonly/static assignment issues.
    }
  }

  window.__qlikCapturedWebSocketFrames = frames;
  window.WebSocket = WrappedWebSocket;
})();
"""


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n...[truncated {len(text) - max_len} chars]"


def stable_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    pairs: list[tuple[str, str]] = []
    for key in sorted(query.keys()):
        for value in sorted(query[key]):
            pairs.append((key, value))
    query_string = urlencode(pairs)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return f"{base}?{query_string}" if query_string else base


def safe_json_loads(text: str) -> Any | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


def parse_dashboard_script(html: str) -> dict[str, Any]:
    app_name = None
    tabs: list[str] = []
    sheets: list[dict[str, Any]] = []

    app_match = re.search(r"getAppIDJson\('([^']+)'\s*,\s*'([^']+)'\)", html)
    if app_match:
        app_name = app_match.group(1)

    tabs_match = re.search(r"tabs\s*=\s*\[(.*?)\];", html, re.DOTALL)
    if tabs_match:
        tabs = re.findall(r"'([^']+)'", tabs_match.group(1))

    for match in re.finditer(
        r"strSheet(\d+)\s*=\s*'([0-9a-f\-]{36})';\s*//([^\n\r]+)",
        html,
        re.IGNORECASE,
    ):
        sheets.append(
            {
                "index": int(match.group(1)),
                "sheet_id": match.group(2),
                "label": match.group(3).strip(),
            }
        )

    return {
        "app_name": app_name,
        "tabs": tabs,
        "sheets": sheets,
    }


def pick_app_id_from_mapping(app_mapping: dict[str, Any], app_name: str | None) -> str | None:
    if not app_name:
        return None
    for section in app_mapping.values():
        if isinstance(section, dict) and app_name in section:
            value = section[app_name]
            if isinstance(value, str):
                return value
    return None


def signal_score(payload: Any) -> int:
    blob = json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
    score = 0
    for token in [
        "qHyperCube",
        "qDataPages",
        "qMatrix",
        "qPivotDataPages",
        "qLayout",
        "qMeasureInfo",
        "qDimensionInfo",
        "qGrandTotalRow",
        "qText",
        "qNum",
        "GetLayout",
        "GetHyperCubeData",
        "GetHyperCubePivotData",
        "GetObject",
    ]:
        if token in blob:
            score += 1
    return score


def classify_url(url: str) -> str:
    lower = url.lower()
    if "app-id-mapping.json" in lower:
        return "app_mapping"
    if "/single/" in lower:
        return "single_page"
    if "/api/v1/features" in lower:
        return "qlik_features"
    if "/translations/" in lower:
        return "translation"
    if "theme.json" in lower:
        return "theme"
    if "product-info.json" in lower or "import-map.json" in lower:
        return "qlik_bootstrap"
    return "other"


def response_is_interesting(url: str, content_type: str, parsed_json: Any | None) -> bool:
    lower = url.lower()
    if parsed_json is not None:
        if signal_score(parsed_json) > 0:
            return True
        if any(x in lower for x in ["app-id-mapping.json", "/single/", "/api/v1/"]):
            return True
        return False
    return "/single/" in lower and "html" in content_type.lower()


async def safe_text(response: Response) -> str:
    try:
        return await response.text()
    except Exception:
        return ""


async def accept_cookies(page: Page) -> str | None:
    candidates = [
        "text=Accept all cookies",
        "text=Accept only essential cookies",
        "button:has-text('Accept all cookies')",
        "button:has-text('Accept only essential cookies')",
        "a:has-text('Accept all cookies')",
        "a:has-text('Accept only essential cookies')",
    ]
    for selector in candidates:
        try:
            locator = page.locator(selector).first
            if await locator.is_visible(timeout=1200):
                label = (await locator.inner_text(timeout=1200)).strip()
                await locator.click(timeout=2500, force=True)
                await page.wait_for_timeout(1200)
                return label
        except Exception:
            continue
    return None


async def extract_dom_summary(page: Page) -> dict[str, Any]:
    js = r"""
() => {
  function text(el) {
    return (el?.innerText || el?.textContent || "").replace(/\s+/g, " ").trim();
  }

  const texts = Array.from(document.querySelectorAll("body *"))
    .map(el => text(el))
    .filter(Boolean)
    .filter(t => t.length >= 3)
    .slice(0, 250);

  const iframes = Array.from(document.querySelectorAll("iframe")).map((el, i) => ({
    index: i,
    src: el.getAttribute("src"),
    title: el.getAttribute("title"),
    name: el.getAttribute("name")
  }));

  return {
    title: document.title,
    location: window.location.href,
    visible_text_sample: texts,
    iframe_count: iframes.length,
    iframes
  };
}
"""
    try:
        return await page.evaluate(js)
    except Exception as exc:
        return {"error": repr(exc)}


async def extract_tables(page: Page) -> list[dict[str, Any]]:
    js = r"""
() => {
  function text(el) {
    return (el?.innerText || el?.textContent || "").replace(/\s+/g, " ").trim();
  }
  return Array.from(document.querySelectorAll("table")).map((table, index) => ({
    table_index: index,
    headers: Array.from(table.querySelectorAll("thead th")).map(text).filter(Boolean),
    rows: Array.from(table.querySelectorAll("tbody tr")).map(tr =>
      Array.from(tr.querySelectorAll("th, td")).map(text)
    ).filter(row => row.some(Boolean))
  }));
}
"""
    try:
        return await page.evaluate(js)
    except Exception:
        return []


async def capture_render_hints(page: Page) -> dict[str, Any]:
    js = r"""
() => {
  const html = document.documentElement.outerHTML;
  const hints = {};
  const tokens = [
    'qHyperCube',
    'qLayout',
    'qMatrix',
    'qDataPages',
    'qPivotDataPages',
    'GetHyperCubeData',
    'GetObject'
  ];
  for (const token of tokens) {
    hints[token] = html.includes(token);
  }
  return hints;
}
"""
    try:
        return await page.evaluate(js)
    except Exception as exc:
        return {"error": repr(exc)}


async def extract_qlik_object_summary(page: Page) -> list[dict[str, Any]]:
    js = r"""
() => {
  function text(el) {
    return (el?.innerText || el?.textContent || "").replace(/\s+/g, " ").trim();
  }

  const nodes = Array.from(document.querySelectorAll(
    "[data-qid], [tid], .qv-object, .qv-object-wrapper, article#content > *"
  ));

  return nodes
    .map((node, index) => {
      const qid = node.getAttribute("data-qid") || node.getAttribute("tid") || null;
      const titleNode = node.querySelector(
        ".qv-object-title, .qv-chart-title, [title], header, h1, h2, h3"
      );
      const nodeText = text(node);
      if (!qid && !nodeText) {
        return null;
      }
      return {
        index,
        qid,
        id: node.id || null,
        classes: typeof node.className === "string" ? node.className : null,
        title: text(titleNode) || null,
        text_sample: nodeText.slice(0, 500),
      };
    })
    .filter(Boolean)
    .slice(0, 200);
}
"""
    try:
        return await page.evaluate(js)
    except Exception:
        return []


async def extract_instrumented_ws_frames(page: Page) -> list[dict[str, Any]]:
    js = r"() => window.__qlikCapturedWebSocketFrames || []"
    try:
        frames = await page.evaluate(js)
        return frames if isinstance(frames, list) else []
    except Exception:
        return []


def qlik_cell_value(cell: Any) -> Any:
    if not isinstance(cell, dict):
        return cell
    return {
        "text": cell.get("qText"),
        "num": cell.get("qNum"),
        "state": cell.get("qState"),
        "elem_number": cell.get("qElemNumber"),
    }


def qlik_header_title(info: dict[str, Any], fallback: str) -> str:
    for key in ("qFallbackTitle", "qGroupFallbackTitles", "qLabel"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = " / ".join(str(item).strip() for item in value if str(item).strip())
            if joined:
                return joined
    return fallback


def build_hypercube_table(cube: dict[str, Any], source: dict[str, Any], path: str) -> dict[str, Any] | None:
    dim_info = cube.get("qDimensionInfo") or []
    meas_info = cube.get("qMeasureInfo") or []
    data_pages = cube.get("qDataPages") or []
    pivot_pages = cube.get("qPivotDataPages") or []

    headers = [
        qlik_header_title(item, f"dimension_{index + 1}")
        for index, item in enumerate(dim_info)
        if isinstance(item, dict)
    ]
    headers.extend(
        qlik_header_title(item, f"measure_{index + 1}")
        for index, item in enumerate(meas_info)
        if isinstance(item, dict)
    )

    rows: list[list[Any]] = []
    for page in data_pages:
        if not isinstance(page, dict):
            continue
        matrix = page.get("qMatrix") or []
        for row in matrix:
            if isinstance(row, list):
                rows.append([qlik_cell_value(cell) for cell in row])

    if not rows and not pivot_pages:
        return None

    return {
        "source": source,
        "path": path,
        "headers": headers,
        "dimension_count": len(dim_info),
        "measure_count": len(meas_info),
        "row_count": len(rows),
        "rows": rows,
        "pivot_pages": pivot_pages if pivot_pages else None,
    }


def extract_qlik_tables_from_payload(payload: Any, source: dict[str, Any]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            direct_cube = build_hypercube_table(node, source, path)
            if direct_cube is not None:
                tables.append(direct_cube)

            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(payload, "root")
    return tables


class SheetCapture:
    def __init__(self, sheet_id: str, max_text_chars: int, max_frames: int) -> None:
        self.sheet_id = sheet_id
        self.max_text_chars = max_text_chars
        self.max_frames = max_frames
        self.responses: list[dict[str, Any]] = []
        self.websockets: list[dict[str, Any]] = []
        self._seen = set()

    async def handle_response(self, response: Response) -> None:
        try:
            request = response.request
            resource_type = request.resource_type
            url = response.url
            status = response.status
            headers = await response.all_headers()
            content_type = headers.get("content-type", "")
            body_text = await safe_text(response)
            parsed_json = None
            if "json" in content_type.lower():
                parsed_json = safe_json_loads(body_text)
            else:
                parsed_json = safe_json_loads(body_text)

            if not response_is_interesting(url, content_type, parsed_json):
                return

            body_sha = sha256_text(body_text)
            key = (stable_url(url), status, body_sha)
            if key in self._seen:
                return
            self._seen.add(key)

            self.responses.append(
                {
                    "sheet_id": self.sheet_id,
                    "url": url,
                    "url_normalized": stable_url(url),
                    "category": classify_url(url),
                    "status": status,
                    "method": request.method,
                    "resource_type": resource_type,
                    "content_type": content_type,
                    "signal_score": signal_score(parsed_json) if parsed_json is not None else 0,
                    "response": {
                        "json": parsed_json,
                        "text": None if parsed_json is not None else truncate(body_text, self.max_text_chars),
                        "body_sha256": body_sha,
                        "body_length": len(body_text),
                    },
                    "captured_at": now_utc_iso(),
                }
            )
        except Exception:
            return

    def _append_ws_frame(self, direction: str, payload: str | bytes, ws_url: str) -> None:
        if isinstance(payload, bytes):
            try:
                payload_text = payload.decode("utf-8", errors="replace")
            except Exception:
                payload_text = repr(payload)
        else:
            payload_text = payload

        parsed = safe_json_loads(payload_text)
        interesting = parsed is not None and signal_score(parsed) > 0
        if not interesting:
            # Keep a small number of non-JSON frames in case the endpoint changes.
            if len(self.websockets) >= self.max_frames:
                return
            if len(payload_text) > 5000:
                return

        if len(self.websockets) >= self.max_frames:
            return

        entry = {
            "sheet_id": self.sheet_id,
            "websocket_url": ws_url,
            "direction": direction,
            "signal_score": signal_score(parsed) if parsed is not None else 0,
            "payload": parsed if parsed is not None else truncate(payload_text, self.max_text_chars),
            "captured_at": now_utc_iso(),
        }
        self.websockets.append(entry)

    def attach_to_page(self, page: Page) -> None:
        page.on("response", lambda response: asyncio.create_task(self.handle_response(response)))

        def on_ws(ws: WebSocket) -> None:
            ws_url = getattr(ws, "url", "")

            try:
                ws.on("framesent", lambda payload: self._append_ws_frame("sent", payload, ws_url))
            except Exception:
                pass
            try:
                ws.on("framereceived", lambda payload: self._append_ws_frame("received", payload, ws_url))
            except Exception:
                pass

        try:
            page.on("websocket", on_ws)
        except Exception:
            pass

    def absorb_instrumented_ws_frames(self, frames: list[dict[str, Any]]) -> None:
        for frame in frames[: self.max_frames]:
            if not isinstance(frame, dict):
                continue
            self._append_ws_frame(
                frame.get("direction", "received"),
                frame.get("payload", ""),
                frame.get("websocket_url", ""),
            )


async def fetch_app_mapping(context: BrowserContext) -> dict[str, Any]:
    response = await context.request.get(APP_MAPPING_URL, timeout=60000)
    if not response.ok:
        raise RuntimeError(f"Failed to fetch app mapping: HTTP {response.status}")
    try:
        return await response.json()
    except Exception:
        return {}


async def goto_and_settle(page: Page, url: str, timeout_ms: int, settle_ms: int) -> None:
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    await page.wait_for_timeout(settle_ms)
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass
    await page.wait_for_timeout(settle_ms)


async def inspect_sheet(
    context: BrowserContext,
    base_host: str,
    app_id: str,
    sheet: dict[str, Any],
    timeout_ms: int,
    settle_ms: int,
    max_text_chars: int,
    max_ws_frames: int,
    headed: bool,
) -> dict[str, Any]:
    del headed  # kept for CLI symmetry

    page = await context.new_page()
    capture = SheetCapture(sheet_id=sheet["sheet_id"], max_text_chars=max_text_chars, max_frames=max_ws_frames)
    capture.attach_to_page(page)

    url = f"{base_host}/single/?appid={app_id}&sheet={sheet['sheet_id']}&opt=nointeraction"
    started_at = now_utc_iso()

    accepted_cookie = None
    try:
        await goto_and_settle(page, url, timeout_ms=timeout_ms, settle_ms=settle_ms)
        accepted_cookie = await accept_cookies(page)
        await page.wait_for_timeout(settle_ms)

        # Gentle interaction to trigger late lazy-loading.
        try:
            await page.locator("body").click(timeout=2000, position={"x": 20, "y": 20})
            await page.wait_for_timeout(1200)
        except Exception:
            pass

        try:
            await page.keyboard.press("Tab")
            await page.wait_for_timeout(500)
        except Exception:
            pass

        for action in (
            lambda: page.mouse.wheel(0, 1200),
            lambda: page.keyboard.press("End"),
            lambda: page.keyboard.press("Home"),
        ):
            try:
                await action()
                await page.wait_for_timeout(1200)
            except Exception:
                continue

        dom = await extract_dom_summary(page)
        tables = await extract_tables(page)
        render_hints = await capture_render_hints(page)
        qlik_objects = await extract_qlik_object_summary(page)
        capture.absorb_instrumented_ws_frames(await extract_instrumented_ws_frames(page))
        html = await page.content()

        result = {
            "sheet_id": sheet["sheet_id"],
            "sheet_index": sheet["index"],
            "sheet_label": sheet["label"],
            "url": url,
            "started_at": started_at,
            "finished_at": now_utc_iso(),
            "accepted_cookie_banner": accepted_cookie,
            "page": {
                "dom": dom,
                "tables": tables,
                "qlik_objects": qlik_objects,
                "render_hints": render_hints,
                "html_sha256": sha256_text(html),
                "html_length": len(html),
            },
            "network": {
                "responses": capture.responses,
                "websocket_frames": capture.websockets,
            },
            "normalized": normalize_sheet_payloads(capture.responses, capture.websockets),
        }
    finally:
        await page.close()

    return result


def normalize_sheet_payloads(
    responses: list[dict[str, Any]],
    websocket_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    json_responses = []
    for item in responses:
        payload = item.get("response", {}).get("json")
        if payload is None:
            continue
        if item.get("signal_score", 0) <= 0 and item.get("category") not in {"app_mapping", "qlik_features"}:
            continue
        json_responses.append(
            {
                "url": item["url"],
                "category": item["category"],
                "signal_score": item["signal_score"],
                "payload": payload,
            }
        )

    json_ws_frames = []
    for frame in websocket_frames:
        payload = frame.get("payload")
        if isinstance(payload, (dict, list)) and frame.get("signal_score", 0) > 0:
            json_ws_frames.append(
                {
                    "direction": frame["direction"],
                    "signal_score": frame["signal_score"],
                    "payload": payload,
                }
            )

    signal_counts = {
        "response_payloads": len(json_responses),
        "websocket_payloads": len(json_ws_frames),
    }

    extracted_tables = []
    for item in json_responses:
        extracted_tables.extend(
            extract_qlik_tables_from_payload(
                item["payload"],
                {
                    "kind": "response",
                    "url": item["url"],
                    "category": item["category"],
                },
            )
        )

    for frame in json_ws_frames:
        extracted_tables.extend(
            extract_qlik_tables_from_payload(
                frame["payload"],
                {
                    "kind": "websocket",
                    "direction": frame["direction"],
                },
            )
        )

    return {
        "signal_counts": signal_counts,
        "data_responses": json_responses,
        "data_websocket_frames": json_ws_frames,
        "extracted_tables": extracted_tables,
        "extracted_table_count": len(extracted_tables),
    }


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_space(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        value = cell.get("text")
        return normalize_space("" if value is None else str(value))
    return normalize_space("" if cell is None else str(cell))


def cell_num(cell: Any) -> float | None:
    value = cell.get("num") if isinstance(cell, dict) else cell
    if value in (None, "", "NaN"):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def parse_percent(text: str) -> float | None:
    cleaned = text.replace("%", "").replace(",", ".").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def format_number(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f}"


def format_signed_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%"


def format_compact_number(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.{decimals}f}k"
    return f"{value:.{decimals}f}"


def collect_sheet_text(sheet: dict[str, Any]) -> str:
    parts: list[str] = []
    dom = sheet.get("page", {}).get("dom", {})
    for item in dom.get("visible_text_sample", []) or []:
        if isinstance(item, str):
            parts.append(item)

    for item in sheet.get("page", {}).get("qlik_objects", []) or []:
        if isinstance(item, dict):
            title = item.get("title")
            text_sample = item.get("text_sample")
            if isinstance(title, str):
                parts.append(title)
            if isinstance(text_sample, str):
                parts.append(text_sample)

    return " ".join(unique_strings(parts))


def find_sheet_by_label(result: dict[str, Any], label: str) -> dict[str, Any] | None:
    for sheet in result.get("sheets", []):
        if normalize_space(sheet.get("sheet_label", "")).lower() == label.lower():
            return sheet
    return None


def parse_metric_rows_from_text(
    text: str,
    categories: list[str],
) -> tuple[str | None, list[dict[str, Any]]]:
    category_pattern = "|".join(re.escape(item) for item in sorted(categories, key=len, reverse=True))
    week_match = re.search(r"Price(?: for)? week (\d{4}-\d{1,2})", text)
    pattern = re.compile(
        rf"({category_pattern})\s+([0-9]+(?:\.[0-9]+)?)\s+([+-]?[0-9]+(?:\.[0-9]+)?%)\s+([+-]?[0-9]+(?:\.[0-9]+)?%)\s+([+-]?[0-9]+(?:\.[0-9]+)?%)"
    )

    rows_by_category: dict[str, dict[str, Any]] = {}
    for match in pattern.finditer(text):
        category = match.group(1)
        rows_by_category[category] = {
            "category": category,
            "price": float(match.group(2)),
            "wow": parse_percent(match.group(3)),
            "mom": parse_percent(match.group(4)),
            "yoy": parse_percent(match.group(5)),
        }

    ordered_rows = [rows_by_category[category] for category in categories if category in rows_by_category]
    return week_match.group(1) if week_match else None, ordered_rows


def parse_production_changes_from_text(text: str) -> list[dict[str, Any]]:
    categories = [
        "Young cattle",
        "Bovine meat",
        "Bullock",
        "Heifer",
        "Bull",
        "Cow",
        "Calf",
    ]
    category_pattern = "|".join(re.escape(item) for item in sorted(categories, key=len, reverse=True))
    pattern = re.compile(
        rf"({category_pattern})\s+([+-]?[0-9]+(?:\.[0-9]+)?%)\s+([+-]?[0-9]+(?:\.[0-9]+)?%)"
    )

    rows_by_category: dict[str, dict[str, Any]] = {}
    for match in pattern.finditer(text):
        category = match.group(1)
        rows_by_category[category] = {
            "category": category,
            "eu": parse_percent(match.group(2)),
            "eu_uk": parse_percent(match.group(3)),
        }

    return [rows_by_category[category] for category in categories if category in rows_by_category]


def collect_monthly_production_points(result: dict[str, Any]) -> list[dict[str, Any]]:
    month_numbers = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    for sheet in result.get("sheets", []):
        for table in sheet.get("normalized", {}).get("extracted_tables", []) or []:
            headers = table.get("headers") or []
            if len(headers) < 3 or "1.000 Tonnes" not in headers[-1]:
                continue
            for row in table.get("rows", []) or []:
                if len(row) < 3:
                    continue
                month = cell_text(row[0])
                year_text = cell_text(row[1])
                value = cell_num(row[2])
                if month not in month_numbers or not year_text.isdigit() or value is None:
                    continue
                key = (int(year_text), month_numbers[month])
                rows_by_key[key] = {
                    "year": int(year_text),
                    "month": month,
                    "month_num": month_numbers[month],
                    "value": value,
                }

    return [rows_by_key[key] for key in sorted(rows_by_key)]


def collect_trade_rankings(result: dict[str, Any], export: bool) -> tuple[int | None, list[dict[str, Any]]]:
    year_prefix = "WT_E_YEAR" if export else "WT_I_YEAR"
    rows_by_country: dict[tuple[int, str], float] = {}

    for sheet in result.get("sheets", []):
        for table in sheet.get("normalized", {}).get("extracted_tables", []) or []:
            headers = table.get("headers") or []
            if len(headers) < 3 or not str(headers[0]).startswith(year_prefix) or "Qty in tonnes" not in headers[-1]:
                continue
            for row in table.get("rows", []) or []:
                if len(row) < 3:
                    continue
                year_num = cell_num(row[0])
                country = cell_text(row[1])
                quantity = cell_num(row[2])
                if year_num is None or not country or quantity is None:
                    continue
                key = (int(year_num), country)
                rows_by_country[key] = max(quantity, rows_by_country.get(key, quantity))

    if not rows_by_country:
        return None, []

    latest_year = max(year for year, _ in rows_by_country)
    latest_rows = [
        {"country": country, "quantity": quantity}
        for (year, country), quantity in rows_by_country.items()
        if year == latest_year
    ]
    latest_rows.sort(key=lambda item: item["quantity"], reverse=True)
    return latest_year, latest_rows


def build_dashboard_summary_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = result.get("meta", {})
    dashboard = result.get("dashboard", {})
    summary = result.get("summary", {})

    carcasse_sheet = find_sheet_by_label(result, "PricesC table") or find_sheet_by_label(result, "Summary Prices Carcasse")
    live_sheet = find_sheet_by_label(result, "PricesL table") or find_sheet_by_label(result, "Summary Prices Live animals")
    production_sheet = find_sheet_by_label(result, "Prod table") or find_sheet_by_label(result, "Summary Production")

    carcasse_text = collect_sheet_text(carcasse_sheet) if carcasse_sheet else ""
    live_text = collect_sheet_text(live_sheet) if live_sheet else ""
    production_text = collect_sheet_text(production_sheet) if production_sheet else ""

    carcasse_week, carcasse_rows = parse_metric_rows_from_text(
        carcasse_text,
        ["A URO", "ACZ URO", "C URO", "Z URO"],
    )
    live_week, live_rows = parse_metric_rows_from_text(
        live_text,
        [
            "Male Calves Dairy Type (€/head)",
            "Male Calves Beef Type (€/head)",
            "Young Store Cattle (€/Kg)",
            "Yearling Male Store Cattle (€/Kg)",
            "Yearling Female Store Cattle (€/Kg)",
            "Calves slaughtered <8M (€/100)",
        ],
    )
    production_rows = parse_production_changes_from_text(production_text)
    monthly_production = collect_monthly_production_points(result)
    export_year, export_rows = collect_trade_rankings(result, export=True)
    import_year, import_rows = collect_trade_rankings(result, export=False)

    all_price_rows = carcasse_rows + live_rows
    strongest_yoy = max(all_price_rows, key=lambda item: item.get("yoy") or float("-inf")) if all_price_rows else None
    weakest_mom = min(all_price_rows, key=lambda item: item.get("mom") or float("inf")) if all_price_rows else None
    strongest_wow = max(all_price_rows, key=lambda item: item.get("wow") or float("-inf")) if all_price_rows else None
    weakest_production = min(production_rows, key=lambda item: item.get("eu") or float("inf")) if production_rows else None

    app_update = None
    combined_text = " ".join(
        collect_sheet_text(sheet)
        for sheet in result.get("sheets", [])[:6]
    )
    match = re.search(r"App update:\s*(\d{2}/\d{2}/\d{4})", combined_text)
    if match:
        app_update = match.group(1)

    lines.append("EU Beef Dashboard Summary")
    lines.append(f"Source URL: {meta.get('source_url', 'n/a')}")
    lines.append(f"Dashboard: {dashboard.get('app_name', 'n/a')}")
    if app_update:
        lines.append(f"Dashboard update: {app_update}")
    lines.append(f"Captured at: {meta.get('finished_at', 'n/a')}")
    lines.append(
        "Coverage: "
        f"{summary.get('sheet_count', 0)} sheets, "
        f"{summary.get('data_response_count', 0)} response payloads, "
        f"{summary.get('data_websocket_frame_count', 0)} websocket payloads, "
        f"{summary.get('extracted_table_count', 0)} extracted tables"
    )
    lines.append("")

    lines.append("Carcasse Prices")
    if carcasse_rows:
        lines.append(f"Week: {carcasse_week or 'n/a'}")
        for row in carcasse_rows:
            lines.append(
                f"- {row['category']}: {format_number(row['price'])} "
                f"(WoW {format_signed_percent(row['wow'])}, "
                f"MoM {format_signed_percent(row['mom'])}, "
                f"YoY {format_signed_percent(row['yoy'])})"
            )
    else:
        lines.append("- No structured carcasse summary could be parsed.")
    lines.append("")

    lines.append("Live Animal Prices")
    if live_rows:
        lines.append(f"Week: {live_week or 'n/a'}")
        for row in live_rows:
            lines.append(
                f"- {row['category']}: {format_number(row['price'])} "
                f"(WoW {format_signed_percent(row['wow'])}, "
                f"MoM {format_signed_percent(row['mom'])}, "
                f"YoY {format_signed_percent(row['yoy'])})"
            )
    else:
        lines.append("- No structured live-animal summary could be parsed.")
    lines.append("")

    lines.append("Production")
    if production_rows:
        for row in production_rows:
            lines.append(
                f"- {row['category']}: EU {format_signed_percent(row['eu'])}, "
                f"EU+UK {format_signed_percent(row['eu_uk'])}"
            )
    else:
        lines.append("- No structured production change table could be parsed.")

    if monthly_production:
        latest = monthly_production[-1]
        lines.append(
            f"- Latest monthly production point: {latest['month']} {latest['year']} = "
            f"{format_number(latest['value'], decimals=2)} (1,000 tonnes)"
        )
        previous_year_point = next(
            (
                item for item in monthly_production
                if item["year"] == latest["year"] - 1 and item["month_num"] == latest["month_num"]
            ),
            None,
        )
        if previous_year_point and previous_year_point["value"]:
            yoy_change = ((latest["value"] - previous_year_point["value"]) / previous_year_point["value"]) * 100
            lines.append(
                f"- Same month previous year: {previous_year_point['month']} {previous_year_point['year']} = "
                f"{format_number(previous_year_point['value'], decimals=2)} "
                f"({format_signed_percent(yoy_change)} YoY)"
            )
    lines.append("")

    lines.append("Trade")
    if export_rows:
        top_exporters = "; ".join(
            f"{item['country']} {format_compact_number(item['quantity'])} t"
            for item in export_rows[:5]
        )
        lines.append(f"- Main exporters in {export_year}: {top_exporters}")
    else:
        lines.append("- No structured export ranking could be parsed.")

    if import_rows:
        top_importers = "; ".join(
            f"{item['country']} {format_compact_number(item['quantity'])} t"
            for item in import_rows[:5]
        )
        lines.append(f"- Main importers in {import_year}: {top_importers}")
    else:
        lines.append("- No structured import ranking could be parsed.")
    lines.append("")

    lines.append("Key Developments")
    if strongest_yoy:
        lines.append(
            f"- Strongest YoY price increase: {strongest_yoy['category']} "
            f"({format_signed_percent(strongest_yoy['yoy'])})"
        )
    if strongest_wow:
        lines.append(
            f"- Strongest short-term weekly move: {strongest_wow['category']} "
            f"({format_signed_percent(strongest_wow['wow'])})"
        )
    if weakest_mom:
        lines.append(
            f"- Weakest monthly momentum: {weakest_mom['category']} "
            f"({format_signed_percent(weakest_mom['mom'])})"
        )
    if weakest_production:
        lines.append(
            f"- Largest production contraction: {weakest_production['category']} "
            f"(EU {format_signed_percent(weakest_production['eu'])})"
        )
    if not any([strongest_yoy, strongest_wow, weakest_mom, weakest_production]):
        lines.append("- No notable developments could be extracted from the captured data.")

    return "\n".join(lines).strip() + "\n"


async def collect_dashboard_data(
    url: str,
    output_path: Path,
    headless: bool,
    timeout_ms: int,
    settle_ms: int,
    max_text_chars: int,
    max_ws_frames: int,
) -> dict[str, Any]:
    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(headless=headless)
        context: BrowserContext = await browser.new_context(
            viewport={"width": 1680, "height": 1300},
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        await context.add_init_script(script=INSTRUMENT_WEBSOCKET_SCRIPT)

        dashboard_page = await context.new_page()
        started_at = now_utc_iso()
        await goto_and_settle(dashboard_page, url, timeout_ms=timeout_ms, settle_ms=settle_ms)
        cookie_label = await accept_cookies(dashboard_page)
        await dashboard_page.wait_for_timeout(settle_ms)

        dashboard_html = await dashboard_page.content()
        dashboard_dom = await extract_dom_summary(dashboard_page)
        dashboard_tables = await extract_tables(dashboard_page)
        parsed = parse_dashboard_script(dashboard_html)

        app_mapping = await fetch_app_mapping(context)
        app_id = pick_app_id_from_mapping(app_mapping, parsed.get("app_name"))

        if not app_id:
            iframe_src = None
            if dashboard_dom.get("iframes"):
                iframe_src = dashboard_dom["iframes"][0].get("src")
            if iframe_src:
                app_id = parse_qs(urlparse(iframe_src).query).get("appid", [None])[0]

        if not app_id:
            raise RuntimeError("Could not determine app_id for the dashboard.")

        base = urlparse(url)
        base_host = f"{base.scheme}://{base.netloc}"

        sheet_results = []
        for sheet in parsed.get("sheets", []):
            result = await inspect_sheet(
                context=context,
                base_host=base_host,
                app_id=app_id,
                sheet=sheet,
                timeout_ms=timeout_ms,
                settle_ms=settle_ms,
                max_text_chars=max_text_chars,
                max_ws_frames=max_ws_frames,
                headed=not headless,
            )
            sheet_results.append(result)

        await dashboard_page.close()
        await context.close()
        await browser.close()

    result = {
        "meta": {
            "source_url": url,
            "started_at": started_at,
            "finished_at": now_utc_iso(),
            "headless": headless,
            "timeout_ms": timeout_ms,
            "settle_ms": settle_ms,
            "dashboard_html_sha256": sha256_text(dashboard_html),
            "dashboard_html_length": len(dashboard_html),
        },
        "dashboard": {
            "accepted_cookie_banner": cookie_label,
            "dom": dashboard_dom,
            "tables": dashboard_tables,
            "app_name": parsed.get("app_name"),
            "app_id": app_id,
            "tabs": parsed.get("tabs", []),
            "sheet_count": len(parsed.get("sheets", [])),
            "sheets": parsed.get("sheets", []),
            "app_mapping_excerpt": {parsed.get("app_name"): app_id} if parsed.get("app_name") else {},
        },
        "sheets": sheet_results,
        "summary": {
            "sheet_count": len(sheet_results),
            "sheet_ids": [item["sheet_id"] for item in sheet_results],
            "data_response_count": sum(item["normalized"]["signal_counts"]["response_payloads"] for item in sheet_results),
            "data_websocket_frame_count": sum(item["normalized"]["signal_counts"]["websocket_payloads"] for item in sheet_results),
            "extracted_table_count": sum(item["normalized"]["extracted_table_count"] for item in sheet_results),
        },
    }

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved downloader for the EU beef dashboard.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dashboard URL")
    parser.add_argument("--output", default="beef_dashboard_v2.json", help="Output JSON path")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary text output path (default: <output>_summary.txt)",
    )
    parser.add_argument("--timeout-ms", type=int, default=90000, help="Navigation timeout in ms")
    parser.add_argument("--settle-ms", type=int, default=3500, help="Extra wait after actions in ms")
    parser.add_argument("--max-text-chars", type=int, default=30000, help="Max chars for raw text storage")
    parser.add_argument("--max-ws-frames", type=int, default=250, help="Max stored WS frames per sheet")
    parser.add_argument("--headed", action="store_true", help="Run Chromium with a visible window")
    return parser


async def async_main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_name(f"{output_path.stem}_summary.txt")
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    result = await collect_dashboard_data(
        url=args.url,
        output_path=output_path,
        headless=not args.headed,
        timeout_ms=args.timeout_ms,
        settle_ms=args.settle_ms,
        max_text_chars=args.max_text_chars,
        max_ws_frames=args.max_ws_frames,
    )
    summary_text = build_dashboard_summary_text(result)
    summary_output_path.write_text(summary_text, encoding="utf-8")

    print(f"[OK] wrote {output_path}")
    print(f"[OK] wrote {summary_output_path}")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "summary_output": str(summary_output_path),
                "app_id": result["dashboard"]["app_id"],
                "sheet_count": result["summary"]["sheet_count"],
                "data_response_count": result["summary"]["data_response_count"],
                "data_websocket_frame_count": result["summary"]["data_websocket_frame_count"],
                "extracted_table_count": result["summary"]["extracted_table_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
