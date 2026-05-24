#!/usr/bin/env python3
"""Upload an Excel file to a shared OneDrive folder through the browser UI.

This is a fallback for client-shared folders where Microsoft Graph cannot resolve
the sharing link, but the folder can be opened in a browser and accepts uploads.

Requirements:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import tempfile
import time
from pathlib import Path

from playwright.async_api import (
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FILE = ROOT / "parsed" / "KW21_2026" / "Artikelvergleich KW21.xlsx"
DEFAULT_TARGET_URL = (
    "https://listgs-my.sharepoint.com/:f:/g/personal/l_kornblum_list-goslar_com/"
    "IgBaiyRxCvptRJ8SzmnU7jQQAdF8FH2QUuMjOUhlkqxI0tc?e=OkXLLr"
)
UPLOAD_BUTTON_RE = re.compile(r"^(Hochladen|Upload)$", re.IGNORECASE)
UPLOAD_MENU_RE = re.compile(
    r"(Dateien|Files|Dateien hochladen|Upload files|File upload)",
    re.IGNORECASE,
)
REPLACE_BUTTON_RE = re.compile(r"^(Ersetzen|Replace|Überschreiben|Overwrite)$", re.IGNORECASE)
KEEP_BOTH_BUTTON_RE = re.compile(r"(Beide behalten|Keep both)", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open OneDrive in a real browser and upload a file through the web UI."
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="Local file to upload.")
    parser.add_argument("--target-url", default=DEFAULT_TARGET_URL, help="Shared OneDrive folder URL.")
    parser.add_argument("--filename", help="Optional upload filename. A temporary copy is created.")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Optional persistent browser profile. By default, no cookies/session are saved.",
    )
    parser.add_argument("--timeout", type=int, default=240, help="Max seconds for UI waits.")
    parser.add_argument("--headless", action="store_true", help="Run without visible browser.")
    parser.add_argument(
        "--keep-both",
        action="store_true",
        help="If OneDrive asks about a duplicate, choose 'keep both' instead of replace.",
    )
    parser.add_argument(
        "--login-pause",
        action="store_true",
        help="Pause so you can manually complete Microsoft login/email-code verification.",
    )
    parser.add_argument(
        "--no-login-pause",
        action="store_true",
        help="Compatibility flag. Login pause is already disabled by default.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_file = args.file.expanduser().resolve()
    if not source_file.exists():
        raise FileNotFoundError(f"Upload file not found: {source_file}")

    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    upload_file = source_file
    if args.filename and args.filename != source_file.name:
        tmp_dir = tempfile.TemporaryDirectory(prefix="onedrive-upload-")
        upload_file = Path(tmp_dir.name) / args.filename
        shutil.copy2(source_file, upload_file)

    try:
        asyncio.run(
            run_upload(
                file_path=upload_file,
                target_url=args.target_url,
                profile_dir=args.profile_dir.expanduser().resolve() if args.profile_dir else None,
                timeout_seconds=args.timeout,
                headless=args.headless,
                keep_both=args.keep_both,
                login_pause=args.login_pause and not args.no_login_pause,
            )
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


async def run_upload(
    file_path: Path,
    target_url: str,
    profile_dir: Path | None,
    timeout_seconds: int,
    headless: bool,
    keep_both: bool,
    login_pause: bool,
) -> None:
    timeout_ms = timeout_seconds * 1000

    async with async_playwright() as p:
        browser = None
        if profile_dir:
            profile_dir.mkdir(parents=True, exist_ok=True)
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=headless,
                accept_downloads=True,
                viewport={"width": 1440, "height": 950},
            )
        else:
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(
                accept_downloads=True,
                viewport={"width": 1440, "height": 950},
            )
        page = context.pages[0] if context.pages else await context.new_page()
        try:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.bring_to_front()
            print(f"Opened OneDrive folder: {target_url}")

            if login_pause and not headless:
                print(
                    "\nIf the browser asks for an email code or Microsoft login, finish it now."
                    "\nAfter the OneDrive folder is visible, press Enter here to continue."
                )
                await asyncio.to_thread(input)

            await wait_for_folder_ui(page, timeout_ms)
            await upload_via_file_chooser(page, file_path, timeout_ms)
            await handle_duplicate_dialog(page, keep_both=keep_both)
            await wait_for_uploaded_file(page, file_path.name, timeout_ms)
            print(f"Uploaded via browser UI: {file_path.name}")
        finally:
            await context.close()
            if browser is not None:
                await browser.close()


async def wait_for_folder_ui(page: Page, timeout_ms: int) -> None:
    candidates = [
        page.get_by_role("button", name=UPLOAD_BUTTON_RE).first,
        page.get_by_text(UPLOAD_BUTTON_RE).first,
        page.locator("[aria-label*='Upload']").first,
        page.locator("[aria-label*='Hochladen']").first,
    ]
    last_error: Exception | None = None
    for locator in candidates:
        try:
            await locator.wait_for(state="visible", timeout=min(timeout_ms, 30000))
            return
        except PlaywrightError as exc:
            last_error = exc
    raise RuntimeError(
        "Could not find the OneDrive upload button. Make sure the shared folder is open "
        "and your account has edit/upload permission."
    ) from last_error


async def upload_via_file_chooser(page: Page, file_path: Path, timeout_ms: int) -> None:
    upload_button = await first_visible_locator(
        page,
        [
            page.get_by_role("button", name=UPLOAD_BUTTON_RE).first,
            page.locator("[aria-label*='Upload']").first,
            page.locator("[aria-label*='Hochladen']").first,
            page.get_by_text(UPLOAD_BUTTON_RE).first,
        ],
    )
    if upload_button is None:
        raise RuntimeError("Upload button not found.")

    try:
        async with page.expect_file_chooser(timeout=5000) as chooser_info:
            await upload_button.click()
        chooser = await chooser_info.value
        await chooser.set_files(str(file_path))
        return
    except PlaywrightTimeoutError:
        # Most OneDrive tenants open a menu first; choose "Files/Dateien".
        pass

    menu_item = await first_visible_locator(
        page,
        [
            page.get_by_role("menuitem", name=UPLOAD_MENU_RE).first,
            page.get_by_role("button", name=UPLOAD_MENU_RE).first,
            page.get_by_text(UPLOAD_MENU_RE).first,
        ],
    )
    if menu_item is None:
        raise RuntimeError("Upload menu opened, but no 'Files/Dateien' option was found.")

    async with page.expect_file_chooser(timeout=timeout_ms) as chooser_info:
        await menu_item.click()
    chooser = await chooser_info.value
    await chooser.set_files(str(file_path))


async def handle_duplicate_dialog(page: Page, keep_both: bool) -> None:
    button_re = KEEP_BOTH_BUTTON_RE if keep_both else REPLACE_BUTTON_RE
    try:
        await page.get_by_role("button", name=button_re).first.click(timeout=10000)
        print("Handled OneDrive duplicate-file dialog.")
    except PlaywrightError:
        return


async def wait_for_uploaded_file(page: Page, filename: str, timeout_ms: int) -> None:
    end_time = time.monotonic() + timeout_ms / 1000
    file_text = page.get_by_text(filename, exact=False).first
    status_re = re.compile(
        r"(hochgeladen|upload(ed)? complete|upload abgeschlossen|fertig|done)",
        re.IGNORECASE,
    )
    while time.monotonic() < end_time:
        try:
            if await file_text.is_visible(timeout=1000):
                return
        except PlaywrightError:
            pass
        try:
            if await page.get_by_text(status_re).first.is_visible(timeout=1000):
                return
        except PlaywrightError:
            pass
        await page.wait_for_timeout(1000)
    raise RuntimeError(
        f"Upload may still be running, but '{filename}' was not visible after "
        f"{timeout_ms // 1000} seconds."
    )


async def first_visible_locator(page: Page, locators: list[object]):
    del page
    for locator in locators:
        try:
            if await locator.is_visible(timeout=3000):
                return locator
        except PlaywrightError:
            continue
    return None


if __name__ == "__main__":
    main()
