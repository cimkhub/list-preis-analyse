import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from src.acquire.base import BaseScraper
from src.convert.pdf_to_images import pdf_to_images, validate_pdf
from src.extract.text_extract import classify_category, extract_selgros_filename_meta
from src.extract.vision import extract_products_from_pdf_images
from src.models import AcquiredDocument, RawProduct
from src.utils.week import week_dir

logger = logging.getLogger("birkenhof.acquire.selgros")

BASE_URL = "https://www.selgros.de"
VIEWER_PATH_FRAGMENT = "/sites/default/files/offers/"
TARGET_TABS = ("Aktuelle Angebote", "Vorschau")
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


class SelgrosScraper(BaseScraper):
    @property
    def supplier_name(self) -> str:
        return "selgros"

    def get_current_offers(self, week: int, year: int, force: bool = False) -> list[AcquiredDocument]:
        data_dir = Path(self.storage_base) / week_dir("selgros", week, year)

        if not force:
            cached_docs = self._load_cached_brochures(data_dir, week, year)
            if cached_docs:
                return cached_docs

        try:
            return self._scrape_and_download(week, year, data_dir)
        except Exception as e:
            logger.warning(f"Selgros scraping failed: {e}", exc_info=True)

        manual_dir = Path(self.storage_base) / "selgros"
        if manual_dir.exists():
            pdfs = sorted(manual_dir.glob("*.pdf"))
            if pdfs:
                return self._pdfs_to_documents(pdfs, week, year)

        logger.error("No Selgros PDF data found")
        return []

    def _scrape_and_download(self, week: int, year: int, data_dir: Path) -> list[AcquiredDocument]:
        from playwright.sync_api import sync_playwright

        data_dir.mkdir(parents=True, exist_ok=True)
        market_url = self.config.get("url", f"{BASE_URL}/markt/braunschweig")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent=REQUEST_HEADERS["User-Agent"],
                locale="de-DE",
            )
            page = context.new_page()

            try:
                brochures = self._discover_brochures(page, market_url)
                documents: list[AcquiredDocument] = []

                for brochure in brochures:
                    brochure["pdf_status"] = "pending"
                    brochure["pdf_local_path"] = None
                    brochure["pdf_url"] = None

                    if self._is_duplicate_skip(brochure):
                        brochure["pdf_status"] = "skipped_duplicate"
                        continue

                    try:
                        pdf_path = self._download_brochure_pdf(context, brochure, data_dir)
                    except Exception as e:
                        brochure["pdf_status"] = "failed"
                        logger.warning(
                            "Selgros brochure download failed for %s: %s",
                            brochure.get("viewer_url"),
                            e,
                        )
                        continue
                    if not pdf_path:
                        brochure["pdf_status"] = "failed"
                        continue

                    brochure["pdf_status"] = "downloaded"
                    brochure["pdf_local_path"] = str(pdf_path)
                    documents.append(self._brochure_to_document(brochure, week, year))

                self._cleanup_duplicate_assets(data_dir, brochures)
                self._save_raw_brochures(data_dir, brochures)
                logger.info(
                    "Selgros: Downloaded %d/%d brochure PDFs from %s",
                    len(documents),
                    len(brochures),
                    market_url,
                )
                return documents
            finally:
                browser.close()

    def _discover_brochures(self, page, market_url: str) -> list[dict]:
        self._goto_page(page, market_url)
        self._accept_cookies(page)

        brochures: list[dict] = []
        for tab_label in TARGET_TABS:
            self._activate_tab(page, tab_label)
            tab_entries = self._extract_visible_brochures(page, tab_label)
            logger.info(f"Selgros: Found {len(tab_entries)} viewer links in tab '{tab_label}'")
            brochures.extend(tab_entries)

        if not brochures:
            logger.warning("Selgros: No viewer links found via tabbed scan, using fallback page scan")
            brochures = self._extract_visible_brochures(page, None)

        brochures = self._dedupe_brochures(brochures)
        self._apply_brochure_selection(brochures)
        return brochures

    def _activate_tab(self, page, tab_label: str):
        try:
            tab = page.locator("button[role='tab']").filter(has_text=tab_label).first
            tab.wait_for(timeout=10000)
            tab.click(timeout=5000)
        except Exception:
            clicked = page.evaluate(
                """label => {
                    const buttons = Array.from(document.querySelectorAll("button[role='tab'], button"));
                    const target = buttons.find((button) => {
                        const text = (button.textContent || "").replace(/\\s+/g, " ").trim();
                        return text === label;
                    });
                    if (!target) {
                        return false;
                    }
                    target.click();
                    return true;
                }""",
                tab_label,
            )
            if not clicked:
                logger.debug(f"Selgros tab '{tab_label}' was not clickable")
        page.wait_for_timeout(2000)

    def _extract_visible_brochures(self, page, tab_label: str | None) -> list[dict]:
        try:
            page.wait_for_function(
                f"""() => Array.from(document.querySelectorAll("a[href], tg-card[full-link], tg-catalog[full-link]"))
                    .some((el) => {{
                        const href = el.getAttribute("href") || el.getAttribute("full-link") || "";
                        if (!href.includes("{VIEWER_PATH_FRAGMENT}")) {{
                            return false;
                        }}
                        const node = el.closest("tg-card, tg-catalog, article, li, div") || el;
                        return node.getClientRects().length > 0 || el.getClientRects().length > 0;
                    }})""",
                timeout=10000,
            )
        except Exception:
            logger.debug(f"Selgros: No visible viewer links became ready for tab '{tab_label}'")

        entries = page.evaluate(
            f"""tabLabel => {{
                const normalize = (value) => (value || "").replace(/\\s+/g, " ").trim();
                const isVisible = (node) => {{
                    if (!node) {{
                        return false;
                    }}
                    const style = window.getComputedStyle(node);
                    if (style.display === "none" || style.visibility === "hidden") {{
                        return false;
                    }}
                    return node.getClientRects().length > 0;
                }};

                const results = [];
                const seen = new Set();
                const nodes = Array.from(document.querySelectorAll("a[href], tg-card[full-link], tg-catalog[full-link]"));
                for (const el of nodes) {{
                    const rawHref = el.getAttribute("href") || el.getAttribute("full-link") || "";
                    if (!rawHref.includes("{VIEWER_PATH_FRAGMENT}") || !rawHref.includes("index.html")) {{
                        continue;
                    }}

                    const container = el.closest("tg-card, tg-catalog, article, li, div") || el;
                    if (!isVisible(container) && !isVisible(el)) {{
                        continue;
                    }}

                    const href = new URL(rawHref, window.location.origin).href;
                    if (seen.has(href)) {{
                        continue;
                    }}
                    seen.add(href);

                    const candidateTexts = [
                        el.getAttribute("title"),
                        el.getAttribute("aria-label"),
                        el.querySelector("img") ? el.querySelector("img").getAttribute("alt") : "",
                        container.querySelector("img") ? container.querySelector("img").getAttribute("alt") : "",
                        el.innerText,
                        container.innerText,
                    ];

                    const title = candidateTexts
                        .map(normalize)
                        .find((value) => value && value !== tabLabel) || "";

                    results.push({{
                        tab: tabLabel || "",
                        title: title.split(" Gültig")[0].split("\\n")[0].trim(),
                        viewer_url: href,
                    }});
                }}

                return results;
            }}""",
            tab_label or "",
        )

        return [
            self._build_brochure_entry(
                tab=entry.get("tab") or tab_label or "",
                title=entry.get("title") or "",
                viewer_url=entry.get("viewer_url"),
            )
            for entry in entries
            if entry.get("viewer_url")
        ]

    def _build_brochure_entry(self, tab: str, title: str, viewer_url: str) -> dict:
        meta = self._parse_flipbook_url(viewer_url) or {}
        fallback_title = title.strip() or meta.get("category_name") or Path(urlparse(viewer_url).path).parts[-2]
        return {
            "tab": tab,
            "title": fallback_title,
            "selected": tab in TARGET_TABS,
            "skip_reason": None if tab in TARGET_TABS else "ignored_tab",
            "viewer_url": viewer_url,
            "pdf_url": None,
            "valid_from": meta.get("valid_from"),
            "valid_to": meta.get("valid_to"),
            "category_name": meta.get("category_name"),
            "scope": meta.get("scope"),
            "folder": meta.get("folder"),
            "pdf_filename": meta.get("pdf_filename"),
        }

    def _dedupe_brochures(self, brochures: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for brochure in brochures:
            viewer_url = brochure.get("viewer_url")
            if not viewer_url or viewer_url in seen:
                continue
            seen.add(viewer_url)
            deduped.append(brochure)
        return deduped

    def _apply_brochure_selection(self, brochures: list[dict]) -> None:
        grouped: dict[tuple, list[dict]] = {}
        for brochure in brochures:
            grouped.setdefault(self._brochure_group_key(brochure), []).append(brochure)

        for group in grouped.values():
            if len(group) <= 1:
                continue

            winner = sorted(group, key=self._brochure_priority_key)[0]
            if (winner.get("skip_reason") or "").startswith("duplicate_"):
                winner["skip_reason"] = None

            for brochure in group:
                if brochure is winner:
                    continue
                brochure["selected"] = False
                brochure["skip_reason"] = "duplicate_scope_copy"

    def _brochure_group_key(self, brochure: dict) -> tuple:
        return (
            brochure.get("tab"),
            (brochure.get("category_name") or "").strip().lower(),
            (brochure.get("title") or "").strip().lower(),
            brochure.get("valid_from"),
            brochure.get("valid_to"),
        )

    def _brochure_priority_key(self, brochure: dict) -> tuple:
        scope = self._extract_scope(brochure)
        if scope == "ALLE":
            scope_rank = 0
        elif scope == "TEIL":
            scope_rank = 1
        else:
            scope_rank = 2

        return (
            scope_rank,
            brochure.get("viewer_url") or "",
        )

    def _extract_scope(self, brochure: dict) -> str | None:
        scope = brochure.get("scope")
        if scope:
            return str(scope).upper()

        viewer_url = brochure.get("viewer_url") or ""
        match = re.search(r"/\d+_([^_/]+)_\d{8}_\d{8}_.+?_(F|A)(?:/|$)", viewer_url)
        if match:
            return match.group(1).upper()
        return None

    def _is_duplicate_skip(self, brochure: dict) -> bool:
        return (brochure.get("skip_reason") or "").startswith("duplicate_")

    def _cleanup_duplicate_assets(self, data_dir: Path, brochures: list[dict]) -> None:
        keep_files: set[str] = set()
        remove_files: set[Path] = set()

        for brochure in brochures:
            if brochure.get("pdf_local_path"):
                filename = Path(str(brochure["pdf_local_path"])).name
            else:
                filename = brochure.get("pdf_filename") or self._filename_from_pdf_url(
                    brochure.get("pdf_url") or brochure.get("viewer_url") or "",
                    f"{brochure.get('title') or 'brochure'}.pdf",
                )
            if not filename:
                continue
            if self._is_duplicate_skip(brochure):
                remove_files.add(data_dir / filename)
            else:
                keep_files.add(filename)

        for path in sorted(remove_files):
            if path.name in keep_files or not path.exists():
                continue
            path.unlink()
            logger.info("Removed duplicate Selgros asset: %s", path.name)

    def _parse_flipbook_url(self, url: str) -> dict | None:
        match = re.search(r"/(\d+)_([^_/]+)_(\d{8})_(\d{8})_(.+?)_(F|A)(?:/|$)", url)
        if not match:
            return None

        valid_from = None
        valid_to = None
        try:
            valid_from = self._parse_compact_date(match.group(3))
            valid_to = self._parse_compact_date(match.group(4))
        except ValueError:
            pass

        folder = (
            f"{match.group(1)}_{match.group(2)}_{match.group(3)}_"
            f"{match.group(4)}_{match.group(5)}_{match.group(6)}"
        )

        return {
            "catalog_id": match.group(1),
            "scope": match.group(2),
            "valid_from": valid_from,
            "valid_to": valid_to,
            "category_name": match.group(5),
            "folder": folder,
            "pdf_filename": f"{folder}_kl.pdf",
        }

    def _parse_compact_date(self, value: str):
        from datetime import date

        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))

    def _download_brochure_pdf(self, context, brochure: dict, data_dir: Path) -> Path | None:
        viewer_url = brochure["viewer_url"]
        default_name = brochure.get("pdf_filename") or f"{Path(urlparse(viewer_url).path).parts[-2]}.pdf"
        default_dest = data_dir / default_name

        if default_dest.exists():
            try:
                validate_pdf(str(default_dest))
                brochure["pdf_url"] = brochure.get("pdf_url") or viewer_url
                return default_dest
            except Exception:
                default_dest.unlink()

        page = context.new_page()
        try:
            self._goto_page(page, viewer_url)
            pdf_url = self._extract_pdf_url_from_viewer(page)
            if pdf_url:
                brochure["pdf_url"] = pdf_url
                filename = self._filename_from_pdf_url(pdf_url, default_name)
                dest = data_dir / filename
                result = self._download_and_validate_pdf(context, pdf_url, dest)
                if result:
                    logger.info(f"Downloaded Selgros PDF: {result.name}")
                    return result

            if self._click_download_control(page):
                page.wait_for_timeout(1000)
                pdf_url = self._extract_pdf_url_from_viewer(page)
                if pdf_url:
                    brochure["pdf_url"] = pdf_url
                    filename = self._filename_from_pdf_url(pdf_url, default_name)
                    dest = data_dir / filename
                    result = self._download_and_validate_pdf(context, pdf_url, dest)
                    if result:
                        logger.info(f"Downloaded Selgros PDF via viewer control: {result.name}")
                        return result

                result = self._capture_download_after_click(page, data_dir, default_name)
                if result:
                    brochure["pdf_url"] = brochure.get("pdf_url") or viewer_url
                    logger.info(f"Downloaded Selgros PDF via browser download: {result.name}")
                    return result
        finally:
            page.close()

        logger.warning(f"Selgros: Could not resolve PDF from viewer {viewer_url}")
        return None

    def _extract_pdf_url_from_viewer(self, page) -> str | None:
        html = page.content()
        pdf_url = self._extract_pdf_url_from_html(html, page.url)
        if pdf_url:
            return pdf_url

        hrefs = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
        for href in hrefs:
            href_lower = href.lower()
            if href_lower.endswith(".pdf") or ".pdf?" in href_lower or "/docs/" in href_lower:
                return urljoin(page.url, href)

        onclicks = page.eval_on_selector_all("[onclick]", "els => els.map(el => el.getAttribute('onclick'))")
        for onclick in onclicks:
            if not onclick:
                continue
            match = re.search(r"downloadPDF\\(['\\\"]([^'\\\"]+\\.pdf(?:\\?[^'\\\"]*)?)['\\\"]\\)", onclick)
            if match:
                return self._resolve_viewer_pdf_url(page.url, match.group(1))

        return None

    def _extract_pdf_url_from_html(self, html: str, viewer_url: str) -> str | None:
        patterns = [
            r"downloadPDF\(['\"]([^'\"]+\.pdf(?:\?[^'\"]*)?)['\"]\)",
            r"href=['\"]([^'\"]+\.pdf(?:\?[^'\"]*)?)['\"]",
            r"['\"]([^'\"]+/docs/[^'\"]+\.pdf(?:\?[^'\"]*)?)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.IGNORECASE)
            if match:
                return self._resolve_viewer_pdf_url(viewer_url, match.group(1))
        return None

    def _resolve_viewer_pdf_url(self, viewer_url: str, candidate: str) -> str:
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate
        if candidate.startswith("/"):
            return urljoin(BASE_URL, candidate)
        if "/docs/" in candidate:
            return urljoin(viewer_url, candidate)
        viewer_base = viewer_url.rsplit("/", 1)[0]
        return f"{viewer_base}/docs/{candidate.lstrip('/')}"

    def _download_and_validate_pdf(self, context, pdf_url: str, dest: Path) -> Path | None:
        try:
            response = context.request.get(pdf_url)
            content = response.body()
            if response.status != 200 or not content.startswith(b"%PDF-"):
                return None

            dest.write_bytes(content)
            validate_pdf(str(dest))
            return dest
        except Exception as e:
            if dest.exists():
                dest.unlink()
            logger.debug(f"Selgros PDF download failed for {pdf_url}: {e}")
            return None

    def _click_download_control(self, page) -> bool:
        selectors = [
            "button[title='Download']",
            "[aria-label='Download']",
            "button:has-text('Download')",
            "a:has-text('Download')",
            "text=Download",
        ]
        for selector in selectors:
            locator = page.locator(selector).first
            try:
                locator.click(timeout=3000)
                return True
            except Exception:
                continue
        return False

    def _capture_download_after_click(self, page, data_dir: Path, default_name: str) -> Path | None:
        selectors = [
            "button[title='Download']",
            "[aria-label='Download']",
            "button:has-text('Download')",
            "a:has-text('Download')",
        ]
        for selector in selectors:
            locator = page.locator(selector).first
            if locator.count() == 0:
                continue
            try:
                with page.expect_download(timeout=10000) as download_info:
                    locator.click()
                download = download_info.value
                filename = download.suggested_filename or default_name
                dest = data_dir / filename
                download.save_as(str(dest))
                validate_pdf(str(dest))
                return dest
            except Exception:
                continue
        return None

    def _accept_cookies(self, page):
        selectors = [
            "button:has-text('Alle Cookies akzeptieren')",
            "button:has-text('Alle akzeptieren')",
            "button:has-text('Akzeptieren')",
            "button:has-text('OK')",
        ]
        for selector in selectors:
            try:
                page.locator(selector).first.click(timeout=3000)
                page.wait_for_timeout(500)
                return
            except Exception:
                continue

    def _goto_page(self, page, url: str, timeout: int = 45000):
        page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)

    def _filename_from_pdf_url(self, pdf_url: str, fallback: str) -> str:
        parsed = urlparse(pdf_url)
        filename = Path(parsed.path).name
        return filename or fallback

    def _save_raw_brochures(self, data_dir: Path, brochures: list[dict]):
        raw_file = data_dir / "raw_brochures.json"
        serializable = []
        for brochure in brochures:
            serializable.append({
                "tab": brochure.get("tab"),
                "title": brochure.get("title"),
                "selected": brochure.get("selected"),
                "skip_reason": brochure.get("skip_reason"),
                "viewer_url": brochure.get("viewer_url"),
                "pdf_url": brochure.get("pdf_url"),
                "scope": self._extract_scope(brochure),
                "valid_from": brochure["valid_from"].isoformat() if brochure.get("valid_from") else None,
                "valid_to": brochure["valid_to"].isoformat() if brochure.get("valid_to") else None,
                "category_name": brochure.get("category_name"),
                "pdf_status": brochure.get("pdf_status"),
                "pdf_local_path": brochure.get("pdf_local_path"),
            })

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def _load_cached_brochures(self, data_dir: Path, week: int, year: int) -> list[AcquiredDocument]:
        raw_file = data_dir / "raw_brochures.json"
        if raw_file.exists():
            with open(raw_file, encoding="utf-8") as f:
                items = json.load(f)
            self._apply_brochure_selection(items)
            self._cleanup_duplicate_assets(data_dir, items)

            documents = []
            for item in items:
                pdf_path = item.get("pdf_local_path")
                if not item.get("selected", True) or not pdf_path or not Path(pdf_path).exists():
                    continue
                documents.append(AcquiredDocument(
                    supplier="selgros",
                    location=self.config.get("location", "braunschweig"),
                    doc_type="pdf",
                    file_path=pdf_path,
                    url=item.get("viewer_url"),
                    title=item.get("title"),
                    tab=item.get("tab"),
                    valid_from=self._parse_iso_date(item.get("valid_from")),
                    valid_to=self._parse_iso_date(item.get("valid_to")),
                    category=item.get("category_name"),
                    calendar_week=week,
                    year=year,
                ))

            if documents:
                logger.info(f"Found {len(documents)} cached Selgros PDFs for KW{week}")
                return documents

        pdfs = sorted(data_dir.glob("*.pdf"))
        if pdfs:
            logger.info(f"Found {len(pdfs)} cached Selgros PDFs for KW{week} without metadata")
            return self._pdfs_to_documents(pdfs, week, year)
        return []

    def _parse_iso_date(self, value: str | None):
        if not value:
            return None
        from datetime import date

        return date.fromisoformat(value)

    def _brochure_to_document(self, brochure: dict, week: int, year: int) -> AcquiredDocument:
        return AcquiredDocument(
            supplier="selgros",
            location=self.config.get("location", "braunschweig"),
            doc_type="pdf",
            file_path=brochure.get("pdf_local_path"),
            url=brochure.get("viewer_url"),
            title=brochure.get("title"),
            tab=brochure.get("tab"),
            valid_from=brochure.get("valid_from"),
            valid_to=brochure.get("valid_to"),
            category=brochure.get("category_name"),
            calendar_week=week,
            year=year,
        )

    def _pdfs_to_documents(self, pdfs: list[Path], week: int, year: int) -> list[AcquiredDocument]:
        docs = []
        for pdf in pdfs:
            meta = extract_selgros_filename_meta(pdf.name)
            docs.append(AcquiredDocument(
                supplier="selgros",
                location=self.config.get("location", "braunschweig"),
                file_path=str(pdf),
                valid_from=meta.get("valid_from"),
                valid_to=meta.get("valid_to"),
                category=meta.get("category", ""),
                title=pdf.stem,
                calendar_week=week,
                year=year,
            ))

        logger.info(f"Found {len(docs)} Selgros PDFs for KW{week}")
        return docs

    def extract_products(self, document: AcquiredDocument) -> list[RawProduct]:
        if not document.file_path:
            return []

        file_path = Path(document.file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []

        doc_id = file_path.stem
        year_part = str(document.year or "unknown")
        week_part = f"KW{document.calendar_week:02d}" if document.calendar_week is not None else "KW00"
        images_dir = Path("images") / "selgros" / year_part / week_part / doc_id
        image_paths = pdf_to_images(document.file_path, str(images_dir))

        if not image_paths:
            logger.warning(f"No images generated from {document.file_path}")
            return []

        fallback_category = classify_category(document.category) if document.category else None
        return extract_products_from_pdf_images(
            image_paths,
            supplier="selgros",
            source_file=document.file_path,
            valid_from=document.valid_from,
            valid_to=document.valid_to,
            calendar_week=document.calendar_week,
            year=document.year,
            location=document.location,
            source_title=document.title,
            source_tab=document.tab,
            fallback_category=fallback_category,
        )
