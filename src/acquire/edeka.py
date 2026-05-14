import json
import logging
import re
from datetime import date, datetime
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.acquire.base import BaseScraper
from src.convert.pdf_to_images import pdf_to_images, validate_pdf
from src.extract.vision import extract_products_from_pdf_images
from src.models import AcquiredDocument, RawProduct
from src.utils.week import week_dir

logger = logging.getLogger("birkenhof.acquire.edeka")

TARGET_TABS = {"Aktuell", "Demnächst"}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


class EdekaScraper(BaseScraper):
    @property
    def supplier_name(self) -> str:
        return "edeka"

    def get_current_offers(self, week: int, year: int, force: bool = False) -> list[AcquiredDocument]:
        data_dir = Path(self.storage_base) / week_dir("edeka", week, year)

        if not force:
            cached_docs = self._load_cached_brochures(data_dir, week, year)
            if cached_docs:
                return cached_docs

        try:
            return self._scrape_and_download(week, year, data_dir)
        except Exception as e:
            logger.warning(f"EDEKA scraping failed: {e}", exc_info=True)

        manual_dir = Path(self.storage_base) / "edeka"
        if manual_dir.exists():
            pdfs = sorted(manual_dir.glob("*.pdf"))
            if pdfs:
                return [
                    AcquiredDocument(
                        supplier="edeka",
                        location=self.config.get("location", "wernigerode"),
                        doc_type="pdf",
                        file_path=str(pdf),
                        title=pdf.stem,
                        calendar_week=week,
                        year=year,
                    )
                    for pdf in pdfs
                ]

        logger.error("No EDEKA PDF data found")
        return []

    def _scrape_and_download(self, week: int, year: int, data_dir: Path) -> list[AcquiredDocument]:
        data_dir.mkdir(parents=True, exist_ok=True)
        legacy_raw_offers = data_dir / "raw_offers.json"
        if legacy_raw_offers.exists():
            legacy_raw_offers.unlink()

        url = self.config.get("url", "https://www.edeka-foodservice.de/angebote/wernigerode")
        page_html = self._fetch_offers_page(url)
        all_brochures = self._parse_brochure_preview(page_html, url)
        if not all_brochures:
            raise RuntimeError("No brochure entries found in cmp-brochure-preview")

        visible_cards = self._discover_visible_brochures(url)
        if not visible_cards:
            raise RuntimeError("No visible brochure cards found on EDEKA market page")

        brochures = self._match_visible_brochures(all_brochures, visible_cards)
        if not brochures:
            raise RuntimeError("Could not match visible brochure cards to configured downloads")
        self._apply_brochure_selection(brochures)

        logger.info(
            "EDEKA visible brochure cards: %s",
            ", ".join(
                f"{tab}={sum(1 for entry in brochures if entry['tab'] == tab)}"
                for tab in TARGET_TABS
            ),
        )

        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)

        documents: list[AcquiredDocument] = []
        for brochure in brochures:
            brochure["pdf_status"] = "pending"
            brochure["pdf_local_path"] = None

            if self._is_duplicate_skip(brochure):
                brochure["pdf_status"] = "skipped_duplicate"
                continue

            if not brochure["selected"]:
                brochure["pdf_status"] = "skipped_tab"
                continue

            pdf_path = self._download_pdf_entry(session, brochure, data_dir)
            if pdf_path:
                brochure["pdf_status"] = "downloaded"
                brochure["pdf_local_path"] = str(pdf_path)
                documents.append(self._brochure_to_document(brochure, week, year))
            else:
                brochure["pdf_status"] = "failed"

        self._cleanup_duplicate_assets(data_dir, brochures)
        self._cleanup_filename_duplicates(data_dir)
        self._save_raw_brochures(data_dir, brochures)
        logger.info(
            "EDEKA: Downloaded %d/%d brochure PDFs for downstream processing",
            len(documents),
            len(brochures),
        )
        return documents

    def _fetch_offers_page(self, url: str) -> str:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        return response.text

    def _parse_brochure_preview(self, page_html: str, base_url: str) -> list[dict]:
        soup = BeautifulSoup(page_html, "html.parser")
        preview = soup.find("cmp-brochure-preview")
        if not preview or not preview.get("data-config"):
            raise RuntimeError("cmp-brochure-preview[data-config] not found")

        config = json.loads(unescape(preview["data-config"]))
        brochures = []
        for tab in config.get("tabs", []):
            tab_title = (tab.get("title") or "").strip()
            selected = tab_title in TARGET_TABS
            for entry in tab.get("entries", []):
                brochures.append({
                    "tab": tab_title,
                    "title": (entry.get("title") or "").strip(),
                    "selected": selected,
                    "skip_reason": None if selected else "ignored_tab",
                    "valid_from": self._parse_config_date(entry.get("validFrom")),
                    "valid_to": self._parse_config_date(entry.get("validTo")),
                    "download_url": self._resolve_url(base_url, entry.get("downloadLink")),
                    "viewer_url": self._resolve_url(base_url, entry.get("pdfUrl")),
                    "article_list_url": self._resolve_url(base_url, entry.get("articleListDownloadLink")),
                    "catalog_category_name": entry.get("catalogCategoryName"),
                    "region_categories_names": entry.get("regionCategoriesNames"),
                })
        return brochures

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
                family = self._brochure_family(brochure)
                if family == "region":
                    brochure["skip_reason"] = "duplicate_region_copy"
                elif family == "price":
                    brochure["skip_reason"] = "duplicate_lower_priority_price_type"
                else:
                    brochure["skip_reason"] = "duplicate_brochure"

    def _brochure_group_key(self, brochure: dict) -> tuple:
        return (
            brochure.get("tab"),
            brochure.get("catalog_category_name"),
            self._brochure_family(brochure),
            self._normalize_brochure_title(brochure.get("title")),
            brochure.get("valid_from"),
            brochure.get("valid_to"),
        )

    def _brochure_priority_key(self, brochure: dict) -> tuple:
        family = self._brochure_family(brochure)
        if family == "price":
            preferred_variant = (self.config.get("price_type") or "").strip().lower() or None
            variant = self._detect_price_variant(brochure)
            if variant == preferred_variant:
                family_rank = 0
            elif variant is not None:
                family_rank = 1
            else:
                family_rank = 2
            extra_rank = len(brochure.get("download_url") or brochure.get("viewer_url") or "")
        elif family == "region":
            family_rank = self._region_priority_rank(self._extract_region_key(brochure))
            region_key = self._extract_region_key(brochure) or ""
            extra_rank = len(region_key)
        else:
            family_rank = 0
            extra_rank = 0

        return (
            family_rank,
            extra_rank,
            brochure.get("download_url") or brochure.get("viewer_url") or "",
        )

    def _brochure_family(self, brochure: dict) -> str:
        if brochure.get("region_categories_names"):
            return "region"

        title = (brochure.get("title") or "").lower()
        urls = " ".join(
            value.lower()
            for value in (
                brochure.get("download_url"),
                brochure.get("viewer_url"),
            )
            if value
        )
        if re.search(r"\[(?:nord|mitte|süd|sued|west|ost|cz|pl)[^\]]*\]", title, flags=re.IGNORECASE):
            return "region"
        if "basis-" in urls:
            return "region"
        if self._detect_price_variant(brochure):
            return "price"
        return "default"

    def _normalize_brochure_title(self, value: str | None) -> str:
        title = (value or "").strip().lower()
        title = re.sub(r"\[[^\]]+\]", "", title)
        title = re.sub(r"\s+", " ", title)
        return title.strip()

    def _detect_price_variant(self, brochure: dict) -> str | None:
        values = [
            brochure.get("title") or "",
            brochure.get("download_url") or "",
            brochure.get("viewer_url") or "",
        ]
        haystack = " ".join(values).lower()
        if re.search(r"\b(?:gross|groß|brutto)\b", haystack):
            return "gross"
        if re.search(r"\bmedium\b", haystack):
            return "medium"
        return None

    def _normalize_region_value(self, value: str | None) -> str:
        normalized = (value or "").lower()
        normalized = (
            normalized.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )
        normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
        return normalized.strip("-")

    def _extract_region_key(self, brochure: dict) -> str | None:
        patterns = (
            r"nord(?:-pl)?",
            r"mitte(?:-pl)?",
            r"sued(?:-pl)?",
            r"west(?:-pl)?",
            r"ost(?:-pl)?",
            r"cz",
            r"pl",
        )
        for value in (
            brochure.get("region_categories_names"),
            brochure.get("title"),
            brochure.get("download_url"),
            brochure.get("viewer_url"),
        ):
            normalized = self._normalize_region_value(value)
            if not normalized:
                continue
            for pattern in patterns:
                match = re.search(fr"(?:^|-){pattern}(?:-|$)", normalized)
                if match:
                    return match.group(0).strip("-")
        return None

    def _region_priority_rank(self, region_key: str | None) -> int:
        preferred_region = self._normalize_region_value(self.config.get("region"))
        if preferred_region and region_key == preferred_region:
            return 0
        if preferred_region and region_key and region_key.startswith(f"{preferred_region}-"):
            return 1
        if region_key:
            return 2
        return 3

    def _is_duplicate_skip(self, brochure: dict) -> bool:
        return (brochure.get("skip_reason") or "").startswith("duplicate_")

    def _cleanup_duplicate_assets(self, data_dir: Path, brochures: list[dict]) -> None:
        keep_files: set[str] = set()
        remove_files: set[Path] = set()

        for brochure in brochures:
            if brochure.get("pdf_local_path"):
                filename = Path(str(brochure["pdf_local_path"])).name
            else:
                filename = self._filename_from_url(
                    brochure.get("download_url") or brochure.get("viewer_url") or "",
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
            logger.info("Removed duplicate EDEKA asset: %s", path.name)

    def _cleanup_filename_duplicates(self, data_dir: Path) -> None:
        grouped: dict[tuple, list[tuple[Path, dict]]] = {}
        for pdf_path in sorted(data_dir.glob("*.pdf")):
            meta = self._parse_duplicate_filename(pdf_path.name)
            if not meta:
                continue
            grouped.setdefault(
                (meta["year"], meta["calendar_week"], meta["family"]),
                [],
            ).append((pdf_path, meta))

        for group in grouped.values():
            if len(group) <= 1:
                continue
            winner_path, _ = sorted(group, key=lambda item: self._duplicate_filename_priority(item[1], item[0]))[0]
            for pdf_path, _ in group:
                if pdf_path == winner_path or not pdf_path.exists():
                    continue
                pdf_path.unlink()
                logger.info("Removed duplicate EDEKA file by filename rule: %s", pdf_path.name)

    def _parse_duplicate_filename(self, filename: str) -> dict | None:
        match = re.fullmatch(
            r"efs-(\d{4})-kw(\d+)-aktuell-(basis-[a-z0-9-]+|gross|medium)\.pdf",
            filename.lower(),
        )
        if not match:
            return None

        variant = match.group(3)
        family = "region" if variant.startswith("basis-") else "price"
        region_key = variant.removeprefix("basis-") if family == "region" else None
        price_variant = variant if family == "price" else None
        return {
            "year": int(match.group(1)),
            "calendar_week": int(match.group(2)),
            "family": family,
            "region_key": region_key,
            "price_variant": price_variant,
        }

    def _duplicate_filename_priority(self, meta: dict, pdf_path: Path) -> tuple:
        if meta["family"] == "region":
            rank = self._region_priority_rank(meta.get("region_key"))
            extra = len(meta.get("region_key") or "")
        else:
            preferred_variant = (self.config.get("price_type") or "").strip().lower() or None
            rank = 0 if meta.get("price_variant") == preferred_variant else 1
            extra = 0
        return (
            rank,
            extra,
            pdf_path.name,
        )

    def _discover_visible_brochures(self, url: str) -> list[dict]:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent=REQUEST_HEADERS["User-Agent"],
                locale="de-DE",
            )
            page = context.new_page()

            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                self._accept_cookies(page)

                visible = []
                for tab_label in ("Aktuell", "Demnächst"):
                    self._activate_tab(page, tab_label)
                    tab_cards = self._extract_visible_cards(page, tab_label)
                    logger.info(f"EDEKA visible cards in tab '{tab_label}': {len(tab_cards)}")
                    visible.extend(tab_cards)
                return visible
            finally:
                browser.close()

    def _accept_cookies(self, page):
        for selector in (
            "button:has-text('Alle akzeptieren')",
            "button:has-text('Akzeptieren')",
            "button:has-text('Zustimmen')",
        ):
            try:
                page.locator(selector).first.click(timeout=3000)
                page.wait_for_timeout(500)
                return
            except Exception:
                continue

    def _activate_tab(self, page, tab_label: str):
        selectors = [
            f"cmp-brochure-preview button:has-text('{tab_label}')",
            f"button:has-text('{tab_label}')",
            f"a:has-text('{tab_label}')",
        ]
        for selector in selectors:
            try:
                page.locator(selector).first.click(timeout=5000)
                page.wait_for_timeout(1500)
                return
            except Exception:
                continue

        clicked = page.evaluate(
            """label => {
                const elements = Array.from(document.querySelectorAll("button, a"));
                const target = elements.find((el) => ((el.textContent || "").replace(/\\s+/g, " ").trim() === label));
                if (!target) {
                    return false;
                }
                target.click();
                return true;
            }""",
            tab_label,
        )
        if clicked:
            page.wait_for_timeout(1500)

    def _extract_visible_cards(self, page, tab_label: str) -> list[dict]:
        cards = page.evaluate(
            """tabLabel => {
                const normalize = (value) => (value || "").replace(/\\s+/g, " ").trim();
                const isVisible = (el) => {
                    if (!el) {
                        return false;
                    }
                    const style = window.getComputedStyle(el);
                    if (style.display === "none" || style.visibility === "hidden") {
                        return false;
                    }
                    return el.getClientRects().length > 0;
                };
                const findCard = (el) => {
                    let node = el;
                    while (node && node !== document.body) {
                        const text = normalize(node.innerText || node.textContent);
                        if (text.includes("Gültigkeit") && text.includes("Download") && text.includes("Blättern")) {
                            return node;
                        }
                        node = node.parentElement;
                    }
                    return null;
                };

                const results = [];
                const seen = new Set();
                const controls = Array.from(document.querySelectorAll("cmp-brochure-preview button, cmp-brochure-preview a"));
                for (const control of controls) {
                    const label = normalize(control.textContent || control.getAttribute("aria-label") || control.getAttribute("title") || "");
                    if (label !== "Download" || !isVisible(control)) {
                        continue;
                    }

                    const card = findCard(control);
                    if (!card || !isVisible(card)) {
                        continue;
                    }

                    const rawText = (card.innerText || card.textContent || "").trim();
                    if (seen.has(rawText)) {
                        continue;
                    }
                    seen.add(rawText);

                    const links = Array.from(card.querySelectorAll("a[href]"));
                    const downloadLink = links.find((a) => (a.href || "").includes("/dam/jcr:") && (a.href || "").toLowerCase().includes(".pdf"));
                    const viewerLink = links.find((a) => (a.href || "").includes("werbung.edeka-foodservice.de/html5/"));

                    results.push({
                        "tab": tabLabel,
                        "raw_text": rawText,
                        "download_url": downloadLink ? downloadLink.href : null,
                        "viewer_url": viewerLink ? viewerLink.href : null,
                    });
                }

                return results;
            }""",
            tab_label,
        )

        parsed = []
        for card in cards:
            meta = self._parse_card_text(card.get("raw_text", ""), tab_label)
            if not meta:
                continue
            parsed.append({
                "tab": tab_label,
                "title": meta["title"],
                "valid_from": meta.get("valid_from"),
                "valid_to": meta.get("valid_to"),
                "download_url": card.get("download_url"),
                "viewer_url": card.get("viewer_url"),
                "raw_text": card.get("raw_text"),
            })
        return parsed

    def _parse_card_text(self, raw_text: str, tab_label: str) -> dict | None:
        if not raw_text:
            return None

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        ignore = {
            "Gültigkeit:",
            "Artikelliste",
            "Download",
            "Blättern",
            "Aktuell",
            "Demnächst",
            "Bestellkataloge",
        }
        title = next(
            (
                line for line in lines
                if line not in ignore and not line.startswith("Gültigkeit") and not self._looks_like_date_range(line)
            ),
            None,
        )
        if not title:
            return None

        valid_from, valid_to = self._parse_validity_range(raw_text)
        return {
            "tab": tab_label,
            "title": title,
            "valid_from": valid_from,
            "valid_to": valid_to,
        }

    def _looks_like_date_range(self, value: str) -> bool:
        valid_from, valid_to = self._parse_validity_range(value)
        return bool(valid_from and valid_to)

    def _parse_validity_range(self, text: str) -> tuple[date | None, date | None]:
        match = re.search(
            r"(\d{1,2})\.(\d{1,2})\.\s*-\s*(\d{1,2})\.(\d{1,2})\.(\d{4})",
            text,
        )
        if not match:
            return None, None

        year = int(match.group(5))
        return (
            date(year, int(match.group(2)), int(match.group(1))),
            date(year, int(match.group(4)), int(match.group(3))),
        )

    def _match_visible_brochures(self, brochures: list[dict], visible_cards: list[dict]) -> list[dict]:
        matched = []
        seen = set()
        for visible in visible_cards:
            candidates = [
                brochure for brochure in brochures
                if brochure["tab"] == visible["tab"] and brochure["title"] == visible["title"]
            ]

            if visible.get("valid_from") or visible.get("valid_to"):
                exact = [
                    brochure for brochure in candidates
                    if brochure.get("valid_from") == visible.get("valid_from")
                    and brochure.get("valid_to") == visible.get("valid_to")
                ]
                if exact:
                    candidates = exact

            if not candidates:
                logger.warning(
                    "EDEKA: No brochure match for visible card %s (%s)",
                    visible.get("title"),
                    visible.get("tab"),
                )
                continue

            brochure = dict(candidates[0])
            if visible.get("download_url"):
                brochure["download_url"] = visible["download_url"]
            if visible.get("viewer_url"):
                brochure["viewer_url"] = visible["viewer_url"]
            key = (
                brochure.get("tab"),
                brochure.get("title"),
                brochure.get("valid_from"),
                brochure.get("valid_to"),
            )
            if key in seen:
                continue
            seen.add(key)
            matched.append(brochure)

        return matched

    def _download_pdf_entry(self, session: requests.Session, brochure: dict, data_dir: Path) -> Path | None:
        download_url = brochure.get("download_url")
        viewer_url = brochure.get("viewer_url")
        filename = self._filename_from_url(download_url or viewer_url or "", f"{brochure['title']}.pdf")
        dest = data_dir / filename

        if viewer_url:
            try:
                fallback = self._download_pdf_from_viewer(viewer_url, dest)
                if fallback:
                    logger.info(f"Downloaded EDEKA PDF via viewer: {fallback.name}")
                    return fallback
            except Exception as e:
                if dest.exists():
                    dest.unlink()
                logger.warning(f"EDEKA viewer download failed for {brochure['title']}: {e}")

        if not download_url:
            return None

        try:
            self._download_asset(session, download_url, dest)
            validate_pdf(str(dest))
            logger.info(f"Downloaded EDEKA PDF: {dest.name}")
            return dest
        except Exception as e:
            if dest.exists():
                dest.unlink()
            logger.warning(f"EDEKA direct PDF download failed for {brochure['title']}: {e}")
            return None

    def _download_asset(self, session: requests.Session, asset_url: str, dest: Path) -> Path:
        response = session.get(asset_url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest

    def _download_pdf_from_viewer(self, viewer_url: str, dest: Path) -> Path | None:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent=REQUEST_HEADERS["User-Agent"],
                locale="de-DE",
            )
            page = context.new_page()

            try:
                page.goto(viewer_url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(2500)

                pdf_url = self._extract_pdf_url_from_viewer(page)
                if pdf_url:
                    response = context.request.get(pdf_url)
                    content = response.body()
                    if response.status == 200:
                        dest.write_bytes(content)
                        validate_pdf(str(dest))
                        return dest
                    if dest.exists():
                        dest.unlink()

                try:
                    page.locator("button._downloads, button[title='Download']").first.click(timeout=5000)
                    page.wait_for_timeout(1000)
                except Exception:
                    pass

                pdf_url = self._extract_pdf_url_from_viewer(page)
                if pdf_url:
                    response = context.request.get(pdf_url)
                    content = response.body()
                    if response.status == 200:
                        dest.write_bytes(content)
                        validate_pdf(str(dest))
                        return dest
                    if dest.exists():
                        dest.unlink()

                for selector in ("._downloadsList a", ".popover--downloads a", "a[href$='.pdf']", "a[href*='.pdf?']"):
                    locator = page.locator(selector)
                    if locator.count() == 0:
                        continue
                    try:
                        with page.expect_download(timeout=10000) as download_info:
                            locator.first.click()
                        download = download_info.value
                        download.save_as(str(dest))
                        validate_pdf(str(dest))
                        return dest
                    except Exception:
                        if dest.exists():
                            dest.unlink()
                        continue
            finally:
                browser.close()

        return None

    def _extract_pdf_url_from_viewer(self, page) -> str | None:
        hrefs = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
        for href in hrefs:
            href_lower = href.lower()
            if href_lower.endswith(".pdf") or ".pdf?" in href_lower or "/dam/jcr:" in href_lower:
                return urljoin(page.url, href)

        html = page.content()
        for marker in ("/dam/jcr:", ".pdf", "download"):
            if marker not in html:
                continue

        import re

        patterns = [
            r"https://edeka-foodservice\.de/dam/jcr:[^\"'\s>]+\.pdf(?:\?[^\"'\s>]*)?",
            r"/dam/jcr:[^\"'\s>]+\.pdf(?:\?[^\"'\s>]*)?",
            r"href=['\"]([^'\"]+\.pdf(?:\?[^'\"]*)?)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.IGNORECASE)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                return urljoin(page.url, value)
        return None

    def _save_raw_brochures(self, data_dir: Path, brochures: list[dict]):
        raw_file = data_dir / "raw_brochures.json"
        serializable = []
        for brochure in brochures:
            serializable.append({
                "tab": brochure.get("tab"),
                "title": brochure.get("title"),
                "selected": brochure.get("selected"),
                "skip_reason": brochure.get("skip_reason"),
                "brochure_family": self._brochure_family(brochure),
                "price_variant": self._detect_price_variant(brochure),
                "region_key": self._extract_region_key(brochure),
                "valid_from": brochure["valid_from"].isoformat() if brochure.get("valid_from") else None,
                "valid_to": brochure["valid_to"].isoformat() if brochure.get("valid_to") else None,
                "download_url": brochure.get("download_url"),
                "viewer_url": brochure.get("viewer_url"),
                "article_list_url": brochure.get("article_list_url"),
                "catalog_category_name": brochure.get("catalog_category_name"),
                "region_categories_names": brochure.get("region_categories_names"),
                "pdf_status": brochure.get("pdf_status"),
                "pdf_local_path": brochure.get("pdf_local_path"),
            })

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def _load_cached_brochures(self, data_dir: Path, week: int, year: int) -> list[AcquiredDocument]:
        if data_dir.exists():
            self._cleanup_filename_duplicates(data_dir)

        raw_file = data_dir / "raw_brochures.json"
        if raw_file.exists():
            with open(raw_file, encoding="utf-8") as f:
                items = json.load(f)
            self._apply_brochure_selection(items)
            self._cleanup_duplicate_assets(data_dir, items)

            documents = []
            for item in items:
                pdf_path = item.get("pdf_local_path")
                if not item.get("selected") or not pdf_path or not Path(pdf_path).exists():
                    continue
                documents.append(AcquiredDocument(
                    supplier="edeka",
                    location=self.config.get("location", "wernigerode"),
                    doc_type="pdf",
                    file_path=pdf_path,
                    url=item.get("viewer_url"),
                    title=item.get("title"),
                    tab=item.get("tab"),
                    valid_from=self._parse_iso_date(item.get("valid_from")),
                    valid_to=self._parse_iso_date(item.get("valid_to")),
                    category=item.get("catalog_category_name"),
                    calendar_week=week,
                    year=year,
                ))

            if documents:
                logger.info(f"Found {len(documents)} cached EDEKA PDFs for KW{week}")
                return documents

        pdfs = sorted(data_dir.glob("*.pdf"))
        if pdfs:
            logger.info(f"Found {len(pdfs)} cached EDEKA PDFs for KW{week} without metadata")
            return [
                AcquiredDocument(
                    supplier="edeka",
                    location=self.config.get("location", "wernigerode"),
                    doc_type="pdf",
                    file_path=str(pdf),
                    title=pdf.stem,
                    calendar_week=week,
                    year=year,
                )
                for pdf in pdfs
            ]

        return []

    def _brochure_to_document(self, brochure: dict, week: int, year: int) -> AcquiredDocument:
        return AcquiredDocument(
            supplier="edeka",
            location=self.config.get("location", "wernigerode"),
            doc_type="pdf",
            file_path=brochure.get("pdf_local_path"),
            url=brochure.get("viewer_url"),
            title=brochure.get("title"),
            tab=brochure.get("tab"),
            valid_from=brochure.get("valid_from"),
            valid_to=brochure.get("valid_to"),
            category=brochure.get("catalog_category_name"),
            calendar_week=week,
            year=year,
        )

    def _resolve_url(self, base_url: str, value: str | None) -> str | None:
        if not value:
            return None
        return urljoin(base_url, value)

    def _parse_config_date(self, value: str | None) -> date | None:
        if not value:
            return None
        value = value.strip()
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None

    def _parse_iso_date(self, value: str | None) -> date | None:
        if not value:
            return None
        return date.fromisoformat(value)

    def _filename_from_url(self, value: str, fallback: str) -> str:
        if not value:
            return fallback
        filename = Path(urlparse(value).path).name
        return filename or fallback

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
        images_dir = Path("images") / "edeka" / year_part / week_part / doc_id
        image_paths = pdf_to_images(document.file_path, str(images_dir))

        if not image_paths:
            logger.warning(f"No images generated from {document.file_path}")
            return []

        return extract_products_from_pdf_images(
            image_paths,
            supplier="edeka",
            source_file=document.file_path,
            valid_from=document.valid_from,
            valid_to=document.valid_to,
            calendar_week=document.calendar_week,
            year=document.year,
            location=document.location,
            source_title=document.title,
            source_tab=document.tab,
        )
