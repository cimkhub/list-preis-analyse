import json
import logging
import re
from datetime import date, datetime
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.acquire.base import BaseScraper
from src.convert.pdf_to_images import pdf_to_images, validate_pdf
from src.extract.vision import extract_products_from_pdf_images
from src.models import AcquiredDocument, RawProduct
from src.utils.week import week_dir

logger = logging.getLogger("birkenhof.acquire.handelshof")

TARGET_TABS = {"Aktuell", "Demnächst"}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


class HandelshofScraper(BaseScraper):
    @property
    def supplier_name(self) -> str:
        return "handelshof"

    def get_current_offers(self, week: int, year: int, force: bool = False) -> list[AcquiredDocument]:
        data_dir = Path(self.storage_base) / week_dir("handelshof", week, year)

        if not force:
            cached_docs = self._load_cached_brochures(data_dir, week, year)
            if cached_docs:
                return cached_docs

        try:
            return self._scrape_and_download(week, year, data_dir)
        except Exception as e:
            logger.warning(f"Handelshof scraping failed: {e}", exc_info=True)

        logger.error("No Handelshof PDF data found")
        return []

    def _scrape_and_download(self, week: int, year: int, data_dir: Path) -> list[AcquiredDocument]:
        data_dir.mkdir(parents=True, exist_ok=True)
        legacy_raw_offers = data_dir / "raw_offers.json"
        if legacy_raw_offers.exists():
            legacy_raw_offers.unlink()
        url = self.config.get("url", "https://www.handelshof.de/angebote/hannover")

        page_html = self._fetch_offers_page(url)
        brochures = self._parse_brochure_preview(page_html, url)
        if not brochures:
            raise RuntimeError("No brochure entries found in cmp-brochure-preview")
        self._apply_brochure_selection(brochures)

        selected = [entry for entry in brochures if entry["selected"]]
        logger.info(
            "Handelshof brochure tabs: %s",
            ", ".join(
                f"{tab}={sum(1 for entry in brochures if entry['tab'] == tab)}"
                for tab in ("Aktuell", "Demnächst", "Bestellkataloge")
            ),
        )

        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)

        documents: list[AcquiredDocument] = []
        for brochure in brochures:
            brochure["pdf_status"] = "pending"
            brochure["xlsx_status"] = "pending"
            brochure["pdf_local_path"] = None
            brochure["xlsx_local_path"] = None

            if self._is_duplicate_skip(brochure):
                brochure["pdf_status"] = "skipped_duplicate"
                brochure["xlsx_status"] = "skipped_duplicate"
                continue

            xlsx_path = self._download_article_list(session, brochure, data_dir)
            if xlsx_path:
                brochure["xlsx_local_path"] = str(xlsx_path)
                brochure["xlsx_status"] = "downloaded"
            elif brochure.get("article_list_url"):
                brochure["xlsx_status"] = "failed"
            else:
                brochure["xlsx_status"] = "missing"

            pdf_path = self._download_pdf_entry(session, brochure, data_dir)
            if pdf_path:
                brochure["pdf_local_path"] = str(pdf_path)
                brochure["pdf_status"] = "downloaded"
                if brochure["selected"]:
                    documents.append(self._brochure_to_document(brochure, week, year))
            else:
                brochure["pdf_status"] = "failed"

        self._cleanup_duplicate_assets(data_dir, brochures)
        self._save_raw_brochures(data_dir, brochures)
        logger.info(
            "Handelshof: Downloaded %d/%d brochure PDFs across all tabs",
            sum(1 for entry in brochures if entry["pdf_status"] == "downloaded"),
            len(brochures),
        )
        logger.info(
            "Handelshof: Selected %d/%d brochure PDFs for downstream processing",
            len(documents),
            len(selected),
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
            winner["selected"] = winner.get("selected", False)
            if (winner.get("skip_reason") or "").startswith("duplicate_"):
                winner["skip_reason"] = None

            winner_variant = self._detect_price_variant(winner)
            winner_region = self._extract_region_code(winner)

            for brochure in group:
                if brochure is winner:
                    continue

                brochure["selected"] = False
                brochure_variant = self._detect_price_variant(brochure)
                brochure_region = self._extract_region_code(brochure)

                if brochure_variant != winner_variant:
                    brochure["skip_reason"] = "duplicate_lower_priority_price_type"
                elif brochure_region != winner_region:
                    brochure["skip_reason"] = "duplicate_region_copy"
                else:
                    brochure["skip_reason"] = "duplicate_brochure"

    def _brochure_group_key(self, brochure: dict) -> tuple:
        return (
            brochure.get("tab"),
            brochure.get("catalog_category_name"),
            self._normalize_brochure_title(brochure.get("title")),
            brochure.get("valid_from"),
            brochure.get("valid_to"),
        )

    def _brochure_priority_key(self, brochure: dict) -> tuple:
        variant = self._detect_price_variant(brochure)
        preferred_variant = (self.config.get("price_type") or "").strip().lower() or None
        if variant == preferred_variant:
            variant_rank = 0
        elif variant is not None:
            variant_rank = 1
        else:
            variant_rank = 2

        region = self._extract_region_code(brochure)
        has_region = 1 if region is not None else 0
        region_rank = int(region) if region is not None else -1

        return (
            variant_rank,
            has_region,
            region_rank,
            brochure.get("download_url") or brochure.get("viewer_url") or "",
        )

    def _normalize_brochure_title(self, value: str | None) -> str:
        title = (value or "").strip().lower()
        title = re.sub(r"\[(?:groß|gross|medium)\]", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s+", " ", title)
        return title.strip()

    def _detect_price_variant(self, brochure: dict) -> str | None:
        values = [
            brochure.get("title") or "",
            brochure.get("download_url") or "",
            brochure.get("viewer_url") or "",
            brochure.get("article_list_url") or "",
        ]
        haystack = " ".join(values).lower()
        if "groß" in haystack or "gross" in haystack:
            return "gross"
        if "medium" in haystack:
            return "medium"
        return None

    def _extract_region_code(self, brochure: dict) -> str | None:
        for value in (
            brochure.get("download_url"),
            brochure.get("viewer_url"),
            brochure.get("article_list_url"),
            brochure.get("title"),
        ):
            if not value:
                continue
            match = re.search(r"region[-_](\d+)", value, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _is_duplicate_skip(self, brochure: dict) -> bool:
        return (brochure.get("skip_reason") or "").startswith("duplicate_")

    def _cleanup_duplicate_assets(self, data_dir: Path, brochures: list[dict]) -> None:
        keep_files: set[str] = set()
        remove_files: set[Path] = set()

        for brochure in brochures:
            filenames = self._brochure_asset_filenames(brochure)
            if self._is_duplicate_skip(brochure):
                remove_files.update(data_dir / name for name in filenames)
            else:
                keep_files.update(filenames)

        for path in sorted(remove_files):
            if path.name in keep_files or not path.exists():
                continue
            path.unlink()
            logger.info("Removed duplicate Handelshof asset: %s", path.name)

    def _brochure_asset_filenames(self, brochure: dict) -> set[str]:
        filenames: set[str] = set()
        if brochure.get("pdf_local_path"):
            pdf_filename = Path(str(brochure["pdf_local_path"])).name
        else:
            pdf_filename = self._filename_from_url(
                brochure.get("download_url") or brochure.get("viewer_url") or "",
                f"{brochure.get('title') or 'brochure'}.pdf",
            )
        if pdf_filename:
            filenames.add(pdf_filename)

        if brochure.get("xlsx_local_path"):
            filenames.add(Path(str(brochure["xlsx_local_path"])).name)
        else:
            article_list_url = brochure.get("article_list_url")
            if article_list_url:
                filenames.add(self._filename_from_url(article_list_url, f"{brochure.get('title') or 'brochure'}.xlsx"))
        return filenames

    def _download_article_list(self, session: requests.Session, brochure: dict, data_dir: Path) -> Path | None:
        asset_url = brochure.get("article_list_url")
        if not asset_url:
            return None

        filename = self._filename_from_url(asset_url, f"{brochure['title']}.xlsx")
        dest = data_dir / filename
        try:
            return self._download_asset(session, asset_url, dest)
        except Exception as e:
            if dest.exists():
                dest.unlink()
            logger.warning(f"Handelshof XLSX download failed for {brochure['title']}: {e}")
            return None

    def _download_pdf_entry(self, session: requests.Session, brochure: dict, data_dir: Path) -> Path | None:
        download_url = brochure.get("download_url")
        viewer_url = brochure.get("viewer_url")
        filename = self._filename_from_url(download_url or viewer_url or "", f"{brochure['title']}.pdf")
        dest = data_dir / filename

        if download_url:
            try:
                self._download_asset(session, download_url, dest)
                validate_pdf(str(dest))
                logger.info(f"Downloaded Handelshof PDF: {dest.name}")
                return dest
            except Exception as e:
                if dest.exists():
                    dest.unlink()
                logger.warning(f"Handelshof direct PDF download failed for {brochure['title']}: {e}")

        if not viewer_url:
            return None

        try:
            fallback = self._download_pdf_from_viewer(viewer_url, dest)
            if fallback:
                logger.info(f"Downloaded Handelshof PDF via viewer fallback: {fallback.name}")
            return fallback
        except Exception as e:
            if dest.exists():
                dest.unlink()
            logger.warning(f"Handelshof viewer fallback failed for {brochure['title']}: {e}")
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
                    page.locator("button[title='Download'], button._downloads").first.click(timeout=5000)
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

                download_targets = [
                    "a[href$='.pdf']",
                    "a[href*='.pdf?']",
                    "a[href*='/docs/']",
                    "._downloadsList a",
                    ".popover--downloads a",
                    "._downloadsList li",
                ]
                for selector in download_targets:
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
            if href_lower.endswith(".pdf") or ".pdf?" in href_lower or "/docs/" in href_lower:
                return urljoin(page.url, href)
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
                "price_variant": self._detect_price_variant(brochure),
                "region_code": self._extract_region_code(brochure),
                "valid_from": brochure["valid_from"].isoformat() if brochure.get("valid_from") else None,
                "valid_to": brochure["valid_to"].isoformat() if brochure.get("valid_to") else None,
                "download_url": brochure.get("download_url"),
                "viewer_url": brochure.get("viewer_url"),
                "article_list_url": brochure.get("article_list_url"),
                "catalog_category_name": brochure.get("catalog_category_name"),
                "pdf_status": brochure.get("pdf_status"),
                "pdf_local_path": brochure.get("pdf_local_path"),
                "xlsx_status": brochure.get("xlsx_status"),
                "xlsx_local_path": brochure.get("xlsx_local_path"),
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
                if not item.get("selected") or not pdf_path or not Path(pdf_path).exists():
                    continue
                documents.append(AcquiredDocument(
                    supplier="handelshof",
                    location=self.config.get("location", "hannover"),
                    doc_type="pdf",
                    file_path=pdf_path,
                    url=item.get("viewer_url"),
                    title=item.get("title"),
                    tab=item.get("tab"),
                    valid_from=self._parse_iso_date(item.get("valid_from")),
                    valid_to=self._parse_iso_date(item.get("valid_to")),
                    calendar_week=week,
                    year=year,
                ))

            if documents:
                logger.info(f"Found {len(documents)} cached Handelshof PDFs for KW{week}")
                return documents

        pdfs = sorted(data_dir.glob("*.pdf"))
        if pdfs:
            logger.info(f"Found {len(pdfs)} cached Handelshof PDFs for KW{week} without metadata")
            return [
                AcquiredDocument(
                    supplier="handelshof",
                    location=self.config.get("location", "hannover"),
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
            supplier="handelshof",
            location=self.config.get("location", "hannover"),
            doc_type="pdf",
            file_path=brochure["pdf_local_path"],
            url=brochure.get("viewer_url"),
            title=brochure.get("title"),
            tab=brochure.get("tab"),
            valid_from=brochure.get("valid_from"),
            valid_to=brochure.get("valid_to"),
            calendar_week=week,
            year=year,
        )

    def _resolve_url(self, base_url: str, value: str | None) -> str | None:
        if not value:
            return None
        return urljoin(base_url, value)

    def _filename_from_url(self, url: str, fallback: str) -> str:
        if not url:
            return fallback
        name = unquote(Path(urlparse(url).path).name)
        return name or fallback

    def _parse_config_date(self, value: str | None) -> date | None:
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None

    def _parse_iso_date(self, value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    def extract_products(self, document: AcquiredDocument) -> list[RawProduct]:
        if not document.file_path:
            return []

        year_part = str(document.year or "unknown")
        week_part = f"KW{document.calendar_week:02d}" if document.calendar_week is not None else "KW00"
        images_dir = Path("images") / "handelshof" / year_part / week_part / Path(document.file_path).stem
        image_paths = pdf_to_images(document.file_path, str(images_dir))
        return extract_products_from_pdf_images(
            image_paths,
            supplier="handelshof",
            source_file=document.file_path,
            valid_from=document.valid_from,
            valid_to=document.valid_to,
            calendar_week=document.calendar_week,
            year=document.year,
            location=document.location,
            source_title=document.title,
            source_tab=document.tab,
        )
