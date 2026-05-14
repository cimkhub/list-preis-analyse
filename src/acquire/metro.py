import logging
import re
import json
import hashlib
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

from src.acquire.base import BaseScraper
from src.convert.pdf_to_images import pdf_to_images, validate_pdf
from src.extract.vision import extract_products_from_pdf_images
from src.models import AcquiredDocument, RawProduct
from src.utils.week import week_dir, week_date_range

logger = logging.getLogger("birkenhof.acquire.metro")


class MetroScraper(BaseScraper):
    @property
    def supplier_name(self) -> str:
        return "metro"

    def get_current_offers(self, week: int, year: int, force: bool = False) -> list[AcquiredDocument]:
        data_dir = Path(self.storage_base) / week_dir("metro", week, year)

        # Check if PDFs already downloaded (only if not forcing)
        if not force and data_dir.exists():
            pdfs = self._dedupe_pdf_files(list(data_dir.glob("*.pdf")))
            if pdfs:
                docs = []
                for pdf in pdfs:
                    meta = self._parse_filename_dates(pdf.name)
                    docs.append(AcquiredDocument(
                        supplier="metro",
                        location=self.config.get("location", "goslar"),
                        file_path=str(pdf),
                        valid_from=meta.get("valid_from"),
                        valid_to=meta.get("valid_to"),
                        calendar_week=week,
                        year=year,
                    ))
                logger.info(f"Found {len(docs)} existing Metro PDFs for KW{week}")
                return docs

        # Try Playwright scraping
        try:
            return self._scrape_with_playwright(week, year, data_dir)
        except Exception as e:
            logger.warning(f"Playwright scraping failed: {e}")

        # Fallback: check for manually placed PDFs in data/metro/
        manual_dir = Path(self.storage_base) / "metro"
        if manual_dir.exists():
            pdfs = self._dedupe_pdf_files(list(manual_dir.glob("*.pdf")))
            for pdf in pdfs:
                meta = self._parse_filename_dates(pdf.name)
                if meta.get("valid_from") and meta.get("valid_to"):
                    return [AcquiredDocument(
                        supplier="metro",
                        location=self.config.get("location", "goslar"),
                        file_path=str(pdf),
                        valid_from=meta.get("valid_from"),
                        valid_to=meta.get("valid_to"),
                        calendar_week=week,
                        year=year,
                    )]

        logger.error("No Metro PDFs found. Please place PDF manually in data/metro/")
        return []

    def _scrape_with_playwright(self, week: int, year: int, data_dir: Path) -> list[AcquiredDocument]:
        from playwright.sync_api import sync_playwright

        data_dir.mkdir(parents=True, exist_ok=True)
        market_url = self.config.get("url", "https://www.metro.de/standorte/goslar")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="de-DE",
            )
            page = context.new_page()

            try:
                offers_url = self._open_offers_overview(page, market_url)
                viewer_urls = self._collect_offer_viewers(page)

                logger.info(f"Metro: Found {len(viewer_urls)} viewer links on {offers_url}")

                docs = []
                seen_files: set[str] = set()

                for viewer_url in viewer_urls:
                    meta = self._parse_filename_dates(viewer_url)
                    if not self._matches_requested_week(meta, week, year):
                        logger.debug(f"Skipping Metro viewer outside requested week: {viewer_url}")
                        continue

                    dest = self._download_viewer_pdf(context, viewer_url, data_dir)
                    if not dest:
                        logger.warning(f"Metro: Could not resolve PDF from viewer {viewer_url}")
                        continue
                    if str(dest) in seen_files:
                        continue

                    seen_files.add(str(dest))
                    logger.info(f"Downloaded Metro PDF: {dest.name}")
                    docs.append(AcquiredDocument(
                        supplier="metro",
                        location=self.config.get("location", "goslar"),
                        file_path=str(dest),
                        valid_from=meta.get("valid_from"),
                        valid_to=meta.get("valid_to"),
                        calendar_week=week,
                        year=year,
                    ))

                return self._dedupe_documents(docs)

            finally:
                browser.close()

    def _open_offers_overview(self, page, market_url: str) -> str:
        offers_url = "https://www.metro.de/angebote"

        try:
            self._goto_page(page, market_url, timeout=90000)
            self._accept_cookies(page)
        except Exception as e:
            logger.warning(f"Metro: market page load failed, falling back to offers overview: {e}")
            self._goto_page(page, offers_url, timeout=90000)
            self._accept_cookies(page)
            return offers_url

        try:
            page.get_by_text("Angebote im Markt", exact=False).first.wait_for(timeout=10000)
        except Exception:
            logger.debug("Metro market page anchor 'Angebote im Markt' not found")

        for label in ("Aktuelle", "Zukünftige"):
            try:
                page.get_by_text(label, exact=True).first.wait_for(timeout=3000)
            except Exception:
                logger.debug(f"Metro market page anchor '{label}' not found")

        offers_link = page.get_by_role("link", name="Zu den Angeboten").first
        try:
            offers_link.wait_for(timeout=10000)
            href = offers_link.get_attribute("href")
            if href:
                offers_url = urljoin(page.url, href)
        except Exception:
            logger.warning("Metro: 'Zu den Angeboten' link not found, falling back to /angebote")

        self._goto_page(page, offers_url)
        self._accept_cookies(page)

        try:
            page.get_by_text("Aktuelle angebote - AB jetzt immer digital", exact=False).first.wait_for(timeout=10000)
        except Exception:
            logger.debug("Metro offers page title anchor not found")

        try:
            offers_tab = page.get_by_text("Angebote aktuell & zukünftig", exact=True).first
            offers_tab.click(timeout=5000)
            page.wait_for_timeout(1500)
        except Exception:
            logger.debug("Metro offers tab 'Angebote aktuell & zukünftig' was not clickable")

        return offers_url

    def _collect_offer_viewers(self, page) -> list[str]:
        try:
            page.wait_for_function(
                "() => Array.from(document.querySelectorAll(\"a[href]\"))"
                ".some(a => a.href.includes('prospekte.metro.de'))",
                timeout=15000,
            )
        except Exception:
            logger.debug("Metro: No prospekte links became visible within timeout")

        hrefs = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(el => el.href)",
        )

        viewers = []
        seen = set()
        for href in hrefs:
            if not href or "prospekte.metro.de" not in href:
                continue
            normalized = self._normalize_viewer_url(href)
            if normalized not in seen:
                seen.add(normalized)
                viewers.append(normalized)

        return viewers

    def _download_viewer_pdf(self, context, viewer_url: str, data_dir: Path) -> Path | None:
        viewer_page = context.new_page()
        try:
            self._goto_page(viewer_page, viewer_url)
            self._accept_cookies(viewer_page)
            viewer_page.wait_for_timeout(1500)

            direct_pdf_url = self._extract_pdf_url_from_viewer_html(viewer_page)
            if direct_pdf_url:
                dest = self._download_pdf_by_url(context, direct_pdf_url, data_dir, viewer_url)
                if dest:
                    return dest

            pdf_link = self._find_pdf_link(viewer_page)
            if pdf_link:
                dest = self._click_and_save_pdf(viewer_page, pdf_link, data_dir, viewer_url)
                if dest:
                    return dest

            try:
                download_trigger = viewer_page.get_by_text("PDF herunterladen", exact=False).first
                download_trigger.wait_for(timeout=5000)
                return self._click_and_save_pdf(viewer_page, download_trigger, data_dir, viewer_url)
            except Exception:
                return None
        finally:
            viewer_page.close()

    def _extract_pdf_url_from_viewer_html(self, page) -> str | None:
        html = page.content()
        patterns = [
            r'"downloadPdfUrl":"([^"]+\.pdf[^"]*)"',
            r'"downloadPdfUrl":"([^"]+)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if not match:
                continue
            candidate = match.group(1)
            try:
                resolved = json.loads(f'"{candidate}"')
            except Exception:
                resolved = candidate.replace("\\/", "/")
            return urljoin(page.url, resolved)
        return None

    def _download_pdf_by_url(self, context, pdf_url: str, data_dir: Path, viewer_url: str) -> Path | None:
        try:
            response = context.request.get(pdf_url, timeout=60000)
            if response.status != 200:
                return None
            content = response.body()
            if not content.startswith(b"%PDF-"):
                return None
            filename = self._build_download_name(viewer_url, Path(urlparse(pdf_url).path).name or "metro.pdf")
            dest = data_dir / filename
            dest.write_bytes(content)
            validate_pdf(str(dest))
            return dest
        except Exception as e:
            logger.debug(f"Metro direct PDF fetch failed for {viewer_url}: {e}")
            return None

    def _click_and_save_pdf(self, page, locator, data_dir: Path, viewer_url: str) -> Path | None:
        try:
            with page.expect_download(timeout=15000) as download_info:
                locator.click()
            download = download_info.value
        except Exception as e:
            logger.debug(f"Metro PDF download click failed for {viewer_url}: {e}")
            return None

        suggested = download.suggested_filename or "metro.pdf"
        filename = self._build_download_name(viewer_url, suggested)
        dest = data_dir / filename
        download.save_as(str(dest))
        try:
            validate_pdf(str(dest))
        except Exception as e:
            if dest.exists():
                dest.unlink()
            logger.warning(f"Discarding invalid Metro PDF {filename}: {e}")
            return None
        return dest

    def _find_pdf_link(self, page):
        for selector in ("a[href*='/pdfs/']", "a[href*='.pdf']"):
            locator = page.locator(selector)
            if locator.count() > 0:
                return locator.first

        return None

    def _accept_cookies(self, page):
        for label in ("Alle akzeptieren", "Akzeptieren", "Zustimmen"):
            try:
                page.get_by_role("button", name=label).first.click(timeout=3000)
                page.wait_for_timeout(500)
                return
            except Exception:
                continue

    def _goto_page(self, page, url: str, timeout: int = 90000):
        page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)

    def _normalize_viewer_url(self, href: str) -> str:
        parsed = urlparse(href)
        path = parsed.path.rstrip("/")

        if self._is_pdf_url(href):
            return href
        if "/page/" not in path:
            path = f"{path}/page/1"

        return parsed._replace(path=path, params="", query="", fragment="").geturl()

    def _is_pdf_url(self, href: str) -> bool:
        href_lower = href.lower()
        path = urlparse(href_lower).path
        return "/pdfs/" in path or path.endswith(".pdf")

    def _build_download_name(self, viewer_url: str, suggested: str) -> str:
        slug = urlparse(viewer_url).path.strip("/").split("/")[0] or Path(suggested).stem
        safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-") or "metro"
        return f"{safe_slug}.pdf"

    def _matches_requested_week(self, meta: dict, week: int, year: int) -> bool:
        valid_from = meta.get("valid_from")
        valid_to = meta.get("valid_to")
        if not valid_from or not valid_to:
            return True

        week_start, week_end = week_date_range(week, year)
        return valid_from <= week_end and valid_to >= week_start

    def _parse_filename_dates(self, filename: str) -> dict:
        # Try to extract dates from filename patterns
        date_pattern = re.compile(r'(\d{2})\.(\d{2})\.?\s*[-–]\s*(\d{2})\.(\d{2})\.?(\d{4})?')
        match = date_pattern.search(filename)
        if match:
            day1, mon1, day2, mon2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            yr = int(match.group(5)) if match.group(5) else date.today().year
            try:
                return {"valid_from": date(yr, mon1, day1), "valid_to": date(yr, mon2, day2)}
            except ValueError:
                pass

        compact_pattern = re.compile(r'(\d{2})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})')
        match = compact_pattern.search(filename)
        if match:
            day1, mon1, yr1, day2, mon2, yr2 = map(int, match.groups())
            try:
                return {
                    "valid_from": date(2000 + yr1, mon1, day1),
                    "valid_to": date(2000 + yr2, mon2, day2),
                }
            except ValueError:
                pass
        return {}

    def _dedupe_documents(self, documents: list[AcquiredDocument]) -> list[AcquiredDocument]:
        deduped: list[AcquiredDocument] = []
        seen_hashes: dict[str, AcquiredDocument] = {}
        for document in documents:
            if not document.file_path:
                deduped.append(document)
                continue

            path = Path(document.file_path)
            if not path.exists():
                deduped.append(document)
                continue

            digest = self._file_sha256(path)
            if digest not in seen_hashes:
                seen_hashes[digest] = document
                deduped.append(document)
                continue

            path.unlink()
            logger.info("Removed duplicate Metro PDF: %s", path.name)

        return deduped

    def _dedupe_pdf_files(self, pdfs: list[Path]) -> list[Path]:
        deduped: list[Path] = []
        seen_hashes: set[str] = set()
        for pdf in sorted(pdfs):
            digest = self._file_sha256(pdf)
            if digest in seen_hashes:
                pdf.unlink()
                logger.info("Removed duplicate Metro PDF: %s", pdf.name)
                continue
            seen_hashes.add(digest)
            deduped.append(pdf)
        return deduped

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def extract_products(self, document: AcquiredDocument) -> list[RawProduct]:
        if not document.file_path:
            return []

        year_part = str(document.year or "unknown")
        week_part = f"KW{document.calendar_week:02d}" if document.calendar_week is not None else "KW00"
        images_dir = Path("images") / "metro" / year_part / week_part / Path(document.file_path).stem
        image_paths = pdf_to_images(document.file_path, str(images_dir))
        return extract_products_from_pdf_images(
            image_paths,
            supplier="metro",
            source_file=document.file_path,
            valid_from=document.valid_from,
            valid_to=document.valid_to,
            calendar_week=document.calendar_week,
            year=document.year,
            location=document.location,
            source_title=document.title,
            source_tab=document.tab,
        )
