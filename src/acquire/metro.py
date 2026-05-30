import logging
import re
import json
import hashlib
from io import BytesIO
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
                dest = self._click_and_save_pdf(viewer_page, download_trigger, data_dir, viewer_url)
                if dest:
                    return dest
            except Exception:
                pass

            return self._recover_publitas_viewer_as_pdf(
                context,
                viewer_url,
                data_dir,
                viewer_html=viewer_page.content(),
            )
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
        temp_dest = None
        try:
            response = context.request.get(pdf_url, timeout=60000)
            if response.status != 200:
                return None
            content = response.body()
            if not content.startswith(b"%PDF-"):
                return None
            filename = self._build_download_name(viewer_url, Path(urlparse(pdf_url).path).name or "metro.pdf")
            dest = data_dir / filename
            temp_dest = data_dir / f".{filename}.download.pdf"
            temp_dest.write_bytes(content)
            validate_pdf(str(temp_dest))
            temp_dest.replace(dest)
            return dest
        except Exception as e:
            if temp_dest and temp_dest.exists():
                temp_dest.unlink()
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
        temp_dest = data_dir / f".{filename}.download.pdf"
        download.save_as(str(temp_dest))
        try:
            validate_pdf(str(temp_dest))
        except Exception as e:
            if temp_dest.exists():
                temp_dest.unlink()
            logger.warning(f"Discarding invalid Metro PDF {filename}: {e}")
            return None
        temp_dest.replace(dest)
        return dest

    def _recover_publitas_viewer_as_pdf(
        self,
        context,
        viewer_url: str,
        data_dir: Path,
        max_pages: int = 180,
        viewer_html: str | None = None,
    ) -> Path | None:
        if "prospekte.metro.de" not in viewer_url:
            return None

        try:
            from PIL import Image
        except Exception as e:
            logger.debug("Metro Publitas image-PDF fallback unavailable: Pillow missing: %s", e)
            return None

        page_image_urls = self._publitas_spread_image_urls(context, viewer_url, viewer_html, max_pages)
        if not page_image_urls:
            page_image_urls = self._publitas_visible_page_image_urls(context, viewer_url, max_pages)

        if not page_image_urls:
            return None

        pil_images = []
        try:
            for image_url in page_image_urls:
                image_response = context.request.get(image_url, timeout=60000)
                if image_response.status != 200:
                    raise RuntimeError(f"image fetch returned HTTP {image_response.status}: {image_url}")
                pil_images.append(Image.open(BytesIO(image_response.body())).convert("RGB"))

            filename = self._build_download_name(viewer_url, "metro-publitas.pdf")
            dest = data_dir / filename
            temp_dest = data_dir / f".{filename}.download.pdf"
            pil_images[0].save(
                temp_dest,
                "PDF",
                save_all=True,
                append_images=pil_images[1:],
                resolution=150.0,
            )
            validate_pdf(str(temp_dest))
            temp_dest.replace(dest)
            logger.info(
                "Recovered Metro Publitas viewer as image PDF: %s (%d pages)",
                dest.name,
                len(page_image_urls),
            )
            return dest
        except Exception as e:
            temp_dest = data_dir / f".{self._build_download_name(viewer_url, 'metro-publitas.pdf')}.download.pdf"
            if temp_dest.exists():
                temp_dest.unlink()
            logger.warning("Metro Publitas image-PDF fallback failed for %s: %s", viewer_url, e)
            return None
        finally:
            for image in pil_images:
                try:
                    image.close()
                except Exception:
                    pass

    def _publitas_spread_image_urls(
        self,
        context,
        viewer_url: str,
        viewer_html: str | None,
        max_pages: int,
    ) -> list[str]:
        html = viewer_html
        if not html:
            try:
                response = context.request.get(viewer_url, timeout=60000)
                if response.status == 200:
                    html = response.text()
            except Exception as e:
                logger.debug("Metro Publitas metadata page fetch failed for %s: %s", viewer_url, e)
                return []

        data = self._extract_publitas_initial_data(html or "")
        if not data:
            return []

        base_url = data.get("url") or data.get("config", {}).get("canonicalUrl") or viewer_url
        cache_token = data.get("cacheToken")
        spreads_url = urljoin(base_url.rstrip("/") + "/", "spreads.json")
        if cache_token:
            spreads_url = f"{spreads_url}?version={cache_token}"

        spreads = []
        page_number: int | None = 1
        while page_number is not None:
            separator = "&" if "?" in spreads_url else "?"
            paged_spreads_url = f"{spreads_url}{separator}page={page_number}"
            try:
                response = context.request.get(paged_spreads_url, timeout=60000)
            except Exception as e:
                logger.debug("Metro Publitas spreads fetch failed for %s: %s", paged_spreads_url, e)
                return []
            if response.status != 200:
                logger.debug(
                    "Metro Publitas spreads fetch returned HTTP %s for %s",
                    response.status,
                    paged_spreads_url,
                )
                return []

            try:
                page_spreads = response.json()
            except Exception as e:
                logger.debug("Metro Publitas spreads JSON parse failed for %s: %s", paged_spreads_url, e)
                return []
            spreads.extend(page_spreads or [])

            next_page = response.headers.get("x-next-page")
            page_number = int(next_page) if next_page and next_page.isdigit() else None

        pages = [
            page
            for spread in spreads or []
            for page in (spread.get("pages", []) or [])
        ]
        if len(pages) > 120:
            preferred_image_sizes = ("at800", "at1000", "at1200", "at1600")
        elif len(pages) > 80:
            preferred_image_sizes = ("at1000", "at1200", "at1600", "at800")
        else:
            preferred_image_sizes = ("at1600", "at1200", "at1000", "at800")

        page_urls: list[str] = []
        seen: set[str] = set()
        for page in pages:
            images = page.get("images", {}) or {}
            image_path = next((images.get(size) for size in preferred_image_sizes if images.get(size)), None)
            if not image_path:
                continue
            image_url = urljoin("https://view.publitas.com", image_path)
            if image_url in seen:
                continue
            seen.add(image_url)
            page_urls.append(image_url)
            if len(page_urls) >= max_pages:
                logger.warning(
                    "Metro Publitas fallback capped %s at %d pages",
                    viewer_url,
                    max_pages,
                )
                return page_urls

        return page_urls

    def _publitas_visible_page_image_urls(self, context, viewer_url: str, max_pages: int) -> list[str]:
        page_image_urls: list[str] = []
        seen_images: set[str] = set()
        for page_number in range(1, max_pages + 1):
            page_url = self._publitas_page_url(viewer_url, page_number)
            try:
                response = context.request.get(page_url, timeout=60000)
            except Exception as e:
                logger.debug("Metro Publitas page fetch failed for %s: %s", page_url, e)
                break
            if response.status != 200:
                break

            image_url = self._extract_publitas_page_image_url(response.text(), page_url)
            if not image_url or image_url in seen_images:
                break
            seen_images.add(image_url)
            page_image_urls.append(image_url)

        return page_image_urls

    def _extract_publitas_initial_data(self, html: str) -> dict | None:
        match = re.search(
            r"var\s+data\s*=\s*(\{.*?\})\s*;\s*Reader\.Bootstrap\.init",
            html,
            flags=re.DOTALL,
        )
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except Exception as e:
            logger.debug("Metro Publitas initial data parse failed: %s", e)
            return None

    def _publitas_page_url(self, viewer_url: str, page_number: int) -> str:
        parsed = urlparse(viewer_url)
        slug = parsed.path.strip("/").split("/")[0]
        if not slug:
            return viewer_url
        return parsed._replace(
            path=f"/{slug}/page/{page_number}",
            params="",
            query="",
            fragment="",
        ).geturl()

    def _extract_publitas_page_image_url(self, html: str, page_url: str) -> str | None:
        patterns = [
            r'https:\\/\\/view\.publitas\.com\\/[^"\'< ]+?\\/pages\\/[^"\'< ]+?-at1600\.jpg',
            r'https://view\.publitas\.com/[^"\'< ]+?/pages/[^"\'< ]+?-at1600\.jpg',
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if not match:
                continue
            candidate = match.group(0).replace("\\/", "/")
            return urljoin(page_url, candidate)
        return None

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
