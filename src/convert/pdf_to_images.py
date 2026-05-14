import logging
import subprocess
import time
from pathlib import Path

from src.utils.logging_setup import log_event

logger = logging.getLogger("birkenhof.convert")


def validate_pdf(pdf_path: str, min_pages: int = 1) -> int:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise RuntimeError(f"PDF not found: {pdf}")
    if pdf.stat().st_size < 1000:
        raise RuntimeError(f"PDF too small or empty: {pdf.name}")
    if pdf.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"Invalid PDF header: {pdf.name}")

    result = subprocess.run(
        ["pdfinfo", str(pdf)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"pdfinfo failed for {pdf.name}")

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":")[1].strip())
            if pages < min_pages:
                raise RuntimeError(f"PDF has {pages} pages: {pdf.name}")
            return pages

    raise RuntimeError(f"Could not determine page count for {pdf.name}")


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    fmt: str = "png",
) -> list[str]:
    pdf = Path(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate_pdf(str(pdf))
    started = time.perf_counter()

    prefix = out / "page"
    existing = sorted(out.glob(f"page-*.{fmt}"))
    if existing:
        logger.info(f"Images already exist for {pdf.name}, skipping conversion")
        log_event(
            logger,
            f"Reused existing PDF page images for {pdf.name}",
            event="pdf_to_images",
            status="cached",
            source_file=str(pdf),
            output_dir=str(out),
            image_count=len(existing),
        )
        return [str(p) for p in existing]

    logger.info(f"Converting {pdf.name} to images at {dpi} DPI...")
    log_event(
        logger,
        f"PDF to images started for {pdf.name}",
        event="pdf_to_images",
        status="start",
        source_file=str(pdf),
        output_dir=str(out),
        dpi=dpi,
        format=fmt,
    )
    cmd = [
        "pdftoppm",
        "-r", str(dpi),
        f"-{fmt}",
        str(pdf),
        str(prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"pdftoppm failed: {result.stderr}")
        log_event(
            logger,
            f"PDF to images failed for {pdf.name}",
            event="pdf_to_images",
            level=logging.ERROR,
            status="error",
            source_file=str(pdf),
            output_dir=str(out),
            error_message=result.stderr.strip(),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise RuntimeError(f"PDF conversion failed: {result.stderr}")

    images = sorted(out.glob(f"page-*.{fmt}"))
    logger.info(f"Created {len(images)} images from {pdf.name}")
    log_event(
        logger,
        f"PDF to images completed for {pdf.name}",
        event="pdf_to_images",
        status="ok",
        source_file=str(pdf),
        output_dir=str(out),
        image_count=len(images),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return [str(p) for p in images]


def pdf_first_page_to_image(
    pdf_path: str,
    output_dir: str,
    dpi: int = 180,
    fmt: str = "png",
) -> str:
    pdf = Path(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate_pdf(str(pdf))
    started = time.perf_counter()

    existing = sorted(out.glob(f"preview-*.{fmt}"))
    if existing:
        log_event(
            logger,
            f"Reused relevance preview for {pdf.name}",
            event="pdf_first_page_preview",
            status="cached",
            source_file=str(pdf),
            preview_path=str(existing[0]),
        )
        return str(existing[0])

    prefix = out / "preview"
    cmd = [
        "pdftoppm",
        "-f", "1",
        "-l", "1",
        "-r", str(dpi),
        f"-{fmt}",
        str(pdf),
        str(prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"pdftoppm first-page preview failed: {result.stderr}")
        log_event(
            logger,
            f"PDF first-page preview failed for {pdf.name}",
            event="pdf_first_page_preview",
            level=logging.ERROR,
            status="error",
            source_file=str(pdf),
            output_dir=str(out),
            error_message=result.stderr.strip(),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise RuntimeError(f"PDF first page conversion failed: {result.stderr}")

    images = sorted(out.glob(f"preview-*.{fmt}"))
    if not images:
        raise RuntimeError(f"No first-page preview created for {pdf.name}")
    log_event(
        logger,
        f"PDF first-page preview created for {pdf.name}",
        event="pdf_first_page_preview",
        status="ok",
        source_file=str(pdf),
        preview_path=str(images[0]),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return str(images[0])


def pdf_page_count(pdf_path: str) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    return 0


def pdf_to_text(pdf_path: str) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.warning(f"pdftotext failed for {pdf_path}: {result.stderr}")
        return ""
    return result.stdout


def pdf_page_to_text(pdf_path: str, page: int) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
        capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else ""
