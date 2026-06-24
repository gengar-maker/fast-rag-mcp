from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from .config import Settings
from .io import document_id
from .types import Document

log = logging.getLogger(__name__)


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _normalize_block_text(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines if line.strip())
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _header_key(text: str) -> str:
    value = re.sub(r"\d+", "#", text.lower())
    value = re.sub(r"\s+", " ", value).strip()
    return value[:180]


def _page_dict(page: Any, settings: Settings) -> dict[str, Any]:
    used_ocr = False
    ocr_error = ""
    text_page = None
    data = page.get_text("dict", sort=True)
    plain = page.get_text("text", sort=True).strip()
    has_images = False
    if settings.pdf_ocr_mode == "auto" and len(plain) < settings.pdf_ocr_min_chars:
        try:
            has_images = bool(page.get_images(full=True))
        except Exception:
            has_images = True
    if settings.pdf_ocr_mode != "off" and (
        settings.pdf_ocr_mode == "always"
        or (len(plain) < settings.pdf_ocr_min_chars and has_images)
    ):
        try:
            text_page = page.get_textpage_ocr(
                language=settings.pdf_ocr_language,
                dpi=settings.pdf_ocr_dpi,
                full=settings.pdf_ocr_mode == "always",
            )
            data = page.get_text("dict", textpage=text_page, sort=True)
            used_ocr = True
        except Exception as exc:  # OCR is optional and must not break text PDFs.
            ocr_error = f"{type(exc).__name__}: {exc}"
            log.warning("PDF OCR failed on page %s: %s", page.number + 1, ocr_error)

    blocks: list[dict[str, Any]] = []
    for raw in data.get("blocks", []):
        if raw.get("type", 0) != 0 or "lines" not in raw:
            continue
        line_texts: list[str] = []
        sizes: list[float] = []
        bold_chars = 0
        mono_chars = 0
        total_chars = 0
        for line in raw.get("lines", []):
            pieces: list[str] = []
            for span in line.get("spans", []):
                value = str(span.get("text", ""))
                if not value:
                    continue
                pieces.append(value)
                chars = len(value.strip())
                total_chars += chars
                size = float(span.get("size", 0.0) or 0.0)
                if size:
                    sizes.extend([size] * max(1, min(chars, 32)))
                flags = int(span.get("flags", 0) or 0)
                font = str(span.get("font", "")).lower()
                if flags & (1 << 4) or "bold" in font:
                    bold_chars += chars
                if flags & (1 << 3) or any(x in font for x in ("mono", "courier", "code")):
                    mono_chars += chars
            line_value = "".join(pieces).strip()
            if line_value:
                line_texts.append(line_value)
        text = _normalize_block_text(line_texts)
        if not text:
            continue
        bbox = [round(float(v), 2) for v in raw.get("bbox", (0, 0, 0, 0))]
        blocks.append(
            {
                "text": text,
                "bbox": bbox,
                "font_size": round(max(sizes) if sizes else 0.0, 2),
                "font_median": round(median(sizes) if sizes else 0.0, 2),
                "bold_ratio": bold_chars / max(1, total_chars),
                "mono_ratio": mono_chars / max(1, total_chars),
            }
        )
    try:
        label = page.get_label() or str(page.number + 1)
    except Exception:
        label = str(page.number + 1)
    rect = page.rect
    return {
        "number": page.number + 1,
        "label": label,
        "width": round(float(rect.width), 2),
        "height": round(float(rect.height), 2),
        "ocr": used_ocr,
        "ocr_error": ocr_error,
        "blocks": blocks,
    }


def _remove_repeated_margins(pages: list[dict[str, Any]]) -> None:
    if len(pages) < 2:
        return
    counts: Counter[str] = Counter()
    for page in pages:
        height = float(page.get("height", 1.0) or 1.0)
        seen: set[str] = set()
        for block in page["blocks"]:
            y0, y1 = float(block["bbox"][1]), float(block["bbox"][3])
            text = block["text"]
            if len(text) <= 220 and (y0 <= height * 0.09 or y1 >= height * 0.91):
                key = _header_key(text)
                if key:
                    seen.add(key)
        counts.update(seen)
    threshold = max(2, int(len(pages) * 0.55 + 0.999))
    repeated = {key for key, count in counts.items() if count >= threshold}
    if not repeated:
        return
    for page in pages:
        height = float(page.get("height", 1.0) or 1.0)
        kept = []
        for block in page["blocks"]:
            y0, y1 = float(block["bbox"][1]), float(block["bbox"][3])
            is_margin = y0 <= height * 0.09 or y1 >= height * 0.91
            if is_margin and _header_key(block["text"]) in repeated:
                continue
            kept.append(block)
        page["blocks"] = kept


def _classify_blocks(pages: list[dict[str, Any]]) -> None:
    weighted_sizes: list[float] = []
    for page in pages:
        for block in page["blocks"]:
            size = float(block.get("font_median", 0.0) or 0.0)
            if size:
                weighted_sizes.extend([size] * max(1, min(len(block["text"]) // 20, 50)))
    body_size = median(weighted_sizes) if weighted_sizes else 10.0
    for page in pages:
        for block in page["blocks"]:
            text = block["text"].strip()
            size = float(block.get("font_size", 0.0) or 0.0)
            short = len(text) <= 180 and text.count("\n") <= 2
            heading = short and (
                size >= body_size * 1.18
                or (float(block.get("bold_ratio", 0.0)) >= 0.65 and size >= body_size * 0.98)
            )
            if heading:
                kind = "heading"
            elif float(block.get("mono_ratio", 0.0)) >= 0.55:
                kind = "code"
            else:
                kind = "text"
            block["kind"] = kind


def _extract_tables(page: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        finder = page.find_tables()
        for index, table in enumerate(finder.tables):
            rows = table.extract()
            clean_rows = [
                ["" if cell is None else str(cell).strip() for cell in row] for row in rows
            ]
            clean_rows = [row for row in clean_rows if any(row)]
            if not clean_rows:
                continue
            width = max(len(row) for row in clean_rows)
            clean_rows = [row + [""] * (width - len(row)) for row in clean_rows]
            header = clean_rows[0]
            lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
            lines.extend("| " + " | ".join(row) + " |" for row in clean_rows[1:])
            out.append(
                {
                    "text": "\n".join(lines),
                    "bbox": [round(float(v), 2) for v in table.bbox],
                    "kind": "table",
                    "table_index": index,
                    "font_size": 0.0,
                    "font_median": 0.0,
                    "bold_ratio": 0.0,
                    "mono_ratio": 0.0,
                }
            )
    except Exception as exc:
        log.debug("PDF table extraction failed: %s", exc)
    return out


def read_pdf_document(root: Path, path: Path, settings: Settings) -> Document:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF indexing requires PyMuPDF; install with `uv sync --extra documents` "
            "or `uv sync --extra mac-metal --extra code --extra documents`"
        ) from exc

    stat = path.stat()
    if stat.st_size > settings.max_pdf_bytes:
        raise ValueError(
            f"PDF exceeds BRAG_MAX_PDF_BYTES: {stat.st_size} > {settings.max_pdf_bytes}"
        )
    pages: list[dict[str, Any]] = []
    outline: list[dict[str, Any]] = []
    source_page_count = 0
    truncated_pages = False
    with pymupdf.open(path) as pdf:
        source_page_count = int(pdf.page_count)
        if pdf.needs_pass and not pdf.authenticate(settings.pdf_password):
            raise PermissionError("encrypted PDF requires BRAG_PDF_PASSWORD")
        page_count = min(pdf.page_count, settings.pdf_max_pages)
        truncated_pages = pdf.page_count > page_count
        try:
            for level, title, page_number in pdf.get_toc(simple=True):
                if page_number <= page_count:
                    outline.append(
                        {"level": int(level), "title": str(title), "page": int(page_number)}
                    )
        except Exception as exc:
            log.debug("PDF outline extraction failed: %s", exc)
        total_chars = 0
        for page_number in range(page_count):
            page = pdf.load_page(page_number)
            page_data = _page_dict(page, settings)
            if settings.pdf_extract_tables:
                page_data["blocks"].extend(_extract_tables(page))
                page_data["blocks"].sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            total_chars += sum(len(block["text"]) for block in page_data["blocks"])
            if total_chars > settings.pdf_max_chars:
                page_data["truncated"] = True
                pages.append(page_data)
                break
            pages.append(page_data)
        metadata = dict(pdf.metadata or {})

    _remove_repeated_margins(pages)
    _classify_blocks(pages)
    parts: list[str] = []
    offset = 0
    for page in pages:
        page["start"] = offset
        page_text = "\n\n".join(block["text"] for block in page["blocks"] if block["text"])
        parts.append(page_text)
        offset += len(page_text)
        page["end"] = offset
        offset += 2
    text = "\n\n".join(parts)
    relative = path.resolve(strict=False).relative_to(root.resolve()).as_posix()
    title = str(metadata.get("title") or path.name).strip()
    return Document(
        doc_id=document_id(root, path),
        root=root,
        path=path,
        rel_path=relative,
        title=title,
        text=text,
        content_hash=_file_sha256(path),
        mtime_ns=stat.st_mtime_ns,
        size_bytes=stat.st_size,
        metadata={
            "document_type": "pdf",
            "page_count": len(pages),
            "source_page_count": source_page_count,
            "truncated_pages": truncated_pages,
            "pages": pages,
            "outline": outline,
            "pdf_metadata": metadata,
            "extractor": "pymupdf",
        },
    )
