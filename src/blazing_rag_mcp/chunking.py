from __future__ import annotations

import hashlib
import re

from .code_index import detect_language, extract_imports, extract_symbols
from .config import Settings
from .io import stable_id
from .types import Chunk, CodeSymbol, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SYMBOL_RE = re.compile(
    r"^\s*(class|def|async\s+def|function|export\s+function|export\s+class|interface|type|struct|enum|impl|fn)\s+([A-Za-z0-9_.$:-]+)",
    re.MULTILINE,
)


def _chunk_id(
    doc: Document,
    *,
    text: str,
    chunk_type: str,
    anchor: str = "",
    part: str = "",
    occurrence: int = 0,
) -> str:
    """Build a content-stable chunk ID.

    Unrelated line insertions no longer invalidate symbol/chunk URIs. The content digest ensures a
    changed chunk gets a new ID, while the semantic anchor and occurrence disambiguate identical
    chunks in one document.
    """
    digest = hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).hexdigest()
    return stable_id(
        "chunk-v2",
        doc.doc_id,
        chunk_type,
        anchor,
        part,
        digest,
        str(occurrence),
        n=24,
    )


def _cheap_token_count(text: str) -> int:
    # Fast approximation good enough for chunk sizing without a tokenizer dependency.
    return max(1, len(text) // 4)


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for m in re.finditer("\n", text):
        offsets.append(m.end())
    return offsets


def _line_for_offset(offsets: list[int], pos: int) -> int:
    # Binary search implemented manually to avoid importing bisect in tight loops.
    lo, hi = 0, len(offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)


def _offset_for_line(offsets: list[int], line: int) -> int:
    if line <= 1:
        return 0
    idx = min(line - 1, len(offsets) - 1)
    return offsets[idx]


def _section_for_window(text: str, start: int) -> str:
    prefix = text[:start]
    section = ""
    for line in reversed(prefix.splitlines()[-120:]):
        m = HEADING_RE.match(line)
        if m:
            section = m.group(2).strip()
            break
    if section:
        return section[:160]

    # For code, closest preceding symbol gives much better provenance.
    last = None
    for m in SYMBOL_RE.finditer(prefix[-50_000:]):
        last = m
    if last:
        return last.group(0).strip()[:160]
    return ""


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"\n\s*\n", text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    if not spans and text.strip():
        spans.append((0, len(text)))
    return spans


def chunk_document(
    doc: Document, settings: Settings, symbols: list[CodeSymbol] | None = None
) -> list[Chunk]:
    text = doc.text
    if not text.strip():
        return []
    if doc.metadata.get("document_type") == "pdf":
        return _chunk_pdf_document(doc, settings)
    language = detect_language(doc.rel_path)
    if language:
        syms = symbols if symbols is not None else extract_symbols(doc)
        code_chunks = _chunk_code_document(doc, settings, language, syms)
        if code_chunks:
            return code_chunks
    return _chunk_text_document(doc, settings, language=language)


def _chunk_code_document(
    doc: Document, settings: Settings, language: str, symbols: list[CodeSymbol]
) -> list[Chunk]:
    offsets = _line_offsets(doc.text)
    chunks: list[Chunk] = []

    if getattr(settings, "add_file_summary_chunks", True):
        summary = _file_summary_text(doc, language, symbols)
        if summary:
            chunks.append(
                _make_chunk(
                    doc,
                    text=summary,
                    section="file summary",
                    start=0,
                    end=min(len(doc.text), max(1, len(summary))),
                    line_start=1,
                    line_end=min(max(1, len(doc.text.splitlines())), 120),
                    chunk_type="file_summary",
                    language=language,
                    extra={"symbol_count": len(symbols), "imports": extract_imports(doc.text)[:80]},
                )
            )

    imports_text = _import_block(doc.text)
    if imports_text and len(imports_text) >= 20:
        imp_end = min(len(doc.text), len(imports_text))
        chunks.append(
            _make_chunk(
                doc,
                text=imports_text,
                section="imports",
                start=0,
                end=imp_end,
                line_start=1,
                line_end=_line_for_offset(offsets, imp_end),
                chunk_type="imports",
                language=language,
                extra={"imports": extract_imports(imports_text)},
            )
        )

    max_chars = max(1200, settings.chunk_tokens * 4)
    overlap_lines = max(2, min(12, settings.chunk_overlap // 12))
    for sym in symbols:
        sym_text = sym.text.strip()
        if not sym_text:
            continue
        spans = _symbol_line_windows(sym, max_chars=max_chars, overlap_lines=overlap_lines)
        for part_idx, (line_start, line_end, rel_start_char, rel_end_char) in enumerate(spans):
            start = _offset_for_line(offsets, line_start)
            end = (
                _offset_for_line(offsets, line_end + 1)
                if line_end < len(offsets)
                else min(len(doc.text), sym.byte_end)
            )
            chunk_text = doc.text[start:end].strip()
            if not chunk_text:
                chunk_text = sym_text[rel_start_char:rel_end_char].strip()
            section = sym.qualified_name
            chunk_type = "symbol" if len(spans) == 1 else "symbol_part"
            chunks.append(
                _make_chunk(
                    doc,
                    text=chunk_text,
                    section=section,
                    start=start,
                    end=end,
                    line_start=line_start,
                    line_end=line_end,
                    chunk_type=chunk_type,
                    language=language,
                    extra={
                        "symbol_id": sym.id,
                        "symbol_name": sym.name,
                        "qualified_name": sym.qualified_name,
                        "symbol_kind": sym.kind,
                        "signature": sym.signature,
                        "docstring": sym.docstring,
                        "part": part_idx,
                        "parts": len(spans),
                        "parser": sym.metadata.get("parser", ""),
                    },
                )
            )

    if len(chunks) <= 1:
        # Code parser found nothing useful. Use generic chunks but tag them as code.
        return _chunk_text_document(doc, settings, language=language, chunk_type="code_text")
    return _dedupe_chunks(chunks)


def _file_summary_text(doc: Document, language: str, symbols: list[CodeSymbol]) -> str:
    imports = extract_imports(doc.text)[:40]
    top_symbols = [s for s in symbols if not s.parent][:80]
    if not symbols and not imports:
        return ""
    lines = [f"File: {doc.rel_path}", f"Language: {language}"]
    if imports:
        lines.append("Imports: " + ", ".join(imports[:40]))
    if top_symbols:
        lines.append("Top-level symbols:")
        for s in top_symbols:
            sig = f" — {s.signature}" if s.signature else ""
            lines.append(f"- {s.kind} {s.qualified_name}{sig}")
    return "\n".join(lines)


def _import_block(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines[:180]:
        stripped = line.strip()
        if not stripped:
            if kept:
                kept.append(line)
            continue
        if stripped.startswith(("import ", "from ", "use ", "#include", "package ")) or (
            " from " in stripped and ("import" in stripped or "export" in stripped)
        ):
            kept.append(line)
            continue
        if kept and len(kept) < 6 and stripped.startswith(("//", "#", "/*", "*")):
            kept.append(line)
            continue
        if kept:
            break
    return "\n".join(kept).strip()


def _symbol_line_windows(
    sym: CodeSymbol, *, max_chars: int, overlap_lines: int
) -> list[tuple[int, int, int, int]]:
    lines = sym.text.splitlines()
    if len(sym.text) <= max_chars or len(lines) <= 2:
        return [(sym.line_start, sym.line_end, 0, len(sym.text))]
    windows: list[tuple[int, int, int, int]] = []
    start_idx = 0
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)
    while start_idx < len(lines):
        end_idx = start_idx
        chars = 0
        while end_idx < len(lines) and chars < max_chars:
            chars += len(lines[end_idx]) + 1
            end_idx += 1
        if end_idx <= start_idx:
            end_idx = min(len(lines), start_idx + 1)
        rel_start = line_offsets[start_idx]
        rel_end = line_offsets[end_idx]
        windows.append(
            (sym.line_start + start_idx, sym.line_start + end_idx - 1, rel_start, rel_end)
        )
        if end_idx >= len(lines):
            break
        start_idx = max(start_idx + 1, end_idx - overlap_lines)
    return windows


def _make_chunk(
    doc: Document,
    *,
    text: str,
    section: str,
    start: int,
    end: int,
    line_start: int,
    line_end: int,
    chunk_type: str,
    language: str,
    extra: dict | None = None,
) -> Chunk:
    metadata = {
        "mtime_ns": doc.mtime_ns,
        "size_bytes": doc.size_bytes,
        "language": language,
        "chunk_type": chunk_type,
    }
    if extra:
        metadata.update(extra)
    anchor = str((extra or {}).get("symbol_id") or (extra or {}).get("qualified_name") or section)
    part = str((extra or {}).get("part", ""))
    cid = _chunk_id(
        doc,
        text=text,
        chunk_type=chunk_type,
        anchor=anchor,
        part=part,
    )
    return Chunk(
        id=cid,
        doc_id=doc.doc_id,
        root=doc.root.as_posix(),
        path=doc.rel_path,
        title=doc.title,
        section=section[:240],
        text=text.strip(),
        line_start=line_start,
        line_end=line_end,
        byte_start=start,
        byte_end=end,
        content_hash=doc.content_hash,
        metadata=metadata,
    )


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    seen: set[tuple[str, int, int, str]] = set()
    for c in chunks:
        key = (c.path, c.line_start, c.line_end, c.chunk_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _chunk_text_document(
    doc: Document, settings: Settings, *, language: str = "", chunk_type: str = "text"
) -> list[Chunk]:
    text = doc.text
    para_spans = _split_paragraphs(text)
    offsets = _line_offsets(text)
    chunks: list[Chunk] = []
    id_counts: dict[tuple[str, str, str], int] = {}
    target = max(128, settings.chunk_tokens)
    overlap_chars = max(0, settings.chunk_overlap * 4)

    cur_start: int | None = None
    cur_end: int | None = None
    cur_texts: list[str] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_texts, cur_tokens
        if cur_start is None or cur_end is None:
            return
        chunk_text = text[cur_start:cur_end].strip()
        if len(chunk_text) < settings.min_chunk_chars and chunks:
            # Merge tiny trailing chunks into previous chunk in docstore terms by extending text field only.
            prev = chunks.pop()
            merged_text = f"{prev.text}\n\n{chunk_text}".strip()
            metadata = dict(prev.metadata)
            metadata.setdefault("language", language)
            metadata.setdefault("chunk_type", chunk_type)
            merged_digest = hashlib.blake2b(merged_text.encode("utf-8"), digest_size=16).hexdigest()
            identity_key = (chunk_type, prev.section, merged_digest)
            occurrence = id_counts.get(identity_key, 0)
            id_counts[identity_key] = occurrence + 1
            chunks.append(
                Chunk(
                    id=_chunk_id(
                        doc,
                        text=merged_text,
                        chunk_type=chunk_type,
                        anchor=prev.section,
                        occurrence=occurrence,
                    ),
                    doc_id=prev.doc_id,
                    root=prev.root,
                    path=prev.path,
                    title=prev.title,
                    section=prev.section,
                    text=merged_text,
                    line_start=prev.line_start,
                    line_end=_line_for_offset(offsets, cur_end),
                    byte_start=prev.byte_start,
                    byte_end=cur_end,
                    content_hash=prev.content_hash,
                    metadata=metadata,
                )
            )
        elif chunk_text:
            line_start = _line_for_offset(offsets, cur_start)
            line_end = _line_for_offset(offsets, cur_end)
            section = _section_for_window(text, cur_start)
            digest = hashlib.blake2b(chunk_text.encode("utf-8"), digest_size=16).hexdigest()
            identity_key = (chunk_type, section, digest)
            occurrence = id_counts.get(identity_key, 0)
            id_counts[identity_key] = occurrence + 1
            cid = _chunk_id(
                doc,
                text=chunk_text,
                chunk_type=chunk_type,
                anchor=section,
                occurrence=occurrence,
            )
            chunks.append(
                Chunk(
                    id=cid,
                    doc_id=doc.doc_id,
                    root=doc.root.as_posix(),
                    path=doc.rel_path,
                    title=doc.title,
                    section=section,
                    text=chunk_text,
                    line_start=line_start,
                    line_end=line_end,
                    byte_start=cur_start,
                    byte_end=cur_end,
                    content_hash=doc.content_hash,
                    metadata={
                        "mtime_ns": doc.mtime_ns,
                        "size_bytes": doc.size_bytes,
                        "language": language,
                        "chunk_type": chunk_type,
                    },
                )
            )
        cur_start = None
        cur_end = None
        cur_texts = []
        cur_tokens = 0

    for start, end in para_spans:
        piece = text[start:end]
        tokens = _cheap_token_count(piece)
        if cur_start is None:
            cur_start, cur_end = start, end
            cur_texts = [piece]
            cur_tokens = tokens
            continue
        if cur_tokens + tokens > target:
            old_end = cur_end
            flush()
            # Rewind a small overlap window where possible.
            if overlap_chars > 0 and old_end is not None:
                ov_start = max(start, old_end - overlap_chars)
                cur_start, cur_end = ov_start, end
                cur_texts = [text[ov_start:end]]
                cur_tokens = _cheap_token_count(text[ov_start:end])
            else:
                cur_start, cur_end = start, end
                cur_texts = [piece]
                cur_tokens = tokens
        else:
            cur_end = end
            cur_texts.append(piece)
            cur_tokens += tokens
    flush()
    return chunks


def _bbox_union(blocks: list[dict]) -> list[float]:
    boxes = [block.get("bbox", [0, 0, 0, 0]) for block in blocks if block.get("bbox")]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(float(box[0]) for box in boxes), 2),
        round(min(float(box[1]) for box in boxes), 2),
        round(max(float(box[2]) for box in boxes), 2),
        round(max(float(box[3]) for box in boxes), 2),
    ]


def _chunk_pdf_document(doc: Document, settings: Settings) -> list[Chunk]:
    pages = list(doc.metadata.get("pages", []))
    if not pages:
        return []
    chunks: list[Chunk] = []
    target_tokens = max(128, settings.pdf_chunk_tokens)
    overlap_tokens = max(0, settings.pdf_chunk_overlap)
    global_line = 1
    occurrence_by_key: dict[tuple[str, str, str], int] = {}
    outline_by_page: dict[int, str] = {}
    for item in doc.metadata.get("outline", []):
        page_no = int(item.get("page", 0) or 0)
        title = str(item.get("title", "")).strip()
        if page_no and title:
            outline_by_page[page_no] = title

    for page in pages:
        blocks = [block for block in page.get("blocks", []) if str(block.get("text", "")).strip()]
        if not blocks:
            continue
        page_number = int(page.get("number", 0) or 0)
        page_label = str(page.get("label", page_number))
        current_section = outline_by_page.get(page_number, "")
        current: list[dict] = []
        current_tokens = 0
        block_index = 0
        local_line = 1

        def flush(
            section_name: str,
            *,
            page_data: dict = page,
            page_no: int = page_number,
            label: str = page_label,
        ) -> None:
            nonlocal current, current_tokens, global_line, local_line
            if not current:
                return
            text = "\n\n".join(str(block["text"]).strip() for block in current).strip()
            if not text:
                current = []
                current_tokens = 0
                return
            kinds = {str(block.get("kind", "text")) for block in current}
            chunk_type = (
                "pdf_table"
                if kinds == {"table"}
                else "pdf_code"
                if kinds == {"code"}
                else "pdf_text"
            )
            section = section_name or f"Page {label}"
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            identity_key = (chunk_type, f"{page_no}:{section}", digest)
            occurrence = occurrence_by_key.get(identity_key, 0)
            occurrence_by_key[identity_key] = occurrence + 1
            line_count = max(1, text.count("\n") + 1)
            start = int(page_data.get("start", 0))
            end = int(page_data.get("end", start + len(text)))
            cid = _chunk_id(
                doc,
                text=text,
                chunk_type=chunk_type,
                anchor=f"page:{page_no}:{section}",
                occurrence=occurrence,
            )
            chunks.append(
                Chunk(
                    id=cid,
                    doc_id=doc.doc_id,
                    root=doc.root.as_posix(),
                    path=doc.rel_path,
                    title=doc.title,
                    section=section[:240],
                    text=text,
                    line_start=global_line,
                    line_end=global_line + line_count - 1,
                    byte_start=start,
                    byte_end=end,
                    content_hash=doc.content_hash,
                    metadata={
                        "mtime_ns": doc.mtime_ns,
                        "size_bytes": doc.size_bytes,
                        "language": "pdf",
                        "document_type": "pdf",
                        "chunk_type": chunk_type,
                        "page": page_no,
                        "page_label": label,
                        "page_count": int(doc.metadata.get("page_count", 0) or 0),
                        "bbox": _bbox_union(current),
                        "ocr": bool(page_data.get("ocr", False)),
                        "ocr_error": str(page_data.get("ocr_error", "")),
                        "extractor": str(doc.metadata.get("extractor", "pymupdf")),
                        "source_title": doc.title,
                        "block_kinds": sorted(kinds),
                    },
                )
            )
            global_line += line_count
            local_line += line_count
            if overlap_tokens > 0:
                carry: list[dict] = []
                carry_tokens = 0
                for block in reversed(current):
                    tokens = _cheap_token_count(str(block["text"]))
                    if carry and carry_tokens + tokens > overlap_tokens:
                        break
                    carry.append(block)
                    carry_tokens += tokens
                current = list(reversed(carry))
                current_tokens = carry_tokens
            else:
                current = []
                current_tokens = 0

        while block_index < len(blocks):
            block = blocks[block_index]
            block_index += 1
            kind = str(block.get("kind", "text"))
            text = str(block.get("text", "")).strip()
            if kind == "heading":
                if current and current_tokens >= max(64, target_tokens // 3):
                    flush(current_section)
                current_section = text[:240]
                continue
            tokens = _cheap_token_count(text)
            if current and current_tokens + tokens > target_tokens:
                flush(current_section)
            current.append(block)
            current_tokens += tokens
            if kind == "table" or tokens >= target_tokens:
                flush(current_section)
        flush(current_section)
    return chunks
