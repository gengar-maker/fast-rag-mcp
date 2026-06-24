from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from .io import stable_id
from .types import CodeReference, CodeSymbol, Document

log = logging.getLogger(__name__)

EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cu": "cpp",
    ".cuh": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
}

SYMBOL_NODE_KINDS: dict[str, tuple[str, ...]] = {
    "python": ("function_definition", "class_definition"),
    "javascript": (
        "function_declaration",
        "method_definition",
        "class_declaration",
        "generator_function_declaration",
    ),
    "typescript": (
        "function_declaration",
        "method_definition",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "generator_function_declaration",
    ),
    "tsx": (
        "function_declaration",
        "method_definition",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "generator_function_declaration",
    ),
    "go": ("function_declaration", "method_declaration", "type_declaration"),
    "rust": ("function_item", "struct_item", "enum_item", "impl_item", "trait_item", "mod_item"),
    "java": (
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "constructor_declaration",
        "enum_declaration",
    ),
    "kotlin": (
        "class_declaration",
        "function_declaration",
        "object_declaration",
        "interface_declaration",
    ),
    "c": ("function_definition", "struct_specifier", "enum_specifier"),
    "cpp": (
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "namespace_definition",
    ),
    "c_sharp": (
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "struct_declaration",
        "enum_declaration",
    ),
    "ruby": ("method", "singleton_method", "class", "module"),
    "php": (
        "function_definition",
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
    ),
    "swift": (
        "function_declaration",
        "class_declaration",
        "struct_declaration",
        "enum_declaration",
        "protocol_declaration",
    ),
    "scala": ("function_definition", "class_definition", "object_definition", "trait_definition"),
}

KIND_BY_NODE = {
    "function_definition": "function",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "function_item": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "method": "method",
    "singleton_method": "method",
    "constructor_declaration": "constructor",
    "class_definition": "class",
    "class_declaration": "class",
    "class": "class",
    "class_specifier": "class",
    "interface_declaration": "interface",
    "trait_declaration": "trait",
    "trait_definition": "trait",
    "trait_item": "trait",
    "type_alias_declaration": "type",
    "type_declaration": "type",
    "struct_item": "struct",
    "struct_specifier": "struct",
    "struct_declaration": "struct",
    "enum_item": "enum",
    "enum_declaration": "enum",
    "enum_specifier": "enum",
    "impl_item": "impl",
    "namespace_definition": "namespace",
    "mod_item": "module",
    "module": "module",
    "object_declaration": "object",
    "object_definition": "object",
    "protocol_declaration": "protocol",
}

IDENT_NODE_TYPES = {
    "identifier",
    "field_identifier",
    "property_identifier",
    "type_identifier",
    "constant",
    "word",
    "name",
}

REGEX_SYMBOLS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^([ \t]*)(async\s+def|def|class)\s+([A-Za-z_][\w]*)", re.MULTILINE),
    "javascript": re.compile(
        r"^([ \t]*)(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)", re.MULTILINE
    ),
    "typescript": re.compile(
        r"^([ \t]*)(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)",
        re.MULTILINE,
    ),
    "tsx": re.compile(
        r"^([ \t]*)(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)",
        re.MULTILINE,
    ),
    "go": re.compile(
        r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)|^type\s+([A-Za-z_][\w]*)", re.MULTILINE
    ),
    "rust": re.compile(
        r"^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_][\w]*)?",
        re.MULTILINE,
    ),
    "java": re.compile(
        r"^\s*(?:public|private|protected|static|final|abstract|\s)+\s*(?:class|interface|enum)\s+([A-Za-z_][\w]*)|^\s*(?:public|private|protected|static|final|synchronized|async|\s)+[\w<>\[\], ?]+\s+([A-Za-z_][\w]*)\s*\(",
        re.MULTILINE,
    ),
    "cpp": re.compile(
        r"^\s*(?:class|struct|enum|namespace)\s+([A-Za-z_][\w]*)|^\s*[\w:<>~*&\s]+\s+([A-Za-z_~][\w:~]*)\s*\([^;]*\)\s*(?:const\s*)?(?:\{|$)",
        re.MULTILINE,
    ),
    "c": re.compile(
        r"^\s*(?:struct|enum)\s+([A-Za-z_][\w]*)|^\s*[\w_*\s]+\s+([A-Za-z_][\w]*)\s*\([^;]*\)\s*\{",
        re.MULTILINE,
    ),
}

IMPORT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w.*, {}]+)", re.MULTILINE),
    re.compile(
        r"^\s*import\s+(?:type\s+)?(?:[\w*{},\s]+\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE
    ),
    re.compile(r"^\s*export\s+.*\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
    re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE),
    re.compile(r"^\s*#include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE),
    re.compile(r"^\s*package\s+([\w.]+)", re.MULTILINE),
)

MEMBER_CALL_RE = re.compile(r"(?:[A-Za-z_$][\w$]*\.)+([A-Za-z_$][\w$]{1,80})\s*\(")
CALL_RE = re.compile(r"(?<![.\w$])([A-Za-z_$][\w$]{1,80})\s*\(")
SKIP_CALL_NAMES = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "typeof",
    "println",
    "print",
    "assert",
    "dict",
    "list",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "len",
    "range",
}


def detect_language(path: str | Path) -> str:
    p = Path(path)
    if p.name == "Dockerfile" or p.name.endswith(".dockerfile"):
        return "dockerfile"
    return EXT_LANGUAGE.get(p.suffix.lower(), "")


@lru_cache(maxsize=64)
def _get_parser(language: str) -> Any | None:
    if not language:
        return None
    # Preferred modern package: prebuilt wheels with get_parser/get_language.
    try:
        from tree_sitter_language_pack import get_parser as get_pack_parser

        return get_pack_parser(language)  # type: ignore[arg-type]
    except Exception as exc:
        log.debug("tree-sitter-language-pack parser unavailable for %s: %s", language, exc)
    # Older package still exists and has the same basic get_parser API, but is unmaintained.
    try:
        from tree_sitter_languages import get_parser as get_legacy_parser

        return get_legacy_parser(language)
    except Exception as exc:
        log.debug("tree-sitter parser unavailable for %s: %s", language, exc)
        return None


def extract_symbols(doc: Document) -> list[CodeSymbol]:
    language = detect_language(doc.rel_path)
    if not language:
        return []
    symbols = _extract_symbols_tree_sitter(doc, language)
    if symbols:
        return _dedupe_symbols(symbols)
    return _extract_symbols_regex(doc, language)


def extract_imports(text: str) -> list[str]:
    imports: list[str] = []
    for pat in IMPORT_PATTERNS:
        for m in pat.finditer(text):
            for group in m.groups():
                if not group:
                    continue
                imports.append(group.strip())
    return sorted(set(imports))[:256]


def extract_references(doc: Document, symbols: list[CodeSymbol]) -> list[CodeReference]:
    language = detect_language(doc.rel_path)
    if not language:
        return []
    refs: list[CodeReference] = []
    lines = doc.text.splitlines()

    # Sweep symbol intervals once instead of scanning every symbol for every source line.
    # This matters for generated/large files with hundreds of symbols.
    symbols_by_start = sorted(symbols, key=lambda s: (s.line_start, -s.line_end))
    active: list[CodeSymbol] = []
    next_symbol = 0

    seen: set[tuple[str, str, int, str]] = set()
    for line_no, line in enumerate(lines, start=1):
        while (
            next_symbol < len(symbols_by_start)
            and symbols_by_start[next_symbol].line_start <= line_no
        ):
            active.append(symbols_by_start[next_symbol])
            next_symbol += 1
        if active:
            active = [sym for sym in active if sym.line_end >= line_no]
        source_symbol_id = ""
        if active:
            # Prefer the innermost symbol containing the line.
            source_symbol_id = max(
                active,
                key=lambda sym: (sym.line_start, -(sym.line_end - sym.line_start)),
            ).id

        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "*")):
            continue
        if _looks_like_import_line(stripped):
            for name in _import_names_from_line(stripped):
                key = (source_symbol_id, name, line_no, "import")
                if key not in seen:
                    seen.add(key)
                    refs.append(_make_ref(doc, language, key[0], name, "import", line_no, stripped))
        if _looks_like_definition_line(stripped):
            continue
        for m in MEMBER_CALL_RE.finditer(line):
            name = m.group(1)
            if name in SKIP_CALL_NAMES:
                continue
            key = (source_symbol_id, name, line_no, "call")
            if key not in seen:
                seen.add(key)
                refs.append(_make_ref(doc, language, key[0], name, "call", line_no, stripped[:240]))
        for m in CALL_RE.finditer(line):
            name = m.group(1)
            if name in SKIP_CALL_NAMES or name[:1].islower() and name in {"require", "import"}:
                continue
            key = (source_symbol_id, name, line_no, "call")
            if key not in seen:
                seen.add(key)
                refs.append(_make_ref(doc, language, key[0], name, "call", line_no, stripped[:240]))
        if len(refs) >= 20_000:
            break
    return refs


def _make_ref(
    doc: Document,
    language: str,
    source_symbol_id: str,
    target_name: str,
    ref_kind: str,
    line: int,
    snippet: str,
) -> CodeReference:
    rid = stable_id(doc.doc_id, source_symbol_id, target_name, ref_kind, str(line), snippet, n=28)
    return CodeReference(
        id=rid,
        doc_id=doc.doc_id,
        root=doc.root.as_posix(),
        path=doc.rel_path,
        language=language,
        source_symbol_id=source_symbol_id,
        target_name=target_name,
        ref_kind=ref_kind,
        line=line,
        snippet=snippet,
        metadata={},
    )


def _extract_symbols_tree_sitter(doc: Document, language: str) -> list[CodeSymbol]:
    parser = _get_parser(language)
    if parser is None:
        return []
    kinds = set(SYMBOL_NODE_KINDS.get(language, ()))
    if not kinds:
        return []
    source = doc.text.encode("utf-8", "ignore")
    try:
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as exc:
        log.debug("tree-sitter parse failed for %s: %s", doc.rel_path, exc)
        return []

    symbols: list[CodeSymbol] = []
    identity_counts: dict[tuple[str, str, str], int] = {}

    def visit(node: Any, parents: list[str]) -> None:
        node_type = getattr(node, "type", "")
        next_parents = parents
        if node_type in kinds:
            name = _node_name(node, source)
            if name:
                kind = KIND_BY_NODE.get(node_type, node_type)
                parent = ".".join(parents)
                qualified = f"{parent}.{name}" if parent else name
                text = _node_text(source, node)
                sig = _signature(text)
                docstring = _docstring_from_symbol_text(text, language)
                identity_key = (kind, qualified, sig)
                occurrence = identity_counts.get(identity_key, 0)
                identity_counts[identity_key] = occurrence + 1
                sym = CodeSymbol(
                    id=stable_id(
                        "symbol-v2",
                        doc.doc_id,
                        language,
                        kind,
                        qualified,
                        sig,
                        str(occurrence),
                        n=28,
                    ),
                    doc_id=doc.doc_id,
                    root=doc.root.as_posix(),
                    path=doc.rel_path,
                    language=language,
                    kind=kind,
                    name=name,
                    qualified_name=qualified,
                    parent=parent,
                    signature=sig,
                    docstring=docstring,
                    text=text,
                    line_start=int(node.start_point[0]) + 1,
                    line_end=int(node.end_point[0]) + 1,
                    byte_start=int(node.start_byte),
                    byte_end=int(node.end_byte),
                    metadata={"parser": "tree-sitter", "node_type": node_type},
                )
                symbols.append(sym)
                if kind in {
                    "class",
                    "interface",
                    "trait",
                    "struct",
                    "enum",
                    "impl",
                    "module",
                    "object",
                    "namespace",
                }:
                    next_parents = [*parents, name]
        for child in getattr(node, "named_children", []) or []:
            visit(child, next_parents)

    visit(root, [])
    return symbols


def _extract_symbols_regex(doc: Document, language: str) -> list[CodeSymbol]:
    pat = REGEX_SYMBOLS.get(language) or REGEX_SYMBOLS.get("typescript")
    if pat is None:
        return []
    offsets = _line_offsets(doc.text)
    matches = list(pat.finditer(doc.text))
    symbols: list[CodeSymbol] = []
    identity_counts: dict[tuple[str, str, str], int] = {}
    for i, m in enumerate(matches):
        name = next(
            (
                g.strip()
                for g in m.groups()
                if g
                and g.strip()
                and not g.isspace()
                and not re.fullmatch(
                    r"async\s+def|def|class|function|interface|type|enum|struct|trait|impl|pub|export",
                    g.strip(),
                )
            ),
            "",
        )
        if not name:
            # Rust impl Foo has no name in first capture; keep it as impl@line.
            if "impl" in m.group(0):
                name = f"impl@{_line_for_offset(offsets, m.start())}"
            else:
                continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)
        line_start = _line_for_offset(offsets, start)
        line_end = _line_for_offset(offsets, end)
        text = doc.text[start:end].strip()
        head = m.group(0)
        kind = _kind_from_regex_head(head)
        signature = _signature(text)
        identity_key = (kind, name, signature)
        occurrence = identity_counts.get(identity_key, 0)
        identity_counts[identity_key] = occurrence + 1
        sym = CodeSymbol(
            id=stable_id(
                "symbol-v2",
                doc.doc_id,
                language,
                kind,
                name,
                signature,
                str(occurrence),
                n=28,
            ),
            doc_id=doc.doc_id,
            root=doc.root.as_posix(),
            path=doc.rel_path,
            language=language,
            kind=kind,
            name=name,
            qualified_name=name,
            parent="",
            signature=signature,
            docstring=_docstring_from_symbol_text(text, language),
            text=text,
            line_start=line_start,
            line_end=line_end,
            byte_start=start,
            byte_end=end,
            metadata={"parser": "regex"},
        )
        symbols.append(sym)
    return _dedupe_symbols(symbols)


def _dedupe_symbols(symbols: Iterable[CodeSymbol]) -> list[CodeSymbol]:
    out: list[CodeSymbol] = []
    seen: set[tuple[str, int, int, str]] = set()
    for sym in symbols:
        key = (sym.qualified_name, sym.line_start, sym.line_end, sym.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(sym)
    return sorted(out, key=lambda s: (s.path, s.line_start, s.line_end, s.qualified_name))


def _node_name(node: Any, source: bytes) -> str:
    for field in ("name", "declarator", "declarator", "type"):
        try:
            candidate = node.child_by_field_name(field)
        except Exception:
            candidate = None
        name = _first_identifier(candidate, source)
        if name:
            return name
    for child in getattr(node, "children", []) or []:
        name = _first_identifier(child, source)
        if name:
            return name
    return ""


def _first_identifier(node: Any | None, source: bytes) -> str:
    if node is None:
        return ""
    node_type = getattr(node, "type", "")
    if node_type in IDENT_NODE_TYPES:
        text = _node_text(source, node).strip()
        if _is_reasonable_identifier(text):
            return text
    for child in getattr(node, "children", []) or []:
        name = _first_identifier(child, source)
        if name:
            return name
    return ""


def _is_reasonable_identifier(text: str) -> bool:
    return bool(text) and len(text) <= 180 and not any(c.isspace() for c in text)


def _node_text(source: bytes, node: Any) -> str:
    return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", "replace")


def _signature(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
        joined = " ".join(lines)
        if "{" in stripped or stripped.endswith(":") or len(joined) > 260:
            break
    sig = " ".join(lines)
    sig = sig.split("{")[0].rstrip()
    return sig[:500]


def _docstring_from_symbol_text(text: str, language: str) -> str:
    lines = text.splitlines()[1:12]
    buf: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buf:
                break
            continue
        if language == "python" and (stripped.startswith('"""') or stripped.startswith("'''")):
            return stripped.strip("\"' ")[:800]
        if stripped.startswith(("#", "//", "///", "*")):
            buf.append(stripped.lstrip("#/ *"))
            continue
        if stripped.startswith("/*"):
            in_block = True
            buf.append(stripped.lstrip("/* "))
            if "*/" in stripped:
                break
            continue
        if in_block:
            buf.append(stripped.rstrip("*/ ").lstrip("* "))
            if "*/" in stripped:
                break
            continue
        break
    return "\n".join(x for x in buf if x)[:800]


def _kind_from_regex_head(head: str) -> str:
    h = head.lower()
    if "class" in h:
        return "class"
    if "interface" in h:
        return "interface"
    if "type" in h:
        return "type"
    if "enum" in h:
        return "enum"
    if "struct" in h:
        return "struct"
    if "trait" in h:
        return "trait"
    if "impl" in h:
        return "impl"
    return "function"


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for m in re.finditer("\n", text):
        offsets.append(m.end())
    return offsets


def _line_for_offset(offsets: list[int], pos: int) -> int:
    lo, hi = 0, len(offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)


def _looks_like_import_line(line: str) -> bool:
    return (
        line.startswith(("import ", "from ", "use ", "#include", "package ", "require("))
        or " from " in line
        and ("import" in line or "export" in line)
    )


def _import_names_from_line(line: str) -> list[str]:
    names = re.findall(r"[A-Za-z_$][\w$]*(?=\s*(?:,|}|$|\sas\s))", line)
    # Also keep module-ish path tail: './foo/bar' -> bar.
    for quoted in re.findall(r"['\"]([^'\"]+)['\"]", line):
        tail = quoted.rstrip("/").split("/")[-1].split(".")[0]
        if tail:
            names.append(tail)
    return [
        n
        for n in names
        if n not in {"import", "from", "as", "use", "package", "require", "export", "type"}
    ][:64]


def _looks_like_definition_line(line: str) -> bool:
    return bool(
        re.match(
            r"^(?:async\s+)?def\s+|^(?:export\s+)?(?:async\s+)?function\s+|^func\s+|^(?:pub\s+)?(?:async\s+)?fn\s+|^(?:public|private|protected|static|final|abstract|\s)+[A-Za-z0-9_<>, ?\[\]]+\s+[A-Za-z_][\w]*\s*\(",
            line,
        )
    )
