#!/usr/bin/env python3
import argparse, re, sys, unicodedata, os
from collections import Counter, defaultdict
from pdfminer.high_level import extract_text
import requests, tempfile

DATA_DIR = "Data"

# ---------- Helpers ----------
def normalize_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s

def looks_like_header_footer(line: str) -> bool:
    # page numbers, running heads, arXiv footers, copyrights, short titles
    return bool(re.search(
        r"^(page\s*\d+|\d+\s*/\s*\d+|arxiv|preprint|©|copyright|all rights reserved|"
        r"this version|doi:|https?://|received\s*\d{4}|accepted\s*\d{4})",
        line, flags=re.I))

def strip_inline_citations(text: str) -> str:
    # [1], [2,3], [12–15]
    text = re.sub(r"\[(?:\d{1,3}(?:\s*[-–]\s*\d{1,3})?)(?:\s*,\s*\d{1,3})*\]", "", text)
    # (Smith, 2020), (Smith & Jones, 2020), (Smith et al., 2020; Doe, 2019)
    text = re.sub(r"\((?:[A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-z\-]+)?"
                  r"(?:,\s*\d{4}[a-z]?)"
                  r"(?:\s*;\s*[A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?)*|"
                  r"e\.g\.,?|i\.e\.,?)\)", "", text)
    return text

def strip_figure_table_captions(lines):
    out = []
    cap_pat = re.compile(r"^(figure|fig\.|table)\s*\d+[:.\s]", re.I)
    for ln in lines:
        if cap_pat.match(ln): 
            continue
        out.append(ln)
    return out

def drop_before_main_body(lines):
    # Remove everything before Abstract/Introduction to kill author/affiliation blocks
    idx = None
    key_pat = re.compile(r"^(abstract|introduction)\b", re.I)
    for i, ln in enumerate(lines):
        if key_pat.match(ln):
            idx = i
            break
    return lines[idx:] if idx is not None else lines

def drop_after_references(lines):
    # Remove references and anything after
    ref_pat = re.compile(r"^(references|bibliography)\b", re.I)
    for i, ln in enumerate(lines):
        if ref_pat.match(ln):
            return lines[:i]
    return lines

def filter_headers_footers(all_pages_lines):
    # Find lines that repeat on many pages → treat as header/footer
    flat = [l for page in all_pages_lines for l in page]
    counts = Counter(flat)
    # If a line occurs on >= 40% of pages, consider it header/footer
    threshold = max(2, int(0.4 * len(all_pages_lines)))
    common = {line for line, c in counts.items() if c >= threshold or looks_like_header_footer(line)}
    cleaned_pages = []
    for page in all_pages_lines:
        cleaned_pages.append([l for l in page if l not in common])
    return cleaned_pages

def fix_linebreak_hyphenation(text: str) -> str:
    """
    Fix words split by line breaks or soft hyphens.
    - Removes discretionary/soft hyphens and zero-width chars.
    - Joins end-of-line hyphenations across newlines.
    - Joins hyphen+space splits inside a line (PDF artifacts).
    Heuristic: if both fragments are long (>=4 chars), keep the hyphen
               (likely a real compound: e.g., 'tool-calling');
               otherwise drop it (e.g., 'mul- timodal' → 'multimodal').
    """
    # 0) Remove soft hyphens and zero-width artifacts
    text = text.replace("\u00AD", "")               # SOFT HYPHEN
    text = text.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")  # ZW*, joiners
    # normalize non-ASCII hyphens to ASCII hyphen for matching
    text = text.replace("\u2010", "-").replace("\u2011", "-")  # hyphen / non-breaking hyphen

    # Helper to decide keep-or-drop hyphen
    def _merge_or_keep(m):
        left, right = m.group(1), m.group(2)
        if len(left) >= 4 and len(right) >= 4:
            return f"{left}-{right}"   # keep real compound
        return f"{left}{right}"        # drop spurious break

    # 1) End-of-line splits: "...left-\n   right..."
    text = re.sub(r"(\b[A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,}\b)", _merge_or_keep, text)

    # 2) Intra-line splits caused by layout: "left-   right"
    text = re.sub(r"(\b[A-Za-z]{2,})-\s+([A-Za-z]{2,}\b)", _merge_or_keep, text)

    # 3) Over-aggressive space inside hyphenated compounds: "state- of-the-art"
    #    First, collapse " - " around hyphens that are *not* linebreak artifacts
    text = re.sub(r"\s*-\s*", "-", text)

    # 4) Final pass: catch stubborn spaced tokens like "mul- timodal" that survived
    text = re.sub(r"(\b[A-Za-z]{2,})-\s+([A-Za-z]{2,}\b)", _merge_or_keep, text)

    return text

# --- Footnote handling ---

# --- Layout-aware extraction with PyMuPDF ---
try:
    import fitz  # pymupdf
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

def extract_blocks_with_layout(pdf_path):
    """Return a list of pages; each page is a list of (x0,y0,x1,y1,text)."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for p in doc:
            h = p.rect.height
            blocks = []
            # "blocks" returns tuples: (x0, y0, x1, y1, text, block_no, block_type, ...)
            for b in p.get_text("blocks"):
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
                if not txt.strip():
                    continue
                blocks.append((x0, y0, x1, y1, txt.strip()))
            pages.append((h, blocks))
    return pages

def drop_footnote_blocks_layout(pages, bottom_ratio=0.18):
    """Remove text blocks in the bottom N% of each page, and blocks that look footnote-ish."""
    cleaned_pages = []
    foot_leader = re.compile(r"^\s*((\d{1,3}\s*[)\].:–-])|(\[\d{1,3}\])|([*†‡§¶]))\s+\S")
    keywords = re.compile(r"(equal contribution|corresponding author|email|acknowledg(e)?ments?|funded by|grant|doi)", re.I)
    for page_h, blocks in pages:
        cut_y = page_h * (1.0 - bottom_ratio)
        kept = []
        for (x0, y0, x1, y1, txt) in blocks:
            is_bottom = y0 >= cut_y
            looks_foot = foot_leader.match(txt) or keywords.search(txt)
            # If block is short and bottom-located or matches typical footnote leaders, drop it
            if is_bottom or looks_foot:
                # keep if it’s obviously a normal paragraph (long) and not bottom
                if not is_bottom and len(txt) > 300 and not looks_foot:
                    kept.append((x0, y0, x1, y1, txt))
                # else drop
            else:
                kept.append((x0, y0, x1, y1, txt))
        # Sort back by y,x and split to lines
        kept.sort(key=lambda b: (b[1], b[0]))
        lines = []
        for _, _, _, _, txt in kept:
            for ln in txt.splitlines():
                ln = normalize_line(ln)
                if ln:
                    lines.append(ln)
        cleaned_pages.append(lines)
    return cleaned_pages

# Unicode superscripts commonly used in PDFs
SUP_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
SUP_SET = set(SUP_DIGITS)

def strip_inline_footnote_markers(text: str) -> str:
    # unicode superscripts
    text = re.sub(r"(?<=\S)[⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", text)
    # caret-style ^12
    text = re.sub(r"(?<=\w)\^\d{1,3}", "", text)
    # trailing symbol markers
    text = re.sub(r"(?<=\S)[*†‡§¶]+", "", text)
    # bracketed inline numbers that are too close to words (avoid references by requiring adjacency)
    text = re.sub(r"(?<=\w)\[(?:\d{1,3})\]", "", text)
    return text

def page_strip_footnote_blocks(lines_on_page):
    """
    Heuristic: remove footnote *blocks* typically printed near the bottom.
    We look from the bottom up and drop lines that start like:
      1)  "1  ...", "12) ...", "1. ...", "1: ..."
      2)  "*, †, ‡, §, ¶" followed by text
      3)  Small scraps that are continuation of a footnote until a blank line or page break
    """
    # Patterns that likely start a footnote line
    starts_foot = re.compile(r"^\s*((\d{1,3}\s*[):.\-–])|([*†‡§¶]))\s+\S")
    # Some venues use bracketed numbers at page foot: "[1] text" – careful not to clash with references;
    # since we remove References section elsewhere, allow at page bottom:
    bracket_num_start = re.compile(r"^\s*\[\d{1,3}\]\s+\S")

    # Walk upward, toggling when we see a footnote starter within the bottom ~12 lines
    MAX_SCAN = min(12, len(lines_on_page))
    to_drop = set()
    in_block = False
    for idx in range(len(lines_on_page) - 1, max(-1, len(lines_on_page) - 1 - MAX_SCAN), -1):
        ln = lines_on_page[idx]
        if not in_block:
            if starts_foot.match(ln) or bracket_num_start.match(ln):
                in_block = True
                to_drop.add(idx)
                continue
        else:
            # keep consuming lines upwards that look like footnote continuation
            # continuation: indented or short or URL/email-heavy
            if (ln.strip() == "" or
                ln.startswith(" ") or
                len(ln) <= 80 or
                re.search(r"https?://|\S+@\S+", ln)):
                to_drop.add(idx)
                continue
            # stop if we hit a clear normal paragraph line
            break

    if not to_drop:
        return lines_on_page

    return [l for i, l in enumerate(lines_on_page) if i not in to_drop]

def strip_footnotes_from_pages(pages_lines):
    """Apply page_strip_footnote_blocks to each page's lines."""
    cleaned = []
    for page in pages_lines:
        cleaned.append(page_strip_footnote_blocks(page))
    return cleaned

def keep_only_sections(lines, sections):
    # Keep requested sections (inclusive until next top-level heading)
    if not sections: 
        return lines
    sec_pat = re.compile(r"^(abstract|introduction|related work|background|methods?|"
                         r"materials and methods|experiments?|results?|analysis|discussion|"
                         r"limitations?|conclusion[s]?|future work)\b", re.I)
    # Build index of headings
    heads = []
    for i, ln in enumerate(lines):
        if sec_pat.match(ln):
            heads.append((i, ln.lower()))
    if not heads:
        return lines  # fallback
    targets = set([s.lower() for s in sections])
    keep_spans = []
    for j, (start, name) in enumerate(heads):
        # map heading to canonical key
        key = re.sub(r"\s+", " ", name.split()[0])  # first word is enough
        if any(name.startswith(s.lower()) or key == s.lower() for s in targets):
            end = heads[j+1][0] if j+1 < len(heads) else len(lines)
            keep_spans.append((start, end))
    kept = []
    for a, b in keep_spans:
        kept.extend(lines[a:b])
    # If nothing matched, fall back
    return kept or lines

# --- Layout-aware extraction with image + vector drawing filtering (PyMuPDF) ---
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

def _rect_union(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    return (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))

def _pad(rect, pad=6):
    x0, y0, x1, y1 = rect
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

def _iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(1e-6, area_a + area_b - inter)

def _merge_rects(rects, iou_thresh=0.05, pad=4, max_passes=3):
    """Greedily merge overlapping/adjacent rectangles (helps form figure regions)."""
    rects = [ _pad(r, pad) for r in rects ]
    for _ in range(max_passes):
        merged, used = [], [False]*len(rects)
        for i in range(len(rects)):
            if used[i]: continue
            cur = rects[i]; used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(i+1, len(rects)):
                    if used[j]: continue
                    if _iou(cur, rects[j]) >= iou_thresh:
                        cur = _rect_union(cur, rects[j])
                        used[j] = True
                        changed = True
            merged.append(cur)
        rects = merged
    return rects

def extract_layout_with_graphics(pdf_path):
    """
    Returns: list of pages.
      Each page is dict:
        {
          "height": page_height,
          "text_blocks": [ { "rect":(x0,y0,x1,y1), "text":str, "font":avg_font_size } ... ],
          "image_rects": [ (x0,y0,x1,y1), ... ],
          "graphic_rects": [ merged drawing rects ... ]
        }
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            h = page.rect.height
            d = page.get_text("dict")  # text and images with structure
            text_blocks, image_rects = [], []

            # collect text & image blocks
            for b in d.get("blocks", []):
                if b.get("type") == 1:
                    image_rects.append(tuple(b["bbox"]))
                elif b.get("type") == 0 and b.get("lines"):
                    x0, y0, x1, y1 = b["bbox"]
                    sizes, parts = [], []
                    for ln in b["lines"]:
                        for sp in ln.get("spans", []):
                            if "size" in sp: sizes.append(sp["size"])
                            if "text" in sp: parts.append(sp["text"])
                    avg_size = sum(sizes)/len(sizes) if sizes else 0.0
                    text = " ".join(t.strip() for t in parts if t and t.strip())
                    if text:
                        text_blocks.append({"rect": (x0,y0,x1,y1), "text": text, "font": avg_size})

            # collect vector drawing rects (lines, curves, fills)
            draw_rects = []
            for dr in page.get_drawings():
                # each drawing item may have 'rect' or list of items with 'rect'
                if "rect" in dr and dr["rect"] is not None:
                    r = dr["rect"]
                    draw_rects.append((r.x0, r.y0, r.x1, r.y1))
                # also consider the bbox over paths if present
                if "items" in dr:
                    for it in dr["items"]:
                        if hasattr(it, "rect") and it.rect is not None:
                            r = it.rect
                            draw_rects.append((r.x0, r.y0, r.x1, r.y1))

            # merge drawings into larger figure/axes regions
            graphic_rects = _merge_rects(draw_rects, iou_thresh=0.05, pad=2, max_passes=2)

            pages.append({
                "height": h,
                "text_blocks": text_blocks,
                "image_rects": image_rects,
                "graphic_rects": graphic_rects
            })
    return pages

def remove_figure_related_text(text_blocks, image_rects, graphic_rects,
                               iou_thresh=0.12, edge_pad=6, small_font=10.5):
    """
    Drop text that overlaps images OR vector graphics, or hugs their edges with tiny fonts,
    plus typical panel labels like '(a)'.
    """
    kept = []
    padded_fig_regions = [ _pad(r, edge_pad) for r in (image_rects + graphic_rects) ]
    for tb in text_blocks:
        rect, font, txt = tb["rect"], tb["font"], tb["text"].strip()

        # exact overlap with image or vector graphics region
        overlap_img = any(_iou(rect, r) >= iou_thresh for r in image_rects)
        overlap_vec = any(_iou(rect, r) >= iou_thresh for r in graphic_rects)

        # hugging edges with small font (axes ticks / panel letters / legend)
        near_edge_small = (font and font <= small_font) and any(_iou(rect, r) > 0.0 for r in padded_fig_regions)

        # panel labels like (a), a), (A), A., etc.
        is_panel = bool(re.fullmatch(r"[\(\[]?[a-hA-H][\)\]\.:]?", txt))

        # short axis-like text heavy with digits or short tokens
        digit_ratio = sum(ch.isdigit() for ch in txt) / max(1, len(txt))
        many_short = sum(1 for t in txt.split() if len(t) <= 3) >= 2
        axis_like = (len(txt) <= 30) and (digit_ratio > 0.25 or many_short)

        if overlap_img or overlap_vec or near_edge_small or is_panel or axis_like:
            continue
        kept.append(tb)
    return kept

def pages_to_lines_layout_filtered(pages, bottom_ratio=0.18, drop_figure_text=True):
    """
    Bottom-of-page (footnotes) + figure-text removal, then return lines per page.
    """
    cleaned_pages = []
    foot_leader = re.compile(r"^\s*((\d{1,3}\s*[)\].:–-])|(\[\d{1,3}\])|([*†‡§¶]))\s+\S", re.I)
    foot_keywords = re.compile(r"(equal contribution|corresponding author|email|acknowledg(e)?ment|funded by|grant|doi)", re.I)

    for pg in pages:
        h = pg["height"]
        text_blocks = pg["text_blocks"]
        image_rects = pg["image_rects"]
        graphic_rects = pg["graphic_rects"]

        # figure-text removal (raster + vector)
        if drop_figure_text:
            text_blocks = remove_figure_related_text(text_blocks, image_rects, graphic_rects)

        # drop bottom footnote-ish blocks
        cut_y = h * (1.0 - bottom_ratio)
        kept = []
        for tb in text_blocks:
            x0,y0,x1,y1 = tb["rect"]
            txt = tb["text"]
            is_bottom = y0 >= cut_y
            looks_foot = foot_leader.match(txt) or foot_keywords.search(txt)
            if is_bottom or looks_foot:
                # keep unusually long paragraphs not near bottom
                if (not is_bottom) and (len(txt) > 300) and (not looks_foot):
                    kept.append(tb)
            else:
                kept.append(tb)

        kept.sort(key=lambda b: (b["rect"][1], b["rect"][0]))
        lines = []
        for tb in kept:
            for ln in tb["text"].splitlines():
                ln = normalize_line(ln)
                if ln:
                    lines.append(ln)
        cleaned_pages.append(lines)
    return cleaned_pages

# --- Math / formula removal ---

# Detect tuple patterns like "( X 0 , I 1 )" with spaced tokens
TUPLE_PAIR = re.compile(
    r"\(\s*[A-Za-z]\s*\d+\s*,\s*[A-Za-z]\s*\d+\s*\)"
)

def looks_like_spaced_math_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False

    # strong triggers: equation/set syntax + braces/parens
    has_eq = any(ch in t for ch in "=≡≤≥≈")
    has_braces = any(ch in t for ch in "{}()[]")
    if has_eq and has_braces:
        # many commas/semicolons or tuple-pair pattern -> almost surely math
        if TUPLE_PAIR.search(t) or t.count(",") + t.count(";") >= 2:
            return True

    # token statistics: many 1–2 char tokens is very math-ish after PDF split
    toks = re.findall(r"\S+", t)
    if toks:
        short = sum(1 for z in toks if len(z) <= 2)
        sym = sum(1 for ch in t if ch in "{}()[]=,;:<>^_+-*/\\")
        if (short / len(toks) > 0.6 and sym >= 2) or (sym / max(1, len(t)) > 0.12):
            return True

    return False

def strip_spaced_math_lines(text: str) -> str:
    out = []
    for ln in text.splitlines():
        if looks_like_spaced_math_line(ln):
            # drop entire line
            continue
        out.append(ln)
    return "\n".join(out)

# LaTeX environments to nuke entirely (multiline)
MATH_ENVS = [
    "equation", "equation*", "align", "align*", "alignat", "alignat*", "gather", "gather*",
    "multline", "multline*", "flalign", "flalign*", "eqnarray", "eqnarray*", "cases", "split",
    "IEEEeqnarray", "IEEEeqnarray*"
]
MATH_ENV_RE = re.compile(
    r"(?s)\\begin\{(" + "|".join(MATH_ENVS) + r")\}.*?\\end\{\1\}"
)

# Inline/display TeX math delimiters: $...$, $$...$$, \(...\), \[...\]
INLINE_MATH_RE = re.compile(
    r"(?s)(\${1,2})(?:(?:(?<!\\).)*?)(?<!\\)\1"   # $...$ or $$...$$ (non-greedy, allow newlines)
)
PAREN_MATH_RE = re.compile(r"(?s)\\\((?:(?:(?<!\\).)*?)(?<!\\)\\\)")
BRACK_MATH_RE = re.compile(r"(?s)\\\[(?:(?:(?<!\\).)*?)(?<!\\)\\\]")

# Lines “dense” with math symbols/operators → drop whole line
MATH_CHARS = r"=+\-*/^_<>≤≥≈≃≅≡→←↔∈∉∋⊂⊃⊆⊇∪∩∑∏∫∮∞∇√′″·⋅∘⊗⊕│‖{}()\[\]|,:;\\"
MATH_DENSE_LINE = re.compile(rf"[{re.escape(MATH_CHARS)}]")

# Subscript/superscript patterns common in text after PDF extraction
SUB_SUP_PAT = re.compile(r"([A-Za-z]\s*[_^]\s*(?:\d+|[A-Za-z]))")

def strip_math(text: str) -> str:
    # 1) Remove explicit LaTeX math environments (multiline)
    text = MATH_ENV_RE.sub("", text)

    # 2) Remove inline/display math spans
    text = INLINE_MATH_RE.sub("", text)
    text = PAREN_MATH_RE.sub("", text)
    text = BRACK_MATH_RE.sub("", text)

    # 3) Drop lines that look like standalone equations or are math-dense
    out_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out_lines.append(ln)
            continue

        # Heuristics: lots of math symbols, braces, or very short symboly lines
        math_syms = len(MATH_DENSE_LINE.findall(s))
        digit_ratio = sum(ch.isdigit() for ch in s) / max(1, len(s))
        brace_ratio = sum(ch in "{}[]" for ch in s) / max(1, len(s))
        op_ratio = sum(ch in "=+-*/^_\\|" for ch in s) / max(1, len(s))

        # Drop if any of these: high operator density / braces / digitish equation
        if op_ratio > 0.12 or brace_ratio > 0.12 or digit_ratio > 0.40:
            # Keep short “plain English with = 0.9” cases by requiring length or multiple ops
            if len(s) <= 80 or s.count("=") >= 2 or s.count("\\") >= 1:
                continue

        # 4) Remove residual sub/sup scripts glued into prose (X_t, y^i)
        s = SUB_SUP_PAT.sub("", s)

        out_lines.append(s)

    # 5) Clean extra spaces/blank lines
    text2 = "\n".join(out_lines)
    text2 = re.sub(r"\s{2,}", " ", text2)
    text2 = re.sub(r"\n{3,}", "\n\n", text2).strip()
    return text2

def clean_pdf(path, sections=None, bottom_ratio=0.18, no_figure_text=True):
    if HAS_FITZ:
        try:
            layout_pages = extract_layout_with_graphics(path)
            pages_lines = pages_to_lines_layout_filtered(
                layout_pages,
                bottom_ratio=bottom_ratio,
                drop_figure_text=no_figure_text
            )
        except Exception:
            # fallback to pdfminer
            raw = extract_text(path)
            pages = [p for p in raw.split("\f") if p.strip()]
            pages_lines = []
            for p in pages:
                lines = [normalize_line(x) for x in p.splitlines()]
                lines = [l for l in lines if l]
                pages_lines.append(lines)
    else:
        raw = extract_text(path)
        pages = [p for p in raw.split("\f") if p.strip()]
        pages_lines = []
        for p in pages:
            lines = [normalize_line(x) for x in p.splitlines()]
            lines = [l for l in lines if l]
            pages_lines.append(lines)

    # headers/footers (+ pdfminer footnotes if no fitz)
    pages_lines = filter_headers_footers(pages_lines)
    if not HAS_FITZ:
        pages_lines = strip_footnotes_from_pages(pages_lines)

    # flatten → your existing pipeline
    lines = [l for page in pages_lines for l in page]
    lines = drop_before_main_body(lines)
    lines = drop_after_references(lines)

    meta_re = re.compile(r"(doi:\s*\S+|arxiv:\s*\S+|©|copyright|creative commons|"
                         r"license|correspondence:\s*\S+@\S+|received:\s*\w+)", re.I)
    lines = [l for l in lines if not meta_re.search(l)]

    lines = strip_figure_table_captions(lines)
    lines = keep_only_sections(lines, sections)

    text = "\n".join(lines)
    text = fix_linebreak_hyphenation(text)
    text = strip_inline_footnote_markers(text)
    text = strip_math(text)
    text = strip_spaced_math_lines(text)
    text = strip_inline_citations(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or ""

def resolve_pdf_input(arg):
    # Local path first
    if not arg.startswith("http"):
        if not os.path.isabs(arg) and not os.path.exists(arg):
            arg = os.path.join(DATA_DIR, arg)
        return arg

    # URL case
    url = arg.strip()

    # Normalize arXiv – accept /abs/ or /pdf/ with/without .pdf
    m = re.search(r"arxiv\.org/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", url)
    if m:
        pid = m.group(2) + (m.group(3) or "")
        url = f"https://arxiv.org/pdf/{pid}.pdf"

    # Download with content-type check
    resp = requests.get(url, stream=True, timeout=30)
    ct = resp.headers.get("Content-Type", "")
    if ("pdf" not in ct.lower()) and (not url.lower().endswith(".pdf")):
        raise ValueError(f"URL does not look like a PDF (Content-Type={ct}). Use a direct PDF link.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in resp.iter_content(1 << 20):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    return tmp.name


def main():
    ap = argparse.ArgumentParser(description="Clean academic PDFs for TTS-friendly listening.")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("-o", "--output", default=None, help="Output .txt path (default: <pdf>_clean.txt)")
    ap.add_argument("--sections", nargs="*", default=None,
                    help="Only keep these sections (e.g. Abstract Introduction Methods Results Discussion Conclusion).")
    ap.add_argument("--no-figure-text", action="store_true",
                help="Drop text overlapping images or vector-drawn figures (axes/ticks/panel labels).")

    args = ap.parse_args()
    pdf_path = resolve_pdf_input(args.pdf)
    out = args.output or (re.sub(r"\.pdf$", "", os.path.basename(pdf_path), flags=re.I) + "_clean.txt")

    try:
        cleaned = clean_pdf(pdf_path, sections=args.sections, no_figure_text=args.no_figure_text)
    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}", file=sys.stderr)
        sys.exit(2)

    with open(out, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"Saved cleaned text to: {out}")

if __name__ == "__main__":
    main()