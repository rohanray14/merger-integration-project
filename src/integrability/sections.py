"""Cut Item 1 (Business) and Item 7 (MD&A) out of a 10-K's text.

Section boundaries per AGENTS.md:
  Item 1: from "Item 1. Business" to "Item 1A" or "Item 2".
  Item 7: from "Item 7. Management's Discussion" to "Item 7A" or "Item 8".
Skip the table-of-contents match (take the occurrence followed by real text).

Handles variant headers: colons, dashes, 10-KSB "Plan of Operation", etc.
"""
import re
from html.parser import HTMLParser


# ── HTML stripping ───────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Simple HTML-to-text converter."""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(raw: str) -> str:
    """Convert HTML to plain text. If input is already plain text, pass through."""
    # Check more of the document for HTML markers (some filings have long SGML preambles)
    probe = raw[:20000].lower()
    if "<html" not in probe and "<body" not in probe and "<table" not in probe:
        # Likely plain text wrapped in SEC SGML envelope
        # Still strip any stray tags (e.g. <PAGE>, <TABLE>)
        raw = re.sub(r"<[^>]+>", " ", raw)
        return raw
    s = _HTMLStripper()
    s.feed(raw)
    text = s.get_text()
    # If HTML parsing produced very little text, fall back to regex stripping
    if len(text.strip()) < 500 and len(raw) > 2000:
        return re.sub(r"<[^>]+>", " ", raw)
    return text


# ── Section extraction ───────────────────────────────────────────────────────

# Patterns for section starts (case-insensitive)
# Allow separators: period, colon, dash, em-dash, or just whitespace after item number
_SEP = r"[\.\:\-\—]?\s+"

_ITEM1_START = re.compile(
    r"\bitem\s+1" + _SEP +
    r"(?:business|description\s+of\s+(?:the\s+)?business|general)[^\n]*",
    re.I,
)
_ITEM1_END = re.compile(
    r"\bitem\s+1a[\.\:\s]|\bitem\s+2[\.\:\s]",
    re.I,
)

_ITEM7_START = re.compile(
    r"\bitem\s+7" + _SEP +
    r"(?:management|mgmt|financial\s+condition|plan\s+of\s+operation|results\s+of\s+operation)[^\n]*",
    re.I,
)
_ITEM7_END = re.compile(
    r"\bitem\s+7a[\.\:\s]|\bitem\s+8[\.\:\s]",
    re.I,
)

# Minimum characters after a header for it to count as the real section
# (not a table-of-contents entry). Lowered from 1000 to catch shorter but
# legitimate sections (especially smaller firms / 10-KSB filers).
_MIN_SECTION_LEN = 400


def _find_section(text: str, start_re, end_re) -> str | None:
    """Find a section bounded by start_re .. end_re, skipping TOC entries."""
    matches = list(start_re.finditer(text))
    if not matches:
        return None

    for m in matches:
        # Find the next end marker after this start
        end_m = end_re.search(text, m.end())
        if end_m is None:
            candidate = text[m.end():]
        else:
            candidate = text[m.end():end_m.start()]

        # Skip TOC: real sections have substantial text
        stripped = candidate.strip()
        if len(stripped) >= _MIN_SECTION_LEN:
            return stripped

    return None


def extract(text: str, item: str) -> str | None:
    """Extract a section from 10-K plain text.

    Args:
        text: Full filing text (already stripped of HTML).
        item: "item1" or "item7".

    Returns:
        Section text or None if not found.
    """
    if item == "item1":
        return _find_section(text, _ITEM1_START, _ITEM1_END)
    elif item == "item7":
        return _find_section(text, _ITEM7_START, _ITEM7_END)
    else:
        raise ValueError(f"Unknown item: {item!r}. Use 'item1' or 'item7'.")


def to_paragraphs(section_text: str, min_words: int = 8) -> list[str]:
    """Split section text into paragraphs, dropping very short fragments."""
    # Split on blank lines
    raw = re.split(r"\n\s*\n", section_text)
    paragraphs = []
    for p in raw:
        p = " ".join(p.split())  # collapse whitespace
        if len(p.split()) >= min_words:
            paragraphs.append(p)
    return paragraphs
