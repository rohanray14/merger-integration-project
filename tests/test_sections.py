from integrability.sections import strip_html, extract, to_paragraphs, _find_section


# ── strip_html ───────────────────────────────────────────────────────────────

def test_strip_html_plain_text():
    """Plain text with SEC SGML tags passes through with tags removed."""
    raw = "<PAGE>\nHello world\n<TABLE>\nrow1\n</TABLE>"
    out = strip_html(raw)
    assert "<" not in out
    assert "Hello world" in out


def test_strip_html_real_html():
    """HTML document is converted to text."""
    raw = "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
    out = strip_html(raw)
    assert "First paragraph." in out
    assert "Second paragraph." in out
    assert "<p>" not in out


def test_strip_html_skips_script_style():
    raw = "<html><body><script>var x=1;</script><style>.a{}</style><p>Visible.</p></body></html>"
    out = strip_html(raw)
    assert "Visible." in out
    assert "var x" not in out
    assert ".a{}" not in out


# ── extract ──────────────────────────────────────────────────────────────────

FAKE_10K = """
PART I

Item 1. Business

Founded in 1990, Acme Corp is a leading provider of widgets.
We operate in three segments: domestic, international, and online.
Our products are sold in over 50 countries worldwide.
""" + ("Additional business details. " * 200) + """

Item 1A. Risk Factors

Investing in our stock involves a high degree of risk.
Market conditions may adversely affect our business.

Item 2. Properties

We maintain our headquarters in New York.

PART II

Item 7. Management's Discussion and Analysis of Financial Condition

Overview
Our revenue increased 15% year over year driven by strong demand.
We expect continued growth in the coming fiscal year.
""" + ("Further MD&A discussion. " * 200) + """

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

We are exposed to interest rate fluctuations.

Item 8. Financial Statements
"""


def test_extract_item1():
    text = strip_html(FAKE_10K)
    item1 = extract(text, "item1")
    assert item1 is not None
    assert "Acme Corp" in item1
    assert "Risk Factors" not in item1
    assert "Properties" not in item1


def test_extract_item7():
    text = strip_html(FAKE_10K)
    item7 = extract(text, "item7")
    assert item7 is not None
    assert "revenue increased" in item7
    assert "Market Risk" not in item7
    assert "Financial Statements" not in item7


def test_extract_missing_item():
    text = "This document has no standard Item headers at all."
    assert extract(text, "item1") is None
    assert extract(text, "item7") is None


def test_extract_invalid_item():
    import pytest
    with pytest.raises(ValueError):
        extract("text", "item99")


# ── TOC skipping ─────────────────────────────────────────────────────────────

FAKE_10K_WITH_TOC = """
TABLE OF CONTENTS

Item 1. Business .................. 5
Item 2. Properties ............... 20
Item 7. Management's Discussion .. 45
Item 8. Financial Statements ..... 90

PART I

Item 1. Business

Founded in 2000, BigCo makes software. We serve enterprise customers globally.
""" + ("More business content. " * 200) + """

Item 2. Properties

Our offices are in San Francisco.

PART II

Item 7. Management's Discussion and Analysis

We achieved record revenue this year due to increased subscription adoption.
""" + ("More MD&A content. " * 200) + """

Item 7A. Market Risk Disclosures
"""


def test_toc_skipped_item1():
    """TOC entry for Item 1 should be skipped, real section found."""
    text = strip_html(FAKE_10K_WITH_TOC)
    item1 = extract(text, "item1")
    assert item1 is not None
    assert "BigCo" in item1


def test_toc_skipped_item7():
    """TOC entry for Item 7 should be skipped, real section found."""
    text = strip_html(FAKE_10K_WITH_TOC)
    item7 = extract(text, "item7")
    assert item7 is not None
    assert "record revenue" in item7


# ── to_paragraphs ────────────────────────────────────────────────────────────

def test_to_paragraphs_basic():
    text = "First paragraph with enough words to pass the filter.\n\nSecond paragraph also has enough words in it.\n\nShort."
    paras = to_paragraphs(text)
    assert len(paras) == 2
    assert "First paragraph" in paras[0]
    assert "Second paragraph" in paras[1]


def test_to_paragraphs_collapses_whitespace():
    text = "  This   has   lots   of   extra   whitespace   throughout   it.  \n\n  Another   one   with   spaces   everywhere   in   it   too.  "
    paras = to_paragraphs(text)
    assert len(paras) == 2
    assert "  " not in paras[0]


def test_to_paragraphs_min_words():
    text = "One two three.\n\nThis paragraph has more than eight words in total for sure."
    paras = to_paragraphs(text, min_words=8)
    assert len(paras) == 1


def test_to_paragraphs_empty():
    assert to_paragraphs("") == []
    assert to_paragraphs("   ") == []
