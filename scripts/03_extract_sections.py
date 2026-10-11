"""Step 3. Download 10-K filings from EDGAR and extract Item 1 / Item 7 sections.

Reads  : data/interim/filings.parquet
Writes : data/interim/sections.parquet
"""
import time
import requests
import pandas as pd
from integrability.config import load
from integrability.counts import log_step
from integrability.sections import strip_html, extract, to_paragraphs

HEADERS = {"User-Agent": "Rohan Ray rohan.ray.26@dartmouth.edu"}
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path}/{doc}"


def _build_url(row: pd.Series) -> str:
    """Build the EDGAR URL for a filing's primary document."""
    cik = str(row["cik"]).lstrip("0")
    acc_path = row["accessionNumber"].replace("-", "")
    doc = row["primaryDocument"]
    return ARCHIVES.format(cik=cik, acc_path=acc_path, doc=doc)


def _download(url: str) -> str | None:
    """Download a filing document from EDGAR. Returns text or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  WARNING: download failed {url}: {e}")
        return None


def main():
    cfg = load()
    paths = cfg["paths"]

    # ── Load filings ────────────────────────────────────────────────────────
    filings = pd.read_parquet(paths["interim"] / "filings.parquet")
    n_filings = len(filings)
    n_deals_in = filings["dealnum"].nunique()
    print(f"Loaded {n_filings} filings across {n_deals_in} deals")

    # Deduplicate: same accession number may appear for multiple deals
    unique_filings = filings.drop_duplicates(subset=["accessionNumber"]).copy()
    print(f"Unique filings to download: {len(unique_filings)}")

    # ── Download and extract ────────────────────────────────────────────────
    extracted = {}  # accessionNumber -> {item1: str|None, item7: str|None}
    n = len(unique_filings)

    for i, (_, row) in enumerate(unique_filings.iterrows()):
        acc = row["accessionNumber"]
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Processing {i + 1}/{n} ...")

        url = _build_url(row)
        raw = _download(url)
        if raw is None:
            extracted[acc] = {"item1": None, "item7": None}
            continue

        text = strip_html(raw)
        extracted[acc] = {
            "item1": extract(text, "item1"),
            "item7": extract(text, "item7"),
        }

        time.sleep(0.11)  # SEC rate limit

    # ── Build output rows ───────────────────────────────────────────────────
    rows = []
    for _, filing in filings.iterrows():
        acc = filing["accessionNumber"]
        sections = extracted.get(acc, {})

        for item_key in ("item1", "item7"):
            section_text = sections.get(item_key)
            if section_text is None:
                continue

            paragraphs = to_paragraphs(section_text)
            if not paragraphs:
                continue

            for para_idx, para in enumerate(paragraphs):
                rows.append({
                    "dealnum": filing["dealnum"],
                    "role": filing["role"],
                    "purpose": filing["purpose"],
                    "accessionNumber": acc,
                    "cik": filing["cik"],
                    "filingDate": filing["filingDate"],
                    "item": item_key,
                    "para_idx": para_idx,
                    "text": para,
                })

    result = pd.DataFrame(rows)
    print(f"\nExtracted {len(result)} paragraphs from {result['accessionNumber'].nunique()} filings")

    # ── Coverage stats ──────────────────────────────────────────────────────
    for purpose in ("pre", "post"):
        sub = result[result["purpose"] == purpose]
        if len(sub) == 0:
            print(f"  {purpose}: no paragraphs")
            continue
        for item in ("item1", "item7"):
            item_sub = sub[sub["item"] == item]
            n_deals = item_sub["dealnum"].nunique()
            n_paras = len(item_sub)
            print(f"  {purpose} {item}: {n_deals} deals, {n_paras} paragraphs")

    # ── Count deals with complete coverage ──────────────────────────────────
    pre = result[result["purpose"] == "pre"]
    post = result[result["purpose"] == "post"]
    deals_pre_item1 = set(pre[pre["item"] == "item1"]["dealnum"])
    deals_pre_item7 = set(pre[pre["item"] == "item7"]["dealnum"])
    deals_post = set(post["dealnum"])
    deals_complete = deals_pre_item1 & deals_pre_item7 & deals_post
    n_out = len(deals_complete)

    log_step("03_extract_sections", n_deals_in, n_out,
             f"paragraphs={len(result)}")

    # ── Save ────────────────────────────────────────────────────────────────
    out = paths["interim"] / "sections.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"\nSaved {len(result)} rows -> {out}")


if __name__ == "__main__":
    main()
