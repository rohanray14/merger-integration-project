"""Step 2. For each pilot deal, find 10-K filings on EDGAR via CIK.

Pre-merger (model inputs):
  - Acquirer's most recent 10-K filed strictly before announcement date
  - Target's most recent 10-K filed strictly before announcement date

Post-merger (labels):
  - Acquirer's 10-Ks filed 1..post_years years after completion date

Reads  : data/interim/deals.parquet
Writes : data/interim/filings.parquet
"""
import time
import requests
import pandas as pd
from integrability.config import load
from integrability.counts import log_step

EDGAR_SUB_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
HEADERS = {"User-Agent": "Rohan Ray rohan.ray.26@dartmouth.edu"}
FORM_TYPES = {"10-K", "10-K405", "10-KSB", "10-KSB40"}


def _get_10k_filings(cik: str) -> pd.DataFrame:
    """Fetch all 10-K-family filings for a CIK from EDGAR submissions API.

    Fetches both the 'recent' block and any older filing history files.
    """
    padded = cik.zfill(10)
    url = EDGAR_SUB_URL.format(cik=padded)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    company = data.get("name", "")

    # Collect filings from 'recent' block
    all_forms, all_dates, all_accns, all_docs = [], [], [], []

    recent = data["filings"]["recent"]
    all_forms.extend(recent["form"])
    all_dates.extend(recent["filingDate"])
    all_accns.extend(recent["accessionNumber"])
    all_docs.extend(recent["primaryDocument"])

    # Fetch older filing history files
    for older in data["filings"].get("files", []):
        older_url = f"https://data.sec.gov/submissions/{older['name']}"
        time.sleep(0.11)
        r2 = requests.get(older_url, headers=HEADERS, timeout=30)
        r2.raise_for_status()
        old = r2.json()
        all_forms.extend(old["form"])
        all_dates.extend(old["filingDate"])
        all_accns.extend(old["accessionNumber"])
        all_docs.extend(old["primaryDocument"])

    df = pd.DataFrame({
        "form": all_forms,
        "filingDate": all_dates,
        "accessionNumber": all_accns,
        "primaryDocument": all_docs,
    })
    df = df[df["form"].isin(FORM_TYPES)].copy()
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["cik"] = cik
    df["company"] = company
    return df


def _fetch_all_filings(ciks: list[str]) -> pd.DataFrame:
    """Query EDGAR for all unique CIKs, with rate limiting."""
    all_frames = []
    n = len(ciks)
    for i, cik in enumerate(ciks):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Querying EDGAR: {i + 1}/{n} ...")
        try:
            df = _get_10k_filings(cik)
            all_frames.append(df)
        except Exception as e:
            print(f"  WARNING: CIK {cik} failed: {e}")
        time.sleep(0.11)  # SEC rate limit: ~10 req/s

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def main():
    cfg = load()
    paths = cfg["paths"]
    post_years = cfg["label"]["post_years"]

    # ── Load pilot deals ──────────────────────────────────────────────────
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    deals["adate"] = pd.to_datetime(deals["adate"])
    deals["edate"] = pd.to_datetime(deals["edate"])
    n_in = len(deals)
    print(f"Loaded {n_in} pilot deals")

    # ── Collect all unique CIKs ───────────────────────────────────────────
    acq_ciks = set(deals["acq_cik"].dropna().unique())
    tgt_ciks = set(deals["tgt_cik"].dropna().unique())
    all_ciks = sorted(acq_ciks | tgt_ciks)
    print(f"Unique CIKs to query: {len(all_ciks)}")

    # ── Fetch 10-K filing metadata from EDGAR ─────────────────────────────
    print("Fetching filing metadata from EDGAR...")
    filings_db = _fetch_all_filings(all_ciks)
    print(f"Total 10-K filings retrieved: {len(filings_db)}")

    # ── Match pre-merger filings ──────────────────────────────────────────
    pre_rows = []
    post_rows = []

    for _, deal in deals.iterrows():
        deal_id = deal["dealnum"]
        adate = deal["adate"]
        edate = deal["edate"]

        # --- Pre-merger: acquirer's latest 10-K before announcement ---
        acq_filings = filings_db[
            (filings_db["cik"] == deal["acq_cik"])
            & (filings_db["filingDate"] < adate)
        ].sort_values("filingDate", ascending=False)

        if len(acq_filings) > 0:
            row = acq_filings.iloc[0].to_dict()
            row.update({"dealnum": deal_id, "role": "acquirer", "purpose": "pre"})
            pre_rows.append(row)

        # --- Pre-merger: target's latest 10-K before announcement ---
        tgt_filings = filings_db[
            (filings_db["cik"] == deal["tgt_cik"])
            & (filings_db["filingDate"] < adate)
        ].sort_values("filingDate", ascending=False)

        if len(tgt_filings) > 0:
            row = tgt_filings.iloc[0].to_dict()
            row.update({"dealnum": deal_id, "role": "target", "purpose": "pre"})
            pre_rows.append(row)

        # --- Post-merger: acquirer's 10-Ks in 1..post_years after completion ---
        window_start = edate + pd.DateOffset(years=1)
        window_end = edate + pd.DateOffset(years=post_years)
        acq_post = filings_db[
            (filings_db["cik"] == deal["acq_cik"])
            & (filings_db["filingDate"] >= window_start)
            & (filings_db["filingDate"] <= window_end)
        ]
        for _, f in acq_post.iterrows():
            row = f.to_dict()
            row.update({"dealnum": deal_id, "role": "acquirer", "purpose": "post"})
            post_rows.append(row)

    pre = pd.DataFrame(pre_rows)
    post = pd.DataFrame(post_rows)
    filings = pd.concat([pre, post], ignore_index=True)
    print(f"\nMatched filings:")
    print(f"  Pre-merger acquirer: {len(pre[pre['role']=='acquirer']) if len(pre) else 0}")
    print(f"  Pre-merger target:   {len(pre[pre['role']=='target']) if len(pre) else 0}")
    print(f"  Post-merger:         {len(post)}")

    # ── Coverage stats ────────────────────────────────────────────────────
    deals_with_pre_acq = pre[pre["role"] == "acquirer"]["dealnum"].nunique() if len(pre) else 0
    deals_with_pre_tgt = pre[pre["role"] == "target"]["dealnum"].nunique() if len(pre) else 0
    deals_with_post = post["dealnum"].nunique() if len(post) else 0
    deals_with_all = set()
    if len(pre) and len(post):
        has_acq = set(pre[pre["role"] == "acquirer"]["dealnum"])
        has_tgt = set(pre[pre["role"] == "target"]["dealnum"])
        has_post = set(post["dealnum"])
        deals_with_all = has_acq & has_tgt & has_post

    n_out = len(deals_with_all)
    print(f"\n  Deals with pre-merger acquirer 10-K: {deals_with_pre_acq}/{n_in}")
    print(f"  Deals with pre-merger target 10-K:   {deals_with_pre_tgt}/{n_in}")
    print(f"  Deals with post-merger 10-K(s):      {deals_with_post}/{n_in}")
    print(f"  Deals with ALL three:                {n_out}/{n_in}")

    log_step("02_link_filings", n_in, n_out,
             f"pre_acq={deals_with_pre_acq} pre_tgt={deals_with_pre_tgt} post={deals_with_post}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = paths["interim"] / "filings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    filings.to_parquet(out, index=False)
    print(f"\nSaved {len(filings)} filing rows -> {out}")


if __name__ == "__main__":
    main()
