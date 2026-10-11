"""Step 2a. Link acquirer and target gvkeys, then map gvkey → CIK.

Strategy:
  1. Join target SDC file (target gvkey) with acquirer SDC file (acquirer gvkey)
     on dealnum.
  2. Map each gvkey → CIK via WRDS gvkey_cik_link.
  3. Fallback: for any remaining unmatched firms, try CUSIP-6 → CIK crosswalk.

Reads  : sdc_target, sdc_acquirer, gvkey_cik_link, cik_cusip_map
Writes : data/interim/deals_with_cik.parquet
"""
import pandas as pd
from integrability.config import load
from integrability.counts import log_step


def _clean_gvkey(s: pd.Series) -> pd.Series:
    """Normalise gvkey to integer string (no decimals, no whitespace)."""
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace({"nan": None, "": None})
    )


def main():
    cfg = load()
    paths = cfg["paths"]

    # ── 1. Load target SDC (has target gvkey) ────────────────────────────
    tar = pd.read_csv(paths["sdc_target"], dtype=str)
    tar["tgt_gvkey"] = _clean_gvkey(tar["gvkey"])
    tar = tar.drop(columns=["gvkey"])
    tar["dn"] = _clean_gvkey(tar["dealnum"])
    n_in = len(tar)
    print(f"Target SDC file: {n_in} deals")

    # ── 2. Load acquirer SDC (has acquirer gvkey) ────────────────────────
    acq = pd.read_stata(paths["sdc_acquirer"], columns=["dealnum", "gvkey"])
    acq["acq_gvkey"] = _clean_gvkey(acq["gvkey"])
    acq["dn"] = acq["dealnum"].astype(int).astype(str)
    acq = acq[["dn", "acq_gvkey"]].drop_duplicates(subset="dn", keep="first")
    print(f"Acquirer SDC file: {len(acq)} deals")

    # ── 3. Merge on dealnum ──────────────────────────────────────────────
    deals = tar.merge(acq, on="dn", how="left")
    acq_gv_hit = deals["acq_gvkey"].notna().sum()
    print(f"Acquirer gvkey matched via dealnum: {acq_gv_hit}/{n_in}")

    # ── 4. Map gvkey → CIK via WRDS link ────────────────────────────────
    gvlink = pd.read_csv(paths["gvkey_cik_link"], dtype=str)
    gvlink = gvlink.rename(columns=str.lower)
    gvlink["gvkey"] = _clean_gvkey(gvlink["gvkey"])
    gvlink["cik"] = gvlink["cik"].str.replace(r"\.0$", "", regex=True).str.strip()
    gvlink = gvlink.dropna(subset=["gvkey", "cik"]).drop_duplicates(
        subset="gvkey", keep="first"
    )[["gvkey", "cik"]]
    print(f"WRDS gvkey-CIK link: {len(gvlink)} mappings")

    # acquirer CIK
    deals = deals.merge(
        gvlink.rename(columns={"gvkey": "acq_gvkey", "cik": "acq_cik"}),
        on="acq_gvkey", how="left",
    )
    # target CIK
    deals = deals.merge(
        gvlink.rename(columns={"gvkey": "tgt_gvkey", "cik": "tgt_cik"}),
        on="tgt_gvkey", how="left",
    )

    acq_cik_gv = deals["acq_cik"].notna().sum()
    tgt_cik_gv = deals["tgt_cik"].notna().sum()
    print(f"After gvkey→CIK:  acq={acq_cik_gv}  tgt={tgt_cik_gv}")

    # ── 5. Fallback: CUSIP-6 → CIK for remaining gaps ───────────────────
    cmap = pd.read_csv(paths["cik_cusip_map"], dtype=str).rename(columns=str.lower)
    cmap["cik"] = cmap["cik"].str.replace(r"\.0$", "", regex=True).str.strip()
    cmap["cusip6"] = cmap["cusip6"].str.strip().str[:6].str.upper().str.zfill(6)
    cmap = cmap.dropna(subset=["cik", "cusip6"]).drop_duplicates(
        subset="cusip6", keep="first"
    )[["cusip6", "cik"]]

    deals["acusip6"] = deals["acusip"].str.strip().str[:6].str.upper().str.zfill(6)
    deals["tcusip6"] = deals["tcusip"].str.strip().str[:6].str.upper().str.zfill(6)

    # fill acquirer gaps
    acq_gap = deals["acq_cik"].isna()
    acq_fill = deals.loc[acq_gap, "acusip6"].map(cmap.set_index("cusip6")["cik"])
    deals.loc[acq_gap, "acq_cik"] = acq_fill
    print(f"CUSIP fallback filled {acq_fill.notna().sum()} acquirer CIKs")

    # fill target gaps
    tgt_gap = deals["tgt_cik"].isna()
    tgt_fill = deals.loc[tgt_gap, "tcusip6"].map(cmap.set_index("cusip6")["cik"])
    deals.loc[tgt_gap, "tgt_cik"] = tgt_fill
    print(f"CUSIP fallback filled {tgt_fill.notna().sum()} target CIKs")

    # ── 6. Report final match rates ──────────────────────────────────────
    acq_hit = deals["acq_cik"].notna().sum()
    tgt_hit = deals["tgt_cik"].notna().sum()
    both_hit = (deals["acq_cik"].notna() & deals["tgt_cik"].notna()).sum()
    print(f"\nFinal match rates:")
    print(f"  Acquirer CIK: {acq_hit}/{n_in} ({acq_hit/n_in:.1%})")
    print(f"  Target  CIK:  {tgt_hit}/{n_in} ({tgt_hit/n_in:.1%})")
    print(f"  Both    CIK:  {both_hit}/{n_in} ({both_hit/n_in:.1%})")

    log_step("02a_link_gvkey_cik", n_in, both_hit,
             f"acq={acq_hit} tgt={tgt_hit} both={both_hit}")

    # ── 7. Save ──────────────────────────────────────────────────────────
    drop_cols = [c for c in ["dn", "acusip6", "tcusip6"] if c in deals.columns]
    deals = deals.drop(columns=drop_cols)
    out = paths["interim"] / "deals_with_cik.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    deals.to_parquet(out, index=False)
    print(f"\nSaved {len(deals)} deals → {out}")

    # ── 8. Spot-check: 30 random matched deals ──────────────────────────
    matched = deals[deals["acq_cik"].notna() & deals["tgt_cik"].notna()]
    show = ["dealnum", "aname", "acq_gvkey", "acq_cik", "tname", "tgt_gvkey", "tgt_cik"]
    cols = [c for c in show if c in matched.columns]
    sample = matched[cols].sample(n=min(30, len(matched)), random_state=0)
    print("\n── 30 random matched deals for hand-checking ──")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
