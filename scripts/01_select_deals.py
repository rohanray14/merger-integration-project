"""Step 1. Keep completed public-public SDC deals announced in sample.announce_years.

Reads  : data/interim/deals_with_cik.parquet  (output of 02a)
Writes : data/interim/deals.parquet
"""
import pandas as pd
from integrability.config import load
from integrability.counts import log_step


def main():
    cfg = load()
    paths = cfg["paths"]
    sample = cfg["sample"]

    # ── Load deals with CIK mappings ──────────────────────────────────────
    df = pd.read_parquet(paths["interim"] / "deals_with_cik.parquet")
    n_in = len(df)
    print(f"Loaded {n_in} deals from deals_with_cik.parquet")

    # ── Filter: completed deals ───────────────────────────────────────────
    df = df[df["status"] == sample["status"]]
    print(f"After status={sample['status']}: {len(df)}")

    # ── Filter: announcement year in pilot range ──────────────────────────
    df["ayear_int"] = df["ayear"].astype(float).astype(int)
    df = df[df["ayear_int"].isin(sample["announce_years"])]
    print(f"After announce_years {sample['announce_years']}: {len(df)}")

    # ── Filter: both firms US ─────────────────────────────────────────────
    df = df[(df["anation"] == "US") & (df["tnation"] == "US")]
    print(f"After both US: {len(df)}")

    # ── Filter: both firms have CIK (proxy for public SEC filers) ─────────
    df = df[df["acq_cik"].notna() & df["tgt_cik"].notna()]
    print(f"After both have CIK: {len(df)}")

    n_out = len(df)
    log_step("01_select_deals", n_in, n_out,
             f"status={sample['status']} years={sample['announce_years']} both_US both_CIK")

    # ── Save ──────────────────────────────────────────────────────────────
    df = df.drop(columns=["ayear_int"])
    out = paths["interim"] / "deals.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nSaved {n_out} pilot deals -> {out}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nBy announcement year:")
    print(df["ayear"].astype(float).astype(int).value_counts().sort_index().to_string())
    print(f"\nBy deal form:")
    print(df["form"].value_counts().to_string())


if __name__ == "__main__":
    main()
