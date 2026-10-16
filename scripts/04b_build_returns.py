"""Step 4b. Build announcement returns (CARs) as a second outcome measure.

Computes acquirer cumulative abnormal returns around the deal announcement
date using CRSP daily returns and the market model.

Requires: CRSP daily returns file in data/raw/crsp_daily.csv
          (columns: PERMNO, date, RET, VWRETD or similar)
          + a PERMNO-CIK or PERMNO-GVKEY link

Reads  : data/interim/deals.parquet, data/raw/crsp_daily.csv
Writes : data/processed/announcement_returns.parquet
"""
import pandas as pd
import numpy as np
from integrability.config import load
from integrability.counts import log_step


# Event window for CARs (trading days relative to announcement)
EVENT_WINDOW = (-1, 1)       # 3-day window: [-1, +1]
ESTIMATION_WINDOW = (-250, -30)  # market model estimation period


def _compute_car(ret_series, mkt_series, alpha, beta):
    """Compute CAR given returns, market returns, and model params."""
    expected = alpha + beta * mkt_series
    abnormal = ret_series - expected
    return abnormal.sum()


def _estimate_market_model(ret_df, est_start, est_end):
    """Estimate market model (alpha, beta) over estimation window."""
    est = ret_df[(ret_df["rel_day"] >= est_start) & (ret_df["rel_day"] <= est_end)].copy()
    if len(est) < 30:
        return None, None
    X = est["mkt_ret"].values
    y = est["ret"].values
    # OLS: y = alpha + beta * X
    X_mat = np.column_stack([np.ones(len(X)), X])
    try:
        params = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        return params[0], params[1]  # alpha, beta
    except np.linalg.LinAlgError:
        return None, None


def main():
    cfg = load()
    paths = cfg["paths"]

    # ── Load deals ────────────────────────────────────────────────────────────
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    deals["adate"] = pd.to_datetime(deals["adate"])
    n_in = len(deals)
    print(f"Loaded {n_in} deals")

    # ── Load CRSP daily returns ───────────────────────────────────────────────
    crsp_path = paths["raw"] / "crsp_daily.csv"
    if not crsp_path.exists():
        print(f"ERROR: CRSP daily returns file not found at {crsp_path}")
        print("Please download from WRDS and place at this path.")
        print("Required columns: PERMNO, date, RET, VWRETD (or sprtrn)")
        print("\nCreating placeholder output with no data...")

        # Save empty placeholder so downstream scripts don't break
        placeholder = pd.DataFrame(columns=[
            "dealnum", "car_3day", "car_5day", "car_11day",
            "alpha", "beta", "n_est_days",
        ])
        out = paths["processed"] / "announcement_returns.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        placeholder.to_parquet(out, index=False)
        log_step("04b_build_returns", n_in, 0, "no CRSP data")
        return

    crsp = pd.read_csv(crsp_path, parse_dates=["date"])
    # Normalize column names
    crsp.columns = crsp.columns.str.lower()
    if "vwretd" in crsp.columns:
        crsp = crsp.rename(columns={"vwretd": "mkt_ret"})
    elif "sprtrn" in crsp.columns:
        crsp = crsp.rename(columns={"sprtrn": "mkt_ret"})

    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp["mkt_ret"] = pd.to_numeric(crsp["mkt_ret"], errors="coerce")
    print(f"CRSP daily returns: {len(crsp)} rows, {crsp['permno'].nunique()} PERMNOs")

    # ── Need PERMNO-CIK link (or PERMNO-GVKEY) ───────────────────────────────
    # Try loading a link file
    link_path = paths["raw"] / "permno_cik_link.csv"
    if not link_path.exists():
        link_path = paths["raw"] / "permno_gvkey_link.csv"

    if link_path.exists():
        link = pd.read_csv(link_path)
        link.columns = link.columns.str.lower()
        print(f"Loaded PERMNO link: {len(link)} rows")
    else:
        print(f"WARNING: No PERMNO link file found. Cannot compute CARs.")
        print("  Expected: data/raw/permno_cik_link.csv or permno_gvkey_link.csv")
        placeholder = pd.DataFrame(columns=[
            "dealnum", "car_3day", "car_5day", "car_11day",
            "alpha", "beta", "n_est_days",
        ])
        out = paths["processed"] / "announcement_returns.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        placeholder.to_parquet(out, index=False)
        log_step("04b_build_returns", n_in, 0, "no PERMNO link")
        return

    # ── Map deals to PERMNOs ──────────────────────────────────────────────────
    # This depends on your link file format — adjust as needed
    if "cik" in link.columns and "permno" in link.columns:
        cik_to_permno = dict(zip(link["cik"].astype(str), link["permno"]))
        deals["acq_permno"] = deals["acq_cik"].astype(str).map(cik_to_permno)
    elif "gvkey" in link.columns and "permno" in link.columns:
        gvkey_to_permno = dict(zip(link["gvkey"].astype(str), link["permno"]))
        if "acq_gvkey" in deals.columns:
            deals["acq_permno"] = deals["acq_gvkey"].astype(str).map(gvkey_to_permno)
        else:
            print("WARNING: Cannot map GVKEY to PERMNO — no acq_gvkey column")
            deals["acq_permno"] = np.nan

    has_permno = deals["acq_permno"].notna()
    print(f"Deals with acquirer PERMNO: {has_permno.sum()}/{n_in}")

    # ── Compute CARs ─────────────────────────────────────────────────────────
    results = []
    trading_dates = sorted(crsp["date"].unique())

    for _, deal in deals[has_permno].iterrows():
        permno = int(deal["acq_permno"])
        adate = deal["adate"]
        dealnum = deal["dealnum"]

        # Get this firm's returns
        firm_ret = crsp[crsp["permno"] == permno].sort_values("date").copy()
        if len(firm_ret) == 0:
            continue

        # Find the announcement date in trading dates
        firm_ret["rel_day"] = np.searchsorted(
            firm_ret["date"].values, adate
        ) - np.arange(len(firm_ret))
        # Simpler: compute relative trading days from announcement
        firm_dates = firm_ret["date"].values
        try:
            event_idx = np.searchsorted(firm_dates, np.datetime64(adate))
        except:
            continue

        firm_ret["rel_day"] = np.arange(len(firm_ret)) - event_idx

        # Estimate market model
        alpha, beta = _estimate_market_model(
            firm_ret, ESTIMATION_WINDOW[0], ESTIMATION_WINDOW[1]
        )
        if alpha is None:
            continue

        n_est = len(firm_ret[
            (firm_ret["rel_day"] >= ESTIMATION_WINDOW[0])
            & (firm_ret["rel_day"] <= ESTIMATION_WINDOW[1])
        ])

        # Compute CARs for different windows
        cars = {}
        for window_name, (w_start, w_end) in [
            ("car_3day", (-1, 1)),
            ("car_5day", (-2, 2)),
            ("car_11day", (-5, 5)),
        ]:
            event = firm_ret[
                (firm_ret["rel_day"] >= w_start)
                & (firm_ret["rel_day"] <= w_end)
            ]
            if len(event) == 0:
                cars[window_name] = np.nan
            else:
                cars[window_name] = _compute_car(
                    event["ret"].values, event["mkt_ret"].values, alpha, beta
                )

        results.append({
            "dealnum": dealnum,
            "alpha": alpha,
            "beta": beta,
            "n_est_days": n_est,
            **cars,
        })

    returns_df = pd.DataFrame(results)
    n_out = len(returns_df)
    print(f"\nComputed CARs for {n_out} deals")

    if n_out > 0:
        print(f"  CAR [-1,+1] mean: {returns_df['car_3day'].mean():.4f}")
        print(f"  CAR [-1,+1] median: {returns_df['car_3day'].median():.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = paths["processed"] / "announcement_returns.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    returns_df.to_parquet(out, index=False)
    print(f"Saved -> {out}")

    log_step("04b_build_returns", n_in, n_out, f"car_deals={n_out}")


if __name__ == "__main__":
    main()
