"""Step 4b. Build announcement returns (CARs) as a second outcome measure.

Computes acquirer cumulative abnormal returns around the deal announcement
date using CRSP daily returns and the market model.

Requires: data/raw/crsp_daily.csv           (CRSP daily stock file)
          data/raw/crsp_market_index.csv     (CRSP daily market index, vwretd)
          data/raw/permno_cusip_link.csv     (PERMNO-CUSIP6 link from CRSP dsenames)

Reads  : data/interim/deals.parquet
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


def _build_acq_permno_map(cfg, deals):
    """Map dealnum -> acquirer PERMNO via acquirer CUSIP-6 + CRSP dsenames link."""
    paths = cfg["paths"]

    # Load PERMNO-CUSIP6 link
    link = pd.read_csv(paths["raw"] / "permno_cusip_link.csv")
    link["cusip6"] = link["cusip6"].astype(str).str.strip()
    link["permno"] = link["permno"].astype(int)

    # CUSIP6 can map to multiple PERMNOs; take the first (most common share class)
    cusip_to_permno = link.drop_duplicates(subset=["cusip6"]).set_index("cusip6")["permno"].to_dict()

    # Map acquirer CUSIP-6 to PERMNO
    deals = deals.copy()
    deals["acq_permno"] = deals["acusip"].astype(str).str.strip().str[:6].map(cusip_to_permno)

    return deals


def main():
    cfg = load()
    paths = cfg["paths"]

    # ── Load deals ────────────────────────────────────────────────────────────
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    deals["adate"] = pd.to_datetime(deals["adate"])
    deals["dealnum"] = deals["dealnum"].astype(float)
    n_in = len(deals)
    print(f"Loaded {n_in} deals")

    # ── Load CRSP daily returns ───────────────────────────────────────────────
    crsp = pd.read_csv(paths["raw"] / "crsp_daily.csv", parse_dates=["date"])
    crsp.columns = crsp.columns.str.lower()
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    print(f"CRSP daily returns: {len(crsp):,} rows, {crsp['permno'].nunique():,} PERMNOs")

    # ── Load market index (separate file) ─────────────────────────────────────
    mkt = pd.read_csv(paths["raw"] / "crsp_market_index.csv", parse_dates=["date"])
    mkt.columns = mkt.columns.str.lower()
    mkt["vwretd"] = pd.to_numeric(mkt["vwretd"], errors="coerce")
    mkt = mkt.rename(columns={"vwretd": "mkt_ret"})

    # Merge market return onto CRSP daily
    crsp = crsp.merge(mkt[["date", "mkt_ret"]], on="date", how="left")
    print(f"After merging market index: {crsp['mkt_ret'].notna().sum():,} rows with mkt_ret")

    # ── Map deals to acquirer PERMNOs ─────────────────────────────────────────
    deals = _build_acq_permno_map(cfg, deals)

    has_permno = deals["acq_permno"].notna()
    print(f"Deals with acquirer PERMNO: {has_permno.sum()}/{n_in}")

    # ── Compute CARs ─────────────────────────────────────────────────────────
    results = []
    crsp_grouped = crsp.groupby("permno")

    for _, deal in deals[has_permno].iterrows():
        permno = int(deal["acq_permno"])
        adate = deal["adate"]
        dealnum = deal["dealnum"]

        # Get this firm's returns
        if permno not in crsp_grouped.groups:
            continue
        firm_ret = crsp_grouped.get_group(permno).sort_values("date").copy()
        firm_ret = firm_ret.dropna(subset=["ret", "mkt_ret"])
        if len(firm_ret) == 0:
            continue

        # Compute relative trading days from announcement
        firm_dates = firm_ret["date"].values
        event_idx = np.searchsorted(firm_dates, np.datetime64(adate))
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
        print(f"  CAR [-1,+1] mean:   {returns_df['car_3day'].mean():.4f}")
        print(f"  CAR [-1,+1] median: {returns_df['car_3day'].median():.4f}")
        print(f"  CAR [-2,+2] mean:   {returns_df['car_5day'].mean():.4f}")
        print(f"  CAR [-5,+5] mean:   {returns_df['car_11day'].mean():.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = paths["processed"] / "announcement_returns.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    returns_df.to_parquet(out, index=False)
    print(f"Saved -> {out}")

    log_step("04b_build_returns", n_in, n_out, f"car_deals={n_out}")


if __name__ == "__main__":
    main()
