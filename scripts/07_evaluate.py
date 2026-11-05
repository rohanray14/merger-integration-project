"""Step 7. Evaluate models: AUC per model × test set, CARs regression.

Keyword label evaluation:
  - AUC per model × test set
  - Same-industry baseline
  - ROC curves

CARs regressions (second outcome measure):
  Original:  CAR = β₁ × skill + β₂ × cosine_sim + controls
  Decomposed: CAR = β₁ × acquirer_skill + β₂ × deal_difficulty
                   + β₃ × skill×difficulty + controls
  (uses outputs from 08_skill_difficulty.py when available)

Reads  : outputs/predictions.parquet, outputs/cosine_similarities.parquet,
         outputs/skill_difficulty.parquet (optional),
         data/processed/labels.parquet, data/processed/announcement_returns.parquet,
         data/interim/deals.parquet
Writes : outputs/results_table.csv, outputs/roc_curves.png,
         outputs/results_by_test_set.csv, outputs/results_cars_regression.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm
from integrability.config import load
from integrability.counts import log_step


def _safe_auc(y_true, y_score) -> float:
    """Return AUC or NaN if undefined (e.g. single class)."""
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _compute_baselines(labels, deals, test_deal_set, test_name):
    """Compute baseline AUCs for a given test set."""
    test_labels = labels[labels["dealnum"].isin(test_deal_set)].copy()
    test_deal_info = deals[deals["dealnum"].isin(test_deal_set)].copy()
    results = []

    # Same-industry baseline
    if "asic3" in test_deal_info.columns and "tsic3" in test_deal_info.columns:
        test_deal_info["same_sic3"] = (
            test_deal_info["asic3"] == test_deal_info["tsic3"]
        ).astype(int)
        merged = test_deal_info.merge(
            test_labels[["dealnum", "hit"]], on="dealnum"
        )
        if len(merged) > 0:
            ind_auc = _safe_auc(merged["hit"], merged["same_sic3"])
            results.append({
                "model": "baseline_same_industry",
                "test_set": test_name,
                "auc": ind_auc,
                "n_test": len(merged),
                "n_pos": int(merged["hit"].sum()),
            })

    return results


def main():
    cfg = load()
    paths = cfg["paths"]
    out_dir = paths["outputs"]

    # ── Load data ────────────────────────────────────────────────────────────
    preds = pd.read_parquet(out_dir / "predictions.parquet")
    labels = pd.read_parquet(paths["processed"] / "labels.parquet")
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")

    test_set_names = sorted(preds["test_set"].unique())
    model_names = sorted(preds["model"].unique())
    print(f"Predictions: {len(preds)} rows")
    print(f"  Models: {model_names}")
    print(f"  Test sets: {test_set_names}")

    # ── AUC per model × test set ─────────────────────────────────────────────
    results = []
    for test_name in test_set_names:
        test_preds = preds[preds["test_set"] == test_name]

        for model_name, grp in test_preds.groupby("model"):
            auc = _safe_auc(grp["y_true"], grp["y_prob"])
            results.append({
                "model": model_name,
                "test_set": test_name,
                "auc": auc,
                "n_test": len(grp),
                "n_pos": int(grp["y_true"].sum()),
            })
            print(f"  {model_name} [{test_name}]: AUC={auc:.3f} (n={len(grp)}, pos={grp['y_true'].sum()})")

        # ── Baselines for this test set ──────────────────────────────────────
        test_deal_set = set(test_preds["dealnum"])
        baseline_results = _compute_baselines(labels, deals, test_deal_set, test_name)
        results.extend(baseline_results)
        for br in baseline_results:
            print(f"  {br['model']} [{test_name}]: AUC={br['auc']:.3f}")

    # ── Results table ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["test_set", "auc"], ascending=[True, False])

    # Full results by test set
    full_path = out_dir / "results_by_test_set.csv"
    results_df.to_csv(full_path, index=False)
    print(f"\nFull results -> {full_path}")

    # Aggregate results table (average AUC across test sets for each model)
    agg = results_df.groupby("model").agg(
        mean_auc=("auc", "mean"),
        total_test=("n_test", "sum"),
        total_pos=("n_pos", "sum"),
    ).sort_values("mean_auc", ascending=False).reset_index()
    agg_path = out_dir / "results_table.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Aggregate results -> {agg_path}")
    print(agg.to_string(index=False))

    # ── ROC curves (one figure per test set) ─────────────────────────────────
    for test_name in test_set_names:
        test_preds = preds[preds["test_set"] == test_name]
        fig, ax = plt.subplots(figsize=(8, 6))

        for model_name, grp in test_preds.groupby("model"):
            if len(set(grp["y_true"])) < 2:
                continue
            fpr, tpr, _ = roc_curve(grp["y_true"], grp["y_prob"])
            auc = roc_auc_score(grp["y_true"], grp["y_prob"])
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves: {test_name}")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal")

        fig_path = out_dir / f"roc_curves_{test_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"ROC curves ({test_name}) -> {fig_path}")

    # ── CARs regression: CAR = β₁×skill + β₂×cosine_sim + controls ─────────
    cars_path = paths["processed"] / "announcement_returns.parquet"
    cos_path = out_dir / "cosine_similarities.parquet"

    if cars_path.exists() and cos_path.exists():
        cars = pd.read_parquet(cars_path)
        # Normalize all dealnums to same string format as predictions
        def _norm_dn(x):
            return str(x) if isinstance(x, str) else str(x)

        pred_dns = set(preds["dealnum"].unique())
        # Determine format from predictions (could be "12345.0" or "12345")
        sample_dn = next(iter(pred_dns))
        print(f"  Prediction dealnum format sample: '{sample_dn}'")

        cars["dealnum"] = cars["dealnum"].astype(float).astype(str)
        cars = cars.dropna(subset=["car_3day"])
        cars_map = dict(zip(cars["dealnum"], cars["car_3day"]))

        cos_df = pd.read_parquet(cos_path)
        # Ensure cosine sim dealnums match prediction format
        cos_df["dealnum"] = cos_df["dealnum"].astype(str)
        cos_item1 = cos_df[cos_df["section"] == "item1"].set_index("dealnum")["cosine_sim"].to_dict()

        # Build control variables from deals, matching prediction dealnum format
        deals["dealnum_norm"] = deals["dealnum"].astype(str)
        deals["log_deal_value"] = pd.to_numeric(deals["transvalue"], errors="coerce").apply(
            lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
        deals["same_industry"] = (deals["asic3"].astype(float) == deals["tsic3"].astype(float)).astype(int)
        deals["pct_cash"] = pd.to_numeric(deals["pcash"], errors="coerce").fillna(0)
        controls_map = {}
        for _, row in deals.iterrows():
            controls_map[row["dealnum_norm"]] = {
                "log_deal_value": row["log_deal_value"],
                "same_industry": row["same_industry"],
                "pct_cash": row["pct_cash"],
            }

        print(f"\n{'#'*60}")
        print(f"CARs regression: {len(cars_map)} deals with CARs, {len(cos_item1)} with cosine sim")

        car_reg_results = []
        for test_name in test_set_names:
            test_preds = preds[preds["test_set"] == test_name]

            for model_name, grp in test_preds.groupby("model"):
                reg_rows = []
                for _, row in grp.iterrows():
                    dn = row["dealnum"]
                    if dn in cars_map and dn in cos_item1 and dn in controls_map:
                        reg_rows.append({
                            "car": cars_map[dn],
                            "skill": 1 - row["y_prob"],
                            "cosine_sim": cos_item1[dn],
                            **controls_map[dn],
                        })

                if len(reg_rows) < 10:
                    continue

                reg_df = pd.DataFrame(reg_rows).dropna()
                if len(reg_df) < 10:
                    continue

                feature_cols = ["skill", "cosine_sim", "log_deal_value",
                                "same_industry", "pct_cash"]
                X = sm.add_constant(reg_df[feature_cols])
                y = reg_df["car"]

                ols = sm.OLS(y, X).fit()

                car_reg_results.append({
                    "model": model_name,
                    "test_set": test_name,
                    "n": len(reg_df),
                    "r2": ols.rsquared,
                    "beta_skill": ols.params["skill"],
                    "t_skill": ols.tvalues["skill"],
                    "p_skill": ols.pvalues["skill"],
                    "beta_cosine": ols.params["cosine_sim"],
                    "t_cosine": ols.tvalues["cosine_sim"],
                    "p_cosine": ols.pvalues["cosine_sim"],
                    "beta_log_dv": ols.params["log_deal_value"],
                    "t_log_dv": ols.tvalues["log_deal_value"],
                    "beta_same_ind": ols.params["same_industry"],
                    "t_same_ind": ols.tvalues["same_industry"],
                    "beta_pct_cash": ols.params["pct_cash"],
                    "t_pct_cash": ols.tvalues["pct_cash"],
                })
                print(f"  {model_name} [{test_name}]: n={len(reg_df)}, "
                      f"β_skill={ols.params['skill']:.4f} (t={ols.tvalues['skill']:.2f}), "
                      f"β_cosine={ols.params['cosine_sim']:.4f} (t={ols.tvalues['cosine_sim']:.2f}), "
                      f"R²={ols.rsquared:.4f}")

        if car_reg_results:
            car_reg_df = pd.DataFrame(car_reg_results)
            car_reg_path = out_dir / "results_cars_regression.csv"
            car_reg_df.to_csv(car_reg_path, index=False)
            print(f"\nCARs regression results -> {car_reg_path}")
    else:
        if not cars_path.exists():
            print("\nSkipping CARs regression: announcement_returns.parquet not found")
        if not cos_path.exists():
            print("\nSkipping CARs regression: cosine_similarities.parquet not found")

    # ── Decomposed CARs: skill vs difficulty (from 08_skill_difficulty) ───
    sd_path = out_dir / "skill_difficulty.parquet"
    if cars_path.exists() and sd_path.exists():
        print(f"\n{'#'*60}")
        print("Decomposed CARs regression: skill + difficulty + interaction")

        sd = pd.read_parquet(sd_path)
        sd["dealnum"] = sd["dealnum"].astype(str)

        # Reload cars if not already loaded
        if "cars_map" not in dir():
            cars = pd.read_parquet(cars_path)
            cars["dealnum"] = cars["dealnum"].astype(float).astype(str)
            cars = cars.dropna(subset=["car_3day"])
            cars_map = dict(zip(cars["dealnum"], cars["car_3day"]))

        # Build controls if not already built
        if "controls_map" not in dir():
            deals["dealnum_norm"] = deals["dealnum"].astype(str)
            deals["log_deal_value"] = pd.to_numeric(deals["transvalue"], errors="coerce").apply(
                lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
            deals["same_industry"] = (deals["asic3"].astype(float) == deals["tsic3"].astype(float)).astype(int)
            deals["pct_cash"] = pd.to_numeric(deals["pcash"], errors="coerce").fillna(0)
            controls_map = {}
            for _, row in deals.iterrows():
                controls_map[row["dealnum_norm"]] = {
                    "log_deal_value": row["log_deal_value"],
                    "same_industry": row["same_industry"],
                    "pct_cash": row["pct_cash"],
                }

        decomp_results = []
        for test_name in ["test_A", "test_B"]:
            sd_test = sd[(sd["test_set"] == test_name) & sd["acquirer_skill"].notna()]

            reg_rows = []
            for _, row in sd_test.iterrows():
                dn = row["dealnum"]
                if dn in cars_map and dn in controls_map:
                    reg_rows.append({
                        "car": cars_map[dn],
                        "acquirer_skill": row["acquirer_skill"],
                        "deal_difficulty": row["difficulty"],
                        "skill_x_difficulty": row["skill_x_difficulty"],
                        **controls_map[dn],
                    })

            if len(reg_rows) < 10:
                print(f"  {test_name}: skipping (n={len(reg_rows)} < 10)")
                continue

            reg_df = pd.DataFrame(reg_rows).dropna()
            if len(reg_df) < 10:
                print(f"  {test_name}: skipping after dropna (n={len(reg_df)} < 10)")
                continue

            # Model 1: skill + difficulty + controls (no interaction)
            cols_no_int = ["acquirer_skill", "deal_difficulty",
                           "log_deal_value", "same_industry", "pct_cash"]
            X1 = sm.add_constant(reg_df[cols_no_int])
            ols1 = sm.OLS(reg_df["car"], X1).fit()

            # Model 2: skill + difficulty + interaction + controls
            cols_int = ["acquirer_skill", "deal_difficulty", "skill_x_difficulty",
                        "log_deal_value", "same_industry", "pct_cash"]
            X2 = sm.add_constant(reg_df[cols_int])
            ols2 = sm.OLS(reg_df["car"], X2).fit()

            for label, ols, has_interaction in [
                ("decomp", ols1, False),
                ("decomp_interaction", ols2, True),
            ]:
                row_out = {
                    "model": label,
                    "test_set": test_name,
                    "n": len(reg_df),
                    "r2": ols.rsquared,
                    "beta_acq_skill": ols.params["acquirer_skill"],
                    "t_acq_skill": ols.tvalues["acquirer_skill"],
                    "p_acq_skill": ols.pvalues["acquirer_skill"],
                    "beta_difficulty": ols.params["deal_difficulty"],
                    "t_difficulty": ols.tvalues["deal_difficulty"],
                    "p_difficulty": ols.pvalues["deal_difficulty"],
                }
                if has_interaction:
                    row_out["beta_interaction"] = ols.params["skill_x_difficulty"]
                    row_out["t_interaction"] = ols.tvalues["skill_x_difficulty"]
                    row_out["p_interaction"] = ols.pvalues["skill_x_difficulty"]
                decomp_results.append(row_out)

            print(f"  {test_name} (n={len(reg_df)}):")
            print(f"    No interaction: β_skill={ols1.params['acquirer_skill']:.4f} "
                  f"(t={ols1.tvalues['acquirer_skill']:.2f}), "
                  f"β_diff={ols1.params['deal_difficulty']:.4f} "
                  f"(t={ols1.tvalues['deal_difficulty']:.2f}), R²={ols1.rsquared:.4f}")
            print(f"    With interaction: β_skill={ols2.params['acquirer_skill']:.4f} "
                  f"(t={ols2.tvalues['acquirer_skill']:.2f}), "
                  f"β_diff={ols2.params['deal_difficulty']:.4f} "
                  f"(t={ols2.tvalues['deal_difficulty']:.2f}), "
                  f"β_int={ols2.params['skill_x_difficulty']:.4f} "
                  f"(t={ols2.tvalues['skill_x_difficulty']:.2f}), R²={ols2.rsquared:.4f}")
            print(f"\n    Full OLS summary (with interaction):\n{ols2.summary()}")

        if decomp_results:
            decomp_df = pd.DataFrame(decomp_results)
            decomp_path = out_dir / "results_cars_decomposed.csv"
            decomp_df.to_csv(decomp_path, index=False)
            print(f"\nDecomposed CARs results -> {decomp_path}")
    elif not sd_path.exists():
        print(f"\nSkipping decomposed CARs: run 08_skill_difficulty.py first")

    n_in = preds["dealnum"].nunique()
    log_step("07_evaluate", n_in, n_in, f"models={len(model_names)} test_sets={len(test_set_names)}")


if __name__ == "__main__":
    main()
