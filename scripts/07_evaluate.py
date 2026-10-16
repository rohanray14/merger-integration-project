"""Step 7. Evaluate models: AUC per model × test set, baselines, tables and figures.

Handles multiple representations (concat, cosine, diff) and multiple test sets
(test_A = in-sample holdout, test_B = out-of-sample).

Baselines:
  - Habit only: pre-merger keyword hit as sole predictor
  - Habit + intensity: pre-merger habit dummy + intensity
  - Same industry: same 2-digit SIC dummy

Reads  : outputs/predictions.parquet, data/processed/labels.parquet,
         data/interim/deals.parquet
Writes : outputs/results_table.csv, outputs/roc_curves.png,
         outputs/results_by_test_set.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
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

    # Habit baseline
    if "pre_habit" in test_labels.columns and len(test_labels) > 0:
        habit_auc = _safe_auc(test_labels["hit"], test_labels["pre_habit"])
        results.append({
            "model": "baseline_habit",
            "test_set": test_name,
            "auc": habit_auc,
            "n_test": len(test_labels),
            "n_pos": int(test_labels["hit"].sum()),
        })

    # Habit intensity baseline
    if "pre_intensity" in test_labels.columns and len(test_labels) > 0:
        habit_int_auc = _safe_auc(test_labels["hit"], test_labels["pre_intensity"])
        results.append({
            "model": "baseline_habit_intensity",
            "test_set": test_name,
            "auc": habit_int_auc,
            "n_test": len(test_labels),
            "n_pos": int(test_labels["hit"].sum()),
        })

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

    n_in = preds["dealnum"].nunique()
    log_step("07_evaluate", n_in, n_in, f"models={len(model_names)} test_sets={len(test_set_names)}")


if __name__ == "__main__":
    main()
