"""Step 7. Evaluate models: AUC per input set, baselines, tables and figures.

Baselines:
  - Habit only: pre-merger keyword hit as sole predictor
  - Deal controls: log(deal value), same-industry dummy
  - TF-IDF + logistic regression on pre-merger text

Reads  : outputs/predictions.parquet, data/processed/labels.parquet,
         data/interim/deals.parquet
Writes : outputs/results_table.csv, outputs/roc_curves.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from integrability.config import load
from integrability.counts import log_step


def _safe_auc(y_true, y_score) -> float:
    """Return AUC or NaN if undefined (e.g. single class)."""
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def main():
    cfg = load()
    paths = cfg["paths"]
    out_dir = paths["outputs"]

    # ── Load predictions ────────────────────────────────────────────────────
    preds = pd.read_parquet(out_dir / "predictions.parquet")
    labels = pd.read_parquet(paths["processed"] / "labels.parquet")
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    print(f"Predictions: {len(preds)} rows, {preds['input_set'].nunique()} input sets")

    # ── AUC per input set ───────────────────────────────────────────────────
    results = []
    for input_set, grp in preds.groupby("input_set"):
        auc = _safe_auc(grp["y_true"], grp["y_prob"])
        n = len(grp)
        n_pos = grp["y_true"].sum()
        results.append({
            "model": input_set,
            "auc": auc,
            "n_test": n,
            "n_pos": n_pos,
        })
        print(f"  {input_set}: AUC={auc:.3f} (n={n}, pos={n_pos})")

    # ── Baseline: habit only ────────────────────────────────────────────────
    # Use the exact same deals as the embedding models (pick any input_set;
    # they all share the same deal set after the common-sample filter in 06_fit).
    first_set = preds["input_set"].iloc[0]
    test_deals = set(preds.loc[preds["input_set"] == first_set, "dealnum"])
    test_labels = labels[labels["dealnum"].isin(test_deals)].copy()

    if "pre_habit" in test_labels.columns and len(test_labels) > 0:
        habit_auc = _safe_auc(test_labels["hit"], test_labels["pre_habit"])
        results.append({
            "model": "baseline_habit",
            "auc": habit_auc,
            "n_test": len(test_labels),
            "n_pos": test_labels["hit"].sum(),
        })
        print(f"  baseline_habit: AUC={habit_auc:.3f}")

    if "pre_intensity" in test_labels.columns and len(test_labels) > 0:
        habit_int_auc = _safe_auc(test_labels["hit"], test_labels["pre_intensity"])
        results.append({
            "model": "baseline_habit_intensity",
            "auc": habit_int_auc,
            "n_test": len(test_labels),
            "n_pos": test_labels["hit"].sum(),
        })
        print(f"  baseline_habit_intensity: AUC={habit_int_auc:.3f}")

    # ── Baseline: same-industry ─────────────────────────────────────────────
    test_deal_info = deals[deals["dealnum"].isin(test_deals)].copy()
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
                "auc": ind_auc,
                "n_test": len(merged),
                "n_pos": merged["hit"].sum(),
            })
            print(f"  baseline_same_industry: AUC={ind_auc:.3f}")

    # ── Results table ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("auc", ascending=False)
    table_path = out_dir / "results_table.csv"
    results_df.to_csv(table_path, index=False)
    print(f"\nResults table -> {table_path}")
    print(results_df.to_string(index=False))

    # ── ROC curves ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for input_set, grp in preds.groupby("input_set"):
        if len(set(grp["y_true"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(grp["y_true"], grp["y_prob"])
        auc = roc_auc_score(grp["y_true"], grp["y_prob"])
        ax.plot(fpr, tpr, label=f"{input_set} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Integration Difficulty Prediction")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    fig_path = out_dir / "roc_curves.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curves -> {fig_path}")

    n_in = preds["dealnum"].nunique()
    log_step("07_evaluate", n_in, n_in, f"models={len(results)}")


if __name__ == "__main__":
    main()
