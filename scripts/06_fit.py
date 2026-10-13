"""Step 6. Penalised logistic regression per input set, 75/25 split.

Input sets (from config.yaml):
  - item1:         acquirer_item1 + target_item1
  - mdna:          acquirer_item7 + target_item7
  - both:          all four embedding vectors
  - acquirer_only: acquirer_item1 + acquirer_item7
  - acquirer_item1: acquirer_item1 only

Reads  : data/processed/embeddings/, data/processed/labels.parquet
Writes : outputs/models/, outputs/predictions.parquet
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from integrability.config import load
from integrability.counts import log_step


# Map input_set name -> which embedding columns to concatenate
INPUT_COLS = {
    "item1": ["acquirer_item1", "target_item1"],
    "mdna": ["acquirer_item7", "target_item7"],
    "both": ["acquirer_item1", "target_item1", "acquirer_item7", "target_item7"],
    "acquirer_only": ["acquirer_item1", "acquirer_item7"],
    "acquirer_item1": ["acquirer_item1"],
}


def _build_emb_indices(dealnums: np.ndarray, meta: pd.DataFrame) -> dict[str, dict]:
    """Build per-column mapping from dealnum -> row index in the npz array.

    The npz arrays are saved with dropna(), so each column has a different
    length. The order matches the non-NaN rows in metadata, which shares
    the same dealnum order as the dealnums array.
    """
    col_map = {}
    for col in ["acquirer_item1", "target_item1", "acquirer_item7", "target_item7"]:
        n_col = f"{col}_n"
        if n_col not in meta.columns:
            continue
        # Deals with a non-NaN count have an embedding, in order
        has = meta[n_col].notna()
        deal_to_idx = {}
        for arr_idx, (_, row) in enumerate(meta[has].iterrows()):
            deal_to_idx[row["dealnum"]] = arr_idx
        col_map[col] = deal_to_idx
    return col_map


def _build_features(emb_data: dict, col_indices: dict[str, dict],
                     cols: list[str], deal_set: set) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and corresponding dealnum array for deals in deal_set.

    Returns (X, filtered_dealnums). Deals missing any required column are dropped.
    """
    # Find deals that are in deal_set AND have all required columns
    valid_deals = []
    rows = []

    # Start with deals that have all columns
    candidates = deal_set.copy()
    for col in cols:
        if col not in col_indices:
            return np.empty((0, 0)), np.array([])
        candidates &= set(col_indices[col].keys())

    for dn in sorted(candidates):
        parts = []
        for col in cols:
            arr_idx = col_indices[col][dn]
            parts.append(emb_data[col][arr_idx])
        rows.append(np.concatenate(parts))
        valid_deals.append(dn)

    if not rows:
        return np.empty((0, 0)), np.array([])

    return np.stack(rows), np.array(valid_deals)


def main():
    cfg = load()
    paths = cfg["paths"]
    fit_cfg = cfg["fit"]
    seed = fit_cfg["seed"]

    # ── Load embeddings ─────────────────────────────────────────────────────
    emb_dir = paths["processed"] / "embeddings"
    emb_data = dict(np.load(emb_dir / "embeddings.npz", allow_pickle=True))
    dealnums = emb_data.pop("dealnums")
    meta = pd.read_parquet(emb_dir / "metadata.parquet")
    col_indices = _build_emb_indices(dealnums, meta)
    print(f"Loaded embeddings for {len(dealnums)} deals")
    for col, idx_map in col_indices.items():
        print(f"  {col}: {len(idx_map)} deals")

    # ── Load labels ─────────────────────────────────────────────────────────
    labels = pd.read_parquet(paths["processed"] / "labels.parquet")
    label_map = dict(zip(labels["dealnum"], labels["hit"]))
    print(f"Labels: {len(labels)} deals, {labels['hit'].sum()} positive")

    # ── Train/test split ────────────────────────────────────────────────────
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    deals["adate"] = pd.to_datetime(deals["adate"])

    # Only keep deals that have ALL four embedding columns and labels,
    # so every model is evaluated on the exact same set of deals.
    all_cols = ["acquirer_item1", "target_item1", "acquirer_item7", "target_item7"]
    valid_deals = set(dealnums) & set(labels["dealnum"])
    for col in all_cols:
        if col in col_indices:
            valid_deals &= set(col_indices[col].keys())
    deals = deals[deals["dealnum"].isin(valid_deals)].copy()
    n_in = len(deals)
    print(f"Deals with all embeddings + labels: {n_in}")

    if fit_cfg["split"] == "time":
        deals = deals.sort_values("adate")
        split_idx = int(len(deals) * (1 - fit_cfg["test_share"]))
        train_deals = set(deals.iloc[:split_idx]["dealnum"])
        test_deals = set(deals.iloc[split_idx:]["dealnum"])
        split_date = deals.iloc[split_idx]["adate"]
        print(f"Time split at {split_date}: {len(train_deals)} train, {len(test_deals)} test")
    else:
        rng = np.random.RandomState(seed)
        shuffled = deals["dealnum"].values.copy()
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - fit_cfg["test_share"]))
        train_deals = set(shuffled[:split_idx])
        test_deals = set(shuffled[split_idx:])
        print(f"Random split: {len(train_deals)} train, {len(test_deals)} test")

    # ── Fit models per input set ────────────────────────────────────────────
    model_dir = paths["outputs"] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []

    for input_set in fit_cfg["input_sets"]:
        cols = INPUT_COLS.get(input_set)
        if cols is None:
            print(f"WARNING: unknown input_set {input_set}, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Input set: {input_set} ({cols})")

        # Build X, y for train and test
        X_train, dn_train = _build_features(emb_data, col_indices, cols, train_deals)
        X_test, dn_test = _build_features(emb_data, col_indices, cols, test_deals)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping: insufficient data (train={len(X_train)}, test={len(X_test)})")
            continue

        y_train = np.array([label_map[d] for d in dn_train])
        y_test = np.array([label_map[d] for d in dn_test])

        print(f"  Train: {len(X_train)} ({y_train.sum()} pos), Test: {len(X_test)} ({y_test.sum()} pos)")
        print(f"  Feature dim: {X_train.shape[1]}")

        # Fit penalised logistic regression with cross-validation
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(
                Cs=10,
                cv=5,
                penalty="l2",
                scoring="roc_auc",
                max_iter=1000,
                random_state=seed,
            )),
        ])
        pipe.fit(X_train, y_train)

        best_C = pipe.named_steps["clf"].C_[0]
        print(f"  Best C: {best_C:.4f}")

        # Predictions
        prob_train = pipe.predict_proba(X_train)[:, 1]
        prob_test = pipe.predict_proba(X_test)[:, 1]

        for dn, prob in zip(dn_test, prob_test):
            all_preds.append({
                "dealnum": dn,
                "input_set": input_set,
                "y_true": label_map[dn],
                "y_prob": prob,
            })

        # Save model
        model_path = model_dir / f"{input_set}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"  Saved -> {model_path}")

    # ── Save predictions ────────────────────────────────────────────────────
    preds_df = pd.DataFrame(all_preds)
    preds_path = paths["outputs"] / "predictions.parquet"
    preds_df.to_parquet(preds_path, index=False)

    n_out = preds_df["dealnum"].nunique() if len(preds_df) else 0
    log_step("06_fit", n_in, n_out,
             f"input_sets={fit_cfg['input_sets']}")

    print(f"\nSaved {len(preds_df)} predictions -> {preds_path}")


if __name__ == "__main__":
    main()
