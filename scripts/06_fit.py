"""Step 6. Penalised logistic regression per input set × representation.

Input sets (from config.yaml):
  - item1:         acquirer_item1 + target_item1
  - mdna:          acquirer_item7 + target_item7
  - both:          all four embedding vectors
  - acquirer_only: acquirer_item1 + acquirer_item7
  - acquirer_item1: acquirer_item1 only

Representations (from config.yaml):
  - concat:    concatenate acquirer + target vectors (e.g. 1536 features)
  - abs_diff:  element-wise |acquirer - target| (e.g. 768 features)
  - product:   element-wise acquirer * target (e.g. 768 features)

Split modes:
  - two_test_sets: 75% random from train_years -> train,
                   25% from train_years -> test_A,
                   matched-size random from test_oos_years -> test_B

Also computes per-deal cosine similarity (acquirer vs target) for use
in the CARs regression in step 7.

Reads  : data/processed/embeddings/, data/processed/labels.parquet
Writes : outputs/models/, outputs/predictions.parquet,
         outputs/cosine_similarities.parquet
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from integrability.config import load
from integrability.counts import log_step


# Map input_set name -> which embedding columns to use
# For concat: these are concatenated. For cosine/diff: first half = acquirer, second = target.
INPUT_COLS = {
    "item1": ["acquirer_item1", "target_item1"],
    "mdna": ["acquirer_item7", "target_item7"],
    "both": ["acquirer_item1", "target_item1", "acquirer_item7", "target_item7"],
    "acquirer_only": ["acquirer_item1", "acquirer_item7"],
    "acquirer_item1": ["acquirer_item1"],
}

# Which input sets support pair-level representations (cosine, diff)
# Only sets with exactly one acquirer and one target column of the same section
PAIR_COLS = {
    "item1": ("acquirer_item1", "target_item1"),
    "mdna": ("acquirer_item7", "target_item7"),
}


def _build_emb_indices(dealnums: np.ndarray, meta: pd.DataFrame) -> dict[str, dict]:
    """Build per-column mapping from dealnum -> row index in the npz array."""
    col_map = {}
    for col in ["acquirer_item1", "target_item1", "acquirer_item7", "target_item7"]:
        n_col = f"{col}_n"
        if n_col not in meta.columns:
            continue
        has = meta[n_col].notna()
        deal_to_idx = {}
        for arr_idx, (_, row) in enumerate(meta[has].iterrows()):
            deal_to_idx[row["dealnum"]] = arr_idx
        col_map[col] = deal_to_idx
    return col_map


def _get_embedding(emb_data, col_indices, col, dealnum):
    """Get a single embedding vector for a deal and column."""
    if col not in col_indices or dealnum not in col_indices[col]:
        return None
    return emb_data[col][col_indices[col][dealnum]]


def _build_features_concat(emb_data, col_indices, cols, deal_set):
    """Concatenate embedding columns for deals in deal_set."""
    valid_deals = []
    rows = []

    candidates = deal_set.copy()
    for col in cols:
        if col not in col_indices:
            return np.empty((0, 0)), np.array([])
        candidates &= set(col_indices[col].keys())

    for dn in sorted(candidates):
        parts = []
        for col in cols:
            parts.append(_get_embedding(emb_data, col_indices, col, dn))
        rows.append(np.concatenate(parts))
        valid_deals.append(dn)

    if not rows:
        return np.empty((0, 0)), np.array([])
    return np.stack(rows), np.array(valid_deals)


def _build_features_abs_diff(emb_data, col_indices, acq_col, tgt_col, deal_set):
    """Compute element-wise absolute difference |acquirer - target|."""
    valid_deals = []
    rows = []

    candidates = deal_set & set(col_indices.get(acq_col, {}).keys()) & set(col_indices.get(tgt_col, {}).keys())

    for dn in sorted(candidates):
        acq_vec = _get_embedding(emb_data, col_indices, acq_col, dn)
        tgt_vec = _get_embedding(emb_data, col_indices, tgt_col, dn)
        rows.append(np.abs(acq_vec - tgt_vec))
        valid_deals.append(dn)

    if not rows:
        return np.empty((0, 0)), np.array([])
    return np.stack(rows), np.array(valid_deals)


def _build_features_product(emb_data, col_indices, acq_col, tgt_col, deal_set):
    """Compute element-wise product (acquirer * target)."""
    valid_deals = []
    rows = []

    candidates = deal_set & set(col_indices.get(acq_col, {}).keys()) & set(col_indices.get(tgt_col, {}).keys())

    for dn in sorted(candidates):
        acq_vec = _get_embedding(emb_data, col_indices, acq_col, dn)
        tgt_vec = _get_embedding(emb_data, col_indices, tgt_col, dn)
        rows.append(acq_vec * tgt_vec)
        valid_deals.append(dn)

    if not rows:
        return np.empty((0, 0)), np.array([])
    return np.stack(rows), np.array(valid_deals)


def _build_features(emb_data, col_indices, input_set, representation, deal_set):
    """Route to the appropriate feature builder."""
    cols = INPUT_COLS[input_set]

    if representation == "concat":
        return _build_features_concat(emb_data, col_indices, cols, deal_set)

    if representation in ("abs_diff", "product"):
        if input_set not in PAIR_COLS:
            return np.empty((0, 0)), np.array([])
        acq_col, tgt_col = PAIR_COLS[input_set]
        if representation == "abs_diff":
            return _build_features_abs_diff(emb_data, col_indices, acq_col, tgt_col, deal_set)
        else:
            return _build_features_product(emb_data, col_indices, acq_col, tgt_col, deal_set)

    raise ValueError(f"Unknown representation: {representation}")


def _fit_and_predict(X_train, y_train, X_test, y_test, dn_test, label_map,
                     input_set, representation, test_name, seed, model_dir,
                     pca_components=50):
    """Fit logistic model, predict, return list of prediction dicts."""
    model_label = f"{input_set}_{representation}"

    steps = [("scaler", StandardScaler())]
    if X_train.shape[1] > pca_components:
        n_comp = min(pca_components, X_train.shape[0] - 1)
        steps.append(("pca", PCA(n_components=n_comp, random_state=seed)))
    steps.append(("clf", LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l2",
        scoring="roc_auc",
        max_iter=1000,
        random_state=seed,
    )))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    best_C = pipe.named_steps["clf"].C_[0]
    prob_test = pipe.predict_proba(X_test)[:, 1]
    pca_info = f", PCA={n_comp}" if "pca" in pipe.named_steps else ""

    print(f"    {test_name}: n={len(X_test)}, pos={y_test.sum()}, C={best_C:.4f}{pca_info}")

    # Save model
    model_path = model_dir / f"{model_label}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    preds = []
    for dn, prob in zip(dn_test, prob_test):
        preds.append({
            "dealnum": dn,
            "input_set": input_set,
            "representation": representation,
            "model": model_label,
            "test_set": test_name,
            "y_true": label_map[dn],
            "y_prob": prob,
        })
    return preds


def main():
    cfg = load()
    paths = cfg["paths"]
    fit_cfg = cfg["fit"]
    sample_cfg = cfg["sample"]
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

    # ── Load deals and build splits ─────────────────────────────────────────
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    deals["adate"] = pd.to_datetime(deals["adate"])
    deals["ayear_int"] = deals["ayear"].astype(float).astype(int)

    # Only keep deals that have ALL four embedding columns and labels
    all_cols = ["acquirer_item1", "target_item1", "acquirer_item7", "target_item7"]
    valid_deals = set(dealnums) & set(labels["dealnum"])
    for col in all_cols:
        if col in col_indices:
            valid_deals &= set(col_indices[col].keys())
    deals = deals[deals["dealnum"].isin(valid_deals)].copy()
    n_in = len(deals)
    print(f"Deals with all embeddings + labels: {n_in}")

    # ── Two test sets split ─────────────────────────────────────────────────
    train_years = sample_cfg.get("train_years", sample_cfg["announce_years"])
    oos_years = sample_cfg.get("test_oos_years", [])

    if fit_cfg["split"] == "two_test_sets" and oos_years:
        # In-sample deals (train_years)
        in_sample = deals[deals["ayear_int"].isin(train_years)].copy()
        oos_deals = deals[deals["ayear_int"].isin(oos_years)].copy()

        # 75/25 random split on in-sample
        rng = np.random.RandomState(seed)
        shuffled = np.array(list(in_sample["dealnum"]))
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - fit_cfg["test_share"]))
        train_deals = set(shuffled[:split_idx])
        test_a_deals = set(shuffled[split_idx:])

        # Test B: random sample from OOS years, same size as test A
        n_test_a = len(test_a_deals)
        oos_dealnums = np.array(list(oos_deals["dealnum"]))
        rng.shuffle(oos_dealnums)
        test_b_deals = set(oos_dealnums[:n_test_a])

        print(f"Train ({train_years}): {len(train_deals)}")
        print(f"Test A (holdout from {train_years}): {len(test_a_deals)}")
        print(f"Test B (OOS from {oos_years}): {len(test_b_deals)}")

        test_sets = {"test_A": test_a_deals, "test_B": test_b_deals}

    elif fit_cfg["split"] == "time":
        deals = deals.sort_values("adate")
        split_idx = int(len(deals) * (1 - fit_cfg["test_share"]))
        train_deals = set(deals.iloc[:split_idx]["dealnum"])
        test_deals = set(deals.iloc[split_idx:]["dealnum"])
        split_date = deals.iloc[split_idx]["adate"]
        print(f"Time split at {split_date}: {len(train_deals)} train, {len(test_deals)} test")
        test_sets = {"test": test_deals}

    else:
        rng = np.random.RandomState(seed)
        shuffled = deals["dealnum"].values.copy()
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - fit_cfg["test_share"]))
        train_deals = set(shuffled[:split_idx])
        test_deals = set(shuffled[split_idx:])
        print(f"Random split: {len(train_deals)} train, {len(test_deals)} test")
        test_sets = {"test": test_deals}

    # ── Fit models per input set × representation ─────────────────────────
    model_dir = paths["outputs"] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    representations = fit_cfg.get("representations", ["concat"])
    pca_components = fit_cfg.get("pca_components", 50)
    all_preds = []

    for input_set in fit_cfg["input_sets"]:
        if input_set not in INPUT_COLS:
            print(f"WARNING: unknown input_set {input_set}, skipping")
            continue

        for rep in representations:
            # Skip pair-level representations for non-pair input sets
            if rep in ("abs_diff", "product") and input_set not in PAIR_COLS:
                continue

            model_label = f"{input_set}_{rep}"
            print(f"\n{'='*50}")
            print(f"Input set: {input_set}, Representation: {rep}")

            # Build training features
            X_train, dn_train = _build_features(
                emb_data, col_indices, input_set, rep, train_deals
            )

            if len(X_train) == 0:
                print(f"  Skipping: no training data")
                continue

            y_train = np.array([label_map[d] for d in dn_train])
            print(f"  Train: {len(X_train)} ({y_train.sum()} pos), Features: {X_train.shape[1]}")

            # Fit and predict on each test set
            for test_name, test_deal_set in test_sets.items():
                X_test, dn_test = _build_features(
                    emb_data, col_indices, input_set, rep, test_deal_set
                )

                if len(X_test) == 0:
                    print(f"    {test_name}: Skipping (no data)")
                    continue

                y_test = np.array([label_map[d] for d in dn_test])

                preds = _fit_and_predict(
                    X_train, y_train, X_test, y_test, dn_test, label_map,
                    input_set, rep, test_name, seed, model_dir, pca_components
                )
                all_preds.extend(preds)

    # ── Save keyword predictions ─────────────────────────────────────────────
    preds_df = pd.DataFrame(all_preds)
    preds_path = paths["outputs"] / "predictions.parquet"
    preds_df.to_parquet(preds_path, index=False)
    print(f"\nSaved {len(preds_df)} keyword predictions -> {preds_path}")

    # ── Cosine similarities for CARs regression (step 7) ────────────────────
    print(f"\n{'#'*60}")
    print("Computing cosine similarities for CARs regression...")
    cos_rows = []
    for section, (acq_col, tgt_col) in PAIR_COLS.items():
        candidates = valid_deals & set(col_indices.get(acq_col, {}).keys()) \
                                 & set(col_indices.get(tgt_col, {}).keys())
        for dn in sorted(candidates):
            acq_vec = _get_embedding(emb_data, col_indices, acq_col, dn).reshape(1, -1)
            tgt_vec = _get_embedding(emb_data, col_indices, tgt_col, dn).reshape(1, -1)
            cos_sim = sk_cosine(acq_vec, tgt_vec)[0, 0]
            cos_rows.append({"dealnum": dn, "section": section, "cosine_sim": cos_sim})

    cos_df = pd.DataFrame(cos_rows)
    cos_path = paths["outputs"] / "cosine_similarities.parquet"
    cos_df.to_parquet(cos_path, index=False)
    print(f"Saved {len(cos_df)} cosine similarities -> {cos_path}")

    n_out = preds_df["dealnum"].nunique() if len(preds_df) else 0
    log_step("06_fit", n_in, n_out,
             f"input_sets={fit_cfg['input_sets']} reps={representations}")


if __name__ == "__main__":
    main()
