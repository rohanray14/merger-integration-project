"""Step 8. Separate acquirer skill from deal difficulty.

Step 1 – Difficulty model: predict integration problems using only pair-level
  features (abs_diff embeddings + cosine similarity). No acquirer identity.

Step 2 – Acquirer skill: for each deal, residual = actual - predicted difficulty.
  For repeat acquirers, average residuals across *prior* deals only (temporal
  ordering). Skill = -mean(residuals), so higher = better at integrating.

Step 3 – Output per-deal difficulty, skill, and interaction for CARs regression.

Reads  : data/processed/embeddings/, data/processed/labels.parquet,
         data/interim/deals.parquet
Writes : outputs/skill_difficulty.parquet
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.model_selection import cross_val_predict
from integrability.config import load
from integrability.counts import log_step


def _build_emb_indices(dealnums, meta):
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
    if col not in col_indices or dealnum not in col_indices[col]:
        return None
    return emb_data[col][col_indices[col][dealnum]]


def _build_pair_features(emb_data, col_indices, deal_set):
    """Build pair-level feature matrix: abs_diff(item1) + abs_diff(mdna) + cosine sims.

    Returns (X, dealnums) where X has columns:
      [abs_diff_item1 (768d) | abs_diff_mdna (768d) | cos_item1 | cos_mdna]
    """
    PAIRS = [
        ("acquirer_item1", "target_item1"),
        ("acquirer_item7", "target_item7"),
    ]

    # Find deals with all four embeddings
    candidates = set(deal_set)
    for acq_col, tgt_col in PAIRS:
        candidates &= set(col_indices.get(acq_col, {}).keys())
        candidates &= set(col_indices.get(tgt_col, {}).keys())

    rows = []
    valid_deals = []
    for dn in sorted(candidates):
        parts = []
        for acq_col, tgt_col in PAIRS:
            acq_vec = _get_embedding(emb_data, col_indices, acq_col, dn)
            tgt_vec = _get_embedding(emb_data, col_indices, tgt_col, dn)
            # Absolute difference
            parts.append(np.abs(acq_vec - tgt_vec))
            # Cosine similarity
            cos = sk_cosine(acq_vec.reshape(1, -1), tgt_vec.reshape(1, -1))[0, 0]
            parts.append(np.array([cos]))
        rows.append(np.concatenate(parts))
        valid_deals.append(dn)

    if not rows:
        return np.empty((0, 0)), np.array([])
    return np.stack(rows), np.array(valid_deals)


def main():
    cfg = load()
    paths = cfg["paths"]
    fit_cfg = cfg["fit"]
    sample_cfg = cfg["sample"]
    seed = fit_cfg["seed"]

    # ── Load embeddings ──────────────────────────────────────────────────
    emb_dir = paths["processed"] / "embeddings"
    emb_data = dict(np.load(emb_dir / "embeddings.npz", allow_pickle=True))
    dealnums = emb_data.pop("dealnums")
    meta = pd.read_parquet(emb_dir / "metadata.parquet")
    col_indices = _build_emb_indices(dealnums, meta)
    print(f"Loaded embeddings for {len(dealnums)} deals")

    # ── Load labels ──────────────────────────────────────────────────────
    labels = pd.read_parquet(paths["processed"] / "labels.parquet")
    label_map = dict(zip(labels["dealnum"], labels["hit"]))
    print(f"Labels: {len(labels)} deals, {labels['hit'].sum()} positive")

    # ── Load deals ───────────────────────────────────────────────────────
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    deals["adate"] = pd.to_datetime(deals["adate"])
    deals["ayear_int"] = deals["ayear"].astype(float).astype(int)

    # Only keep deals with all four embeddings + labels
    all_cols = ["acquirer_item1", "target_item1", "acquirer_item7", "target_item7"]
    valid_deals = set(dealnums) & set(labels["dealnum"])
    for col in all_cols:
        if col in col_indices:
            valid_deals &= set(col_indices[col].keys())
    deals = deals[deals["dealnum"].isin(valid_deals)].copy()
    n_in = len(deals)
    print(f"Deals with all embeddings + labels: {n_in}")

    # ── Train/test split (same as 06_fit.py) ─────────────────────────────
    train_years = sample_cfg.get("train_years", sample_cfg["announce_years"])
    oos_years = sample_cfg.get("test_oos_years", [])

    in_sample = deals[deals["ayear_int"].isin(train_years)]
    oos_deals = deals[deals["ayear_int"].isin(oos_years)]

    rng = np.random.RandomState(seed)
    shuffled = np.array(list(in_sample["dealnum"]))
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - fit_cfg["test_share"]))
    train_deal_set = set(shuffled[:split_idx])
    test_a_set = set(shuffled[split_idx:])
    test_b_set = set(oos_deals["dealnum"])
    all_deal_set = train_deal_set | test_a_set | test_b_set

    print(f"Train: {len(train_deal_set)}, Test A: {len(test_a_set)}, Test B: {len(test_b_set)}")

    # ── Step 1: Difficulty model (pair-level features only) ──────────────
    print(f"\n{'='*60}")
    print("Step 1: Fitting difficulty model (pair-level features only)")

    X_all, dn_all = _build_pair_features(emb_data, col_indices, all_deal_set)
    dn_to_idx = {dn: i for i, dn in enumerate(dn_all)}
    y_all = np.array([label_map[dn] for dn in dn_all])

    # Training subset indices
    train_mask = np.array([dn in train_deal_set for dn in dn_all])
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    print(f"  Pair features: {X_all.shape[1]} dims for {len(dn_all)} deals")
    print(f"  Training: {len(X_train)} deals, {y_train.sum()} positive")

    # Fit difficulty pipeline
    pca_components = fit_cfg.get("pca_components", 50)
    steps = [("scaler", StandardScaler())]
    if X_train.shape[1] > pca_components:
        n_comp = min(pca_components, X_train.shape[0] - 1)
        steps.append(("pca", PCA(n_components=n_comp, random_state=seed)))
    steps.append(("clf", LogisticRegressionCV(
        Cs=10, cv=5, penalty="l2", scoring="roc_auc",
        max_iter=1000, random_state=seed,
    )))
    difficulty_pipe = Pipeline(steps)
    difficulty_pipe.fit(X_train, y_train)

    best_C = difficulty_pipe.named_steps["clf"].C_[0]
    pca_info = f", PCA={n_comp}" if "pca" in difficulty_pipe.named_steps else ""
    print(f"  Best C={best_C:.4f}{pca_info}")

    # Predict difficulty for ALL deals
    # For training deals: use cross-validated predictions to avoid overfitting
    difficulty_scores = np.full(len(dn_all), np.nan)

    # CV predictions for training deals
    cv_probs = cross_val_predict(
        difficulty_pipe, X_train, y_train, cv=5, method="predict_proba",
    )[:, 1]
    difficulty_scores[train_mask] = cv_probs

    # Direct predictions for test deals
    test_mask = ~train_mask
    if test_mask.any():
        difficulty_scores[test_mask] = difficulty_pipe.predict_proba(
            X_all[test_mask]
        )[:, 1]

    print(f"  Difficulty scores: mean={np.nanmean(difficulty_scores):.3f}, "
          f"std={np.nanstd(difficulty_scores):.3f}")

    # ── Step 2: Acquirer skill (rolling historical residuals) ────────────
    print(f"\n{'='*60}")
    print("Step 2: Computing acquirer skill (rolling, temporal)")

    # Build deal-level dataframe
    deal_df = pd.DataFrame({
        "dealnum": dn_all,
        "difficulty": difficulty_scores,
        "y_true": y_all,
    })
    deal_df["residual"] = deal_df["y_true"] - deal_df["difficulty"]

    # Merge announcement date and acquirer identity
    deal_info = deals[["dealnum", "adate", "acq_gvkey"]].copy()
    deal_df = deal_df.merge(deal_info, on="dealnum", how="left")
    deal_df = deal_df.sort_values("adate")

    # Compute rolling skill: for each deal, average residuals of the same
    # acquirer's prior deals (announced strictly before this deal)
    skill_values = {}
    acq_history = {}  # acq_gvkey -> list of (adate, residual)

    for _, row in deal_df.iterrows():
        dn = row["dealnum"]
        acq = row["acq_gvkey"]
        adate = row["adate"]
        resid = row["residual"]

        if pd.isna(acq):
            skill_values[dn] = np.nan
            continue

        acq = str(acq)

        # Look up prior deals for this acquirer
        if acq in acq_history and len(acq_history[acq]) > 0:
            prior_resids = [r for d, r in acq_history[acq] if d < adate]
            if len(prior_resids) > 0:
                # Skill = negative mean residual (better = higher)
                skill_values[dn] = -np.mean(prior_resids)
            else:
                skill_values[dn] = np.nan
        else:
            skill_values[dn] = np.nan

        # Add this deal to history
        if acq not in acq_history:
            acq_history[acq] = []
        acq_history[acq].append((adate, resid))

    deal_df["acquirer_skill"] = deal_df["dealnum"].map(skill_values)

    n_with_skill = deal_df["acquirer_skill"].notna().sum()
    n_repeat = deal_df[deal_df["acquirer_skill"].notna()]["acq_gvkey"].nunique()
    print(f"  Deals with skill estimate: {n_with_skill} "
          f"({n_repeat} unique acquirers)")
    print(f"  Skill: mean={deal_df['acquirer_skill'].mean():.3f}, "
          f"std={deal_df['acquirer_skill'].std():.3f}")

    # ── Step 3: Output ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3: Saving outputs")

    # Add test set membership
    deal_df["test_set"] = "train"
    deal_df.loc[deal_df["dealnum"].isin(test_a_set), "test_set"] = "test_A"
    deal_df.loc[deal_df["dealnum"].isin(test_b_set), "test_set"] = "test_B"

    # Interaction term
    deal_df["skill_x_difficulty"] = deal_df["acquirer_skill"] * deal_df["difficulty"]

    out_cols = ["dealnum", "acq_gvkey", "adate", "test_set",
                "y_true", "difficulty", "residual",
                "acquirer_skill", "skill_x_difficulty"]
    out_df = deal_df[out_cols].copy()

    out_path = paths["outputs"] / "skill_difficulty.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} deals -> {out_path}")

    # Summary stats by test set
    for ts in ["train", "test_A", "test_B"]:
        sub = out_df[out_df["test_set"] == ts]
        n_skill = sub["acquirer_skill"].notna().sum()
        print(f"  {ts}: {len(sub)} deals, {n_skill} with skill")

    n_out = len(out_df)
    log_step("08_skill_difficulty", n_in, n_out,
             "difficulty model (pair-level) + rolling acquirer skill")


if __name__ == "__main__":
    main()
