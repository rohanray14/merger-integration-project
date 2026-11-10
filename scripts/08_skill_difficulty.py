"""Step 8. Separate acquirer skill from deal difficulty.

Two difficulty-model specifications (compared on the same sample):
  A) Item 1 only:  abs_diff(item1) + cosine_sim(item1)
  B) All available: abs_diff(item1) + cosine_sim(item1)
                  + abs_diff(mdna) + cosine_sim(mdna)  [when available]
     Spec B does NOT require all four sections — it uses whatever is available.

For each spec:
  1. Fit difficulty model on training deals (pair-level features, no acquirer identity).
  2. Compute acquirer skill from rolling historical residuals (prior deals only).
  3. Output per-deal difficulty, skill, and interaction for CARs regression.

Reads  : data/processed/embeddings/, data/processed/labels.parquet,
         data/interim/deals.parquet
Writes : outputs/skill_difficulty.parquet (spec A — Item 1 only, larger sample)
         outputs/skill_difficulty_all.parquet (spec B — all available text)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
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


def _build_pair_features(emb_data, col_indices, deal_set, pairs):
    """Build pair-level feature matrix for given section pairs.

    Args:
        pairs: list of (acq_col, tgt_col) tuples to include.
               Deals must have ALL listed pairs.
    Returns:
        (X, dealnums) where X columns are [abs_diff | cosine_sim] per pair.
    """
    candidates = set(deal_set)
    for acq_col, tgt_col in pairs:
        candidates &= set(col_indices.get(acq_col, {}).keys())
        candidates &= set(col_indices.get(tgt_col, {}).keys())

    rows = []
    valid_deals = []
    for dn in sorted(candidates):
        parts = []
        for acq_col, tgt_col in pairs:
            acq_vec = _get_embedding(emb_data, col_indices, acq_col, dn)
            tgt_vec = _get_embedding(emb_data, col_indices, tgt_col, dn)
            parts.append(np.abs(acq_vec - tgt_vec))
            cos = sk_cosine(acq_vec.reshape(1, -1), tgt_vec.reshape(1, -1))[0, 0]
            parts.append(np.array([cos]))
        rows.append(np.concatenate(parts))
        valid_deals.append(dn)

    if not rows:
        return np.empty((0, 0)), np.array([])
    return np.stack(rows), np.array(valid_deals)


def _fit_difficulty_and_skill(emb_data, col_indices, pairs, label_map, deals,
                              train_deal_set, test_a_set, test_b_set,
                              seed, pca_components, spec_name):
    """Fit difficulty model and compute rolling skill for one specification."""
    all_deal_set = train_deal_set | test_a_set | test_b_set

    print(f"\n{'#'*60}")
    print(f"Spec {spec_name}")
    print(f"  Pairs: {[(a.split('_',1)[1], t.split('_',1)[1]) for a, t in pairs]}")

    # ── Build features ────────────────────────────────────────────────
    X_all, dn_all = _build_pair_features(emb_data, col_indices, all_deal_set, pairs)
    if len(X_all) == 0:
        print(f"  No deals with required embeddings — skipping")
        return None

    y_all = np.array([label_map[dn] for dn in dn_all])
    train_mask = np.array([dn in train_deal_set for dn in dn_all])
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    print(f"  Features: {X_all.shape[1]} dims, {len(dn_all)} deals")
    print(f"  Training: {train_mask.sum()} deals, {y_train.sum()} positive")

    # ── Fit difficulty pipeline ───────────────────────────────────────
    steps = [("scaler", StandardScaler())]
    if X_train.shape[1] > pca_components:
        n_comp = min(pca_components, X_train.shape[0] - 1)
        steps.append(("pca", PCA(n_components=n_comp, random_state=seed)))
    steps.append(("clf", LogisticRegressionCV(
        Cs=10, cv=5, penalty="l2", scoring="roc_auc",
        max_iter=1000, random_state=seed,
    )))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    best_C = pipe.named_steps["clf"].C_[0]
    pca_info = f", PCA={n_comp}" if "pca" in pipe.named_steps else ""
    print(f"  Best C={best_C:.4f}{pca_info}")

    # ── Predict difficulty ────────────────────────────────────────────
    difficulty_scores = np.full(len(dn_all), np.nan)

    cv_probs = cross_val_predict(
        pipe, X_train, y_train, cv=5, method="predict_proba",
    )[:, 1]
    difficulty_scores[train_mask] = cv_probs

    test_mask = ~train_mask
    if test_mask.any():
        difficulty_scores[test_mask] = pipe.predict_proba(X_all[test_mask])[:, 1]

    # AUC per test set
    for label, mask_set in [("test_A", test_a_set), ("test_B", test_b_set)]:
        idx = np.array([dn in mask_set for dn in dn_all])
        if idx.any() and len(set(y_all[idx])) >= 2:
            auc = roc_auc_score(y_all[idx], difficulty_scores[idx])
            print(f"  {label}: AUC={auc:.3f} (n={idx.sum()}, pos={y_all[idx].sum()})")

    # ── Compute rolling skill ─────────────────────────────────────────
    deal_df = pd.DataFrame({
        "dealnum": dn_all,
        "difficulty": difficulty_scores,
        "y_true": y_all,
    })
    deal_df["residual"] = deal_df["y_true"] - deal_df["difficulty"]

    deal_info = deals[["dealnum", "adate", "acq_gvkey"]].copy()
    deal_df = deal_df.merge(deal_info, on="dealnum", how="left")
    deal_df = deal_df.sort_values("adate")

    skill_values = {}
    acq_history = {}

    for _, row in deal_df.iterrows():
        dn = row["dealnum"]
        acq = row["acq_gvkey"]
        adate = row["adate"]
        resid = row["residual"]

        if pd.isna(acq):
            skill_values[dn] = np.nan
            continue

        acq = str(acq)

        if acq in acq_history and len(acq_history[acq]) > 0:
            prior_resids = [r for d, r in acq_history[acq] if d < adate]
            if len(prior_resids) > 0:
                skill_values[dn] = -np.mean(prior_resids)
            else:
                skill_values[dn] = np.nan
        else:
            skill_values[dn] = np.nan

        if acq not in acq_history:
            acq_history[acq] = []
        acq_history[acq].append((adate, resid))

    deal_df["acquirer_skill"] = deal_df["dealnum"].map(skill_values)

    n_with_skill = deal_df["acquirer_skill"].notna().sum()
    n_repeat = deal_df[deal_df["acquirer_skill"].notna()]["acq_gvkey"].nunique()
    print(f"  Deals with skill: {n_with_skill} ({n_repeat} acquirers)")

    # ── Finalize output ───────────────────────────────────────────────
    deal_df["test_set"] = "train"
    deal_df.loc[deal_df["dealnum"].isin(test_a_set), "test_set"] = "test_A"
    deal_df.loc[deal_df["dealnum"].isin(test_b_set), "test_set"] = "test_B"
    deal_df["skill_x_difficulty"] = deal_df["acquirer_skill"] * deal_df["difficulty"]
    deal_df["spec"] = spec_name

    for ts in ["train", "test_A", "test_B"]:
        sub = deal_df[deal_df["test_set"] == ts]
        n_skill = sub["acquirer_skill"].notna().sum()
        print(f"  {ts}: {len(sub)} deals, {n_skill} with skill")

    return deal_df


def main():
    cfg = load()
    paths = cfg["paths"]
    fit_cfg = cfg["fit"]
    sample_cfg = cfg["sample"]
    seed = fit_cfg["seed"]
    pca_components = fit_cfg.get("pca_components", 50)

    # ── Load data ────────────────────────────────────────────────────
    emb_dir = paths["processed"] / "embeddings"
    emb_data = dict(np.load(emb_dir / "embeddings.npz", allow_pickle=True))
    dealnums = emb_data.pop("dealnums")
    meta = pd.read_parquet(emb_dir / "metadata.parquet")
    col_indices = _build_emb_indices(dealnums, meta)
    print(f"Loaded embeddings for {len(dealnums)} deals")
    for col, idx_map in col_indices.items():
        print(f"  {col}: {len(idx_map)} deals")

    labels = pd.read_parquet(paths["processed"] / "labels.parquet")
    label_map = dict(zip(labels["dealnum"], labels["hit"]))
    print(f"Labels: {len(labels)} deals, {labels['hit'].sum()} positive")

    deals = pd.read_parquet(paths["interim"] / "deals.parquet")
    deals["adate"] = pd.to_datetime(deals["adate"])
    deals["ayear_int"] = deals["ayear"].astype(float).astype(int)

    # ── Eligible deals: have labels + at least Item 1 for both firms ─
    item1_deals = (set(dealnums) & set(labels["dealnum"])
                   & set(col_indices.get("acquirer_item1", {}).keys())
                   & set(col_indices.get("target_item1", {}).keys()))
    deals_eligible = deals[deals["dealnum"].isin(item1_deals)].copy()
    n_in = len(deals_eligible)
    print(f"\nDeals with both Item 1 + labels: {n_in}")

    # ── Train/test split on the larger Item 1 sample ─────────────────
    train_years = sample_cfg.get("train_years", sample_cfg["announce_years"])
    oos_years = sample_cfg.get("test_oos_years", [])

    in_sample = deals_eligible[deals_eligible["ayear_int"].isin(train_years)]
    oos_deals = deals_eligible[deals_eligible["ayear_int"].isin(oos_years)]

    rng = np.random.RandomState(seed)
    shuffled = np.array(list(in_sample["dealnum"]))
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - fit_cfg["test_share"]))
    train_deal_set = set(shuffled[:split_idx])
    test_a_set = set(shuffled[split_idx:])
    test_b_set = set(oos_deals["dealnum"])

    print(f"Train: {len(train_deal_set)}, Test A: {len(test_a_set)}, Test B: {len(test_b_set)}")

    # ── Spec A: Item 1 only ──────────────────────────────────────────
    pairs_item1 = [("acquirer_item1", "target_item1")]
    df_a = _fit_difficulty_and_skill(
        emb_data, col_indices, pairs_item1, label_map, deals_eligible,
        train_deal_set, test_a_set, test_b_set,
        seed, pca_components, "item1_only",
    )

    # ── Spec B: All available (Item 1 + MD&A where both exist) ───────
    # Use the same train/test split, but only deals that also have MD&A
    pairs_all = [
        ("acquirer_item1", "target_item1"),
        ("acquirer_item7", "target_item7"),
    ]
    df_b = _fit_difficulty_and_skill(
        emb_data, col_indices, pairs_all, label_map, deals_eligible,
        train_deal_set, test_a_set, test_b_set,
        seed, pca_components, "item1_mdna",
    )

    # ── Compare specs on overlapping sample ──────────────────────────
    if df_a is not None and df_b is not None:
        overlap = set(df_a["dealnum"]) & set(df_b["dealnum"])
        print(f"\n{'='*60}")
        print(f"Comparing specs on {len(overlap)} overlapping deals")
        for ts in ["test_A", "test_B"]:
            a_ts = df_a[(df_a["test_set"] == ts) & df_a["dealnum"].isin(overlap)]
            b_ts = df_b[(df_b["test_set"] == ts) & df_b["dealnum"].isin(overlap)]
            if len(a_ts) > 0 and len(set(a_ts["y_true"])) >= 2:
                auc_a = roc_auc_score(a_ts["y_true"], a_ts["difficulty"])
                auc_b = roc_auc_score(b_ts.set_index("dealnum").loc[a_ts["dealnum"]]["y_true"],
                                      b_ts.set_index("dealnum").loc[a_ts["dealnum"]]["difficulty"])
                print(f"  {ts} (n={len(a_ts)}): Item1={auc_a:.3f}, Item1+MD&A={auc_b:.3f}")

    # ── Save outputs ─────────────────────────────────────────────────
    out_cols = ["dealnum", "acq_gvkey", "adate", "test_set", "spec",
                "y_true", "difficulty", "residual",
                "acquirer_skill", "skill_x_difficulty"]

    if df_a is not None:
        out_a = df_a[out_cols].copy()
        path_a = paths["outputs"] / "skill_difficulty.parquet"
        out_a.to_parquet(path_a, index=False)
        print(f"\nSpec A (Item 1 only): {len(out_a)} deals -> {path_a}")

    if df_b is not None:
        out_b = df_b[out_cols].copy()
        path_b = paths["outputs"] / "skill_difficulty_all.parquet"
        out_b.to_parquet(path_b, index=False)
        print(f"Spec B (Item 1 + MD&A): {len(out_b)} deals -> {path_b}")

    n_out = len(df_a) if df_a is not None else 0
    log_step("08_skill_difficulty", n_in, n_out,
             "two specs: item1_only + item1_mdna")


if __name__ == "__main__":
    main()
