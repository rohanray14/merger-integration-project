"""Step 4. Apply the keyword rule to build integration-difficulty labels.

Post-merger acquirer 10-K paragraphs -> dummy + intensity (the dependent variable).
Pre-merger acquirer 10-K paragraphs  -> pre_habit (control for habitual language).

Reads  : data/interim/sections.parquet
Writes : data/processed/labels.parquet
"""
import pandas as pd
from integrability.config import load
from integrability.counts import log_step
from integrability.keywords import make_rule, score


def main():
    cfg = load()
    paths = cfg["paths"]

    # ── Load sections ───────────────────────────────────────────────────────
    sections = pd.read_parquet(paths["interim"] / "sections.parquet")
    n_deals_in = sections["dealnum"].nunique()
    print(f"Loaded {len(sections)} paragraphs across {n_deals_in} deals")

    is_hit = make_rule(cfg["label"])

    # ── Post-merger labels (dependent variable) ─────────────────────────────
    post = sections[
        (sections["purpose"] == "post")
        & (sections["role"] == "acquirer")
    ]
    print(f"Post-merger acquirer paragraphs: {len(post)}")

    post_labels = []
    for dealnum, grp in post.groupby("dealnum"):
        paragraphs = grp["text"].tolist()
        result = score(paragraphs, is_hit)
        result["dealnum"] = dealnum
        post_labels.append(result)

    post_df = pd.DataFrame(post_labels)
    if len(post_df):
        print(f"  Deals with post-merger label: {len(post_df)}")
        print(f"  Dummy=1: {post_df['hit'].sum()} / {len(post_df)} "
              f"({post_df['hit'].mean():.1%})")
        print(f"  Mean intensity: {post_df['intensity'].mean():.4f}")

    # ── Pre-merger habit control ────────────────────────────────────────────
    pre_acq = sections[
        (sections["purpose"] == "pre")
        & (sections["role"] == "acquirer")
    ]
    print(f"\nPre-merger acquirer paragraphs: {len(pre_acq)}")

    pre_labels = []
    for dealnum, grp in pre_acq.groupby("dealnum"):
        paragraphs = grp["text"].tolist()
        result = score(paragraphs, is_hit)
        pre_labels.append({
            "dealnum": dealnum,
            "pre_habit": result["hit"],
            "pre_intensity": result["intensity"],
        })

    pre_df = pd.DataFrame(pre_labels)
    if len(pre_df):
        print(f"  Deals with pre-merger habit: {len(pre_df)}")
        print(f"  Pre-habit=1: {pre_df['pre_habit'].sum()} / {len(pre_df)}")

    # ── Merge ───────────────────────────────────────────────────────────────
    if len(post_df) == 0:
        print("WARNING: No post-merger labels produced.")
        labels = pd.DataFrame(columns=["dealnum", "hit", "hit_paragraphs",
                                        "paragraphs", "intensity",
                                        "pre_habit", "pre_intensity"])
    else:
        labels = post_df.copy()
        if len(pre_df):
            labels = labels.merge(pre_df, on="dealnum", how="left")
            labels["pre_habit"] = labels["pre_habit"].fillna(0).astype(int)
            labels["pre_intensity"] = labels["pre_intensity"].fillna(0.0)

    n_out = len(labels)
    print(f"\nFinal labels: {n_out} deals")

    log_step("04_build_labels", n_deals_in, n_out,
             f"dummy_rate={labels['hit'].mean():.3f}" if n_out else "empty")

    # ── Save ────────────────────────────────────────────────────────────────
    out = paths["processed"] / "labels.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out, index=False)
    print(f"Saved {n_out} rows -> {out}")


if __name__ == "__main__":
    main()
