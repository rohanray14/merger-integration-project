"""Step 5 (GPU). Embed pre-merger sections with vintage ChronoBERT.

For each deal, selects the ChronoBERT model dated (announce_year - lag),
tokenises paragraphs into chunks of max_tokens, encodes them, and
mean-pools to one 768-dim vector per deal × role × item.

Reads  : data/interim/sections.parquet, data/interim/deals.parquet
Writes : data/processed/embeddings.parquet
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from integrability.config import load, vintage_repo
from integrability.counts import log_step


def _load_model(repo: str, device: torch.device):
    """Load tokenizer and model, moving model to device."""
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def _embed_paragraphs(
    paragraphs: list[str],
    tokenizer,
    model,
    max_tokens: int,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode paragraphs and return mean-pooled embedding (768,).

    Each paragraph is truncated to max_tokens. All paragraph embeddings
    (CLS vectors) are averaged into a single vector.
    """
    if not paragraphs:
        return np.zeros(768, dtype=np.float32)

    all_cls = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :]  # [batch, 768]
        all_cls.append(cls.cpu().numpy())

    all_cls = np.concatenate(all_cls, axis=0)  # [n_paragraphs, 768]
    return all_cls.mean(axis=0).astype(np.float32)


def main():
    cfg = load()
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    max_tokens = model_cfg["max_tokens"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    sections = pd.read_parquet(paths["interim"] / "sections.parquet")
    deals = pd.read_parquet(paths["interim"] / "deals.parquet")

    # Keep only pre-merger sections (features must not look ahead)
    sections = sections[sections["purpose"] == "pre"].copy()
    n_deals_in = sections["dealnum"].nunique()
    print(f"Pre-merger paragraphs: {len(sections)} across {n_deals_in} deals")

    # Map dealnum -> announce year
    deal_year = deals.set_index("dealnum")["ayear"].astype(float).astype(int).to_dict()

    # ── Group by vintage year to minimise model loads ────────────────────────
    sections["ayear"] = sections["dealnum"].map(deal_year)
    sections = sections.dropna(subset=["ayear"])
    sections["ayear"] = sections["ayear"].astype(int)

    vintage_groups = sections.groupby("ayear")
    print(f"Vintage years: {sorted(sections['ayear'].unique())}")

    # ── Embed ────────────────────────────────────────────────────────────────
    results = []  # list of dicts: dealnum, role, item, embedding
    model_cache = {}

    for ayear, year_df in vintage_groups:
        repo = vintage_repo(cfg, ayear)
        print(f"\n{'='*60}")
        print(f"Vintage {ayear} -> {repo}")

        if repo not in model_cache:
            print(f"  Loading model...")
            model_cache[repo] = _load_model(repo, device)
        tokenizer, model = model_cache[repo]

        deal_groups = year_df.groupby(["dealnum", "role", "item"])
        n_groups = len(deal_groups)
        for idx, ((dealnum, role, item), grp) in enumerate(deal_groups):
            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{n_groups} embeddings...")

            paragraphs = grp["text"].tolist()
            emb = _embed_paragraphs(paragraphs, tokenizer, model, max_tokens, device)

            results.append({
                "dealnum": dealnum,
                "role": role,
                "item": item,
                "n_paragraphs": len(paragraphs),
                "embedding": emb,
            })

    print(f"\nTotal embeddings: {len(results)}")

    # ── Pivot to deal-level feature vectors ──────────────────────────────────
    emb_df = pd.DataFrame(results)
    n_deals_out = emb_df["dealnum"].nunique()

    # Widen: each (role, item) combination becomes its own column set
    rows = []
    for dealnum, grp in emb_df.groupby("dealnum"):
        row = {"dealnum": dealnum}
        for _, r in grp.iterrows():
            prefix = f"{r['role']}_{r['item']}"  # e.g. acquirer_item1
            row[prefix] = r["embedding"]
            row[f"{prefix}_n"] = r["n_paragraphs"]
        rows.append(row)

    deal_emb = pd.DataFrame(rows)
    print(f"Deal-level embeddings: {len(deal_emb)} deals")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = paths["processed"] / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw embedding arrays as .npz (easier for downstream)
    emb_cols = [c for c in deal_emb.columns if c.endswith("_item1") or c.endswith("_item7")]
    np.savez(
        out_dir / "embeddings.npz",
        dealnums=deal_emb["dealnum"].values,
        **{col: np.stack(deal_emb[col].dropna().values) for col in emb_cols},
    )

    # Also save a metadata parquet (paragraph counts, dealnum)
    meta_cols = ["dealnum"] + [c for c in deal_emb.columns if c.endswith("_n")]
    deal_emb[meta_cols].to_parquet(out_dir / "metadata.parquet", index=False)

    log_step("05_embed", n_deals_in, n_deals_out,
             f"vectors={len(results)} device={device}")

    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
