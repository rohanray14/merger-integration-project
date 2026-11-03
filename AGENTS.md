# Merger integrability pilot

Rohan Ray with Prof. Gordon Phillips (Tuck) and Dr. Will (Huaizhou) Li.

**Goal:** test whether ChronoBERT embeddings of pre-merger 10-K sections
(Item 1 business description, Item 7 MD&A) for acquirer and target predict
post-merger integration problems, measured with the Hoberg & Phillips (2018)
page-17 keyword rule on the acquirer's post-merger 10-Ks.

## Pilot 3 changes (this branch)

- **Expanded sample:** 2000–2008. Train on 2000–2006, OOS test on 2007–2008.
- **Two test sets:** Test A = 25% random holdout from train years. Test B = ALL deals from 2007–2008.
- **Three representations:** concat, abs_diff (|acq − tgt|), product (acq × tgt). Replaced cosine and signed diff.
- **PCA reduction:** 50 components before logistic regression when features > 50.
- **CARs regression:** OLS with controls: `CAR = β₁×skill + β₂×cosine_sim + β₃×log_deal_value + β₄×same_industry + β₅×pct_cash`.
  Skill = 1 − P(integration problems) from keyword model. Caveat: generated regressor (Pagan 1984).
- **Staleness filter:** Pre-merger filings >18 months before announcement are dropped.
- **Removed habit baselines** (captured serial acquirer reporting style, not deal-specific signal).
- All-4-sections requirement kept for apples-to-apples comparison across models.

## Hard rules

- **No look-ahead.** A deal announced in year t uses ChronoBERT
`manelalab/chrono-bert-v1-{t-1}1231`, and only 10-Ks filed strictly before
the announcement date as inputs.
- Never modify or delete anything in `data/raw/`. Each script reads one stage
and writes the next (`raw -> interim -> processed`).
- Never commit anything under `data/` (licensed SDC/WRDS data).
- All settings (paths, years, lag, seed, label window) come from `config.yaml`
via `integrability.config.load()`. No hard-coded paths or years.
- Every script logs deals in / deals out with `integrability.counts.log_step()`
to `outputs/sample_counts.csv`.
- Reusable logic goes in `src/integrability/`; notebooks are exploration only.
- Record every judgment call (filters, windows, edge cases) in
`notes/decisions.md` with a date.



## Pipeline


| Step | Script                   | Input                   | Output                                                               |
| ---- | ------------------------ | ----------------------- | -------------------------------------------------------------------- |
| 1    | `01_select_deals.py`     | SDC file in `data/raw/` | `data/interim/deals.parquet`                                         |
| 2    | `02_link_filings.py`     | deals + gvkey-CIK link  | `data/interim/filings.parquet` (pre/post 10-Ks, staleness filter)    |
| 3    | `03_extract_sections.py` | filings                 | `data/interim/sections.parquet` (Item 1, Item 7, paragraphs)         |
| 4    | `04_build_labels.py`     | post-merger 10-Ks       | `data/processed/labels.parquet` (dummy, intensity, pre-merger habit) |
| 4b   | `04b_build_returns.py`   | deals + CRSP daily      | `data/processed/announcement_returns.parquet` (acquirer CARs)        |
| 5    | `05_embed.py` (GPU)      | sections                | `data/processed/embeddings/`                                         |
| 6    | `06_fit.py`              | embeddings + labels     | `outputs/models/`, predictions, cosine similarities                  |
| 7    | `07_evaluate.py`         | predictions + CARs      | AUC tables, ROC curves, CARs regression with controls                |




## Keyword rule (label), Hoberg & Phillips 2018 p.17

A paragraph is a hit if it contains at least one word from EACH list,
excluding paragraphs in the risk-factor section (Item 1A):

- Acquisition: merger, mergers, merged, acquisition, acquisitions, acquired
- Integration: integration, integrate, integrating
- Difficulty: challenge, challenging, difficulties, difficulty, inability,
failure, unsuccessful, "substantial expense"
- Dummy = any hit; Intensity = hit paragraphs / total paragraphs.
- The word lists live in `config.yaml`; `src/integrability/keywords.py` applies them.



## Section extraction

Item 1: from "Item 1. Business" to "Item 1A" or "Item 2".
Item 7: from "Item 7. Management's Discussion" to "Item 7A" or "Item 8".
Skip the table-of-contents match (take the occurrence followed by real text).
Spot-check extractions by hand; tests in `tests/` must pass.

## Environment

Python 3.11, dependencies in `pyproject.toml`. Steps 1-4 and 6-7 run locally;
step 5 needs a GPU (Colab or Dartmouth Discovery) and caches its output.
