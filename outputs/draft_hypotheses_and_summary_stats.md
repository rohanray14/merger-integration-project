# Predicting Post-Merger Integration Difficulties Using Pre-Merger 10-K Embeddings

**Rohan Ray, Gordon Phillips, Huaizhou (Will) Li**

*Rough Draft — October 2026*

---

## 1. Research Question

Can contextual language embeddings of pre-merger SEC filings predict whether an acquirer will report post-merger integration difficulties? We embed Item 1 (Business Description) and Item 7 (MD&A) sections from the acquirer's and target's pre-announcement 10-Ks using ChronoBERT, a time-aware BERT model, and test whether these representations predict integration problems disclosed in the acquirer's post-merger filings.

## 2. Hypotheses

**H1 (Integration signal in pre-merger text).** ChronoBERT embeddings of the acquirer's and target's pre-merger 10-K sections contain information that predicts post-merger integration difficulties beyond a naive baseline.

*Motivation:* Pre-merger business descriptions and management discussion encode information about organizational complexity, operational scope, and strategic direction. Differences or incompatibilities in these narratives may signal future integration frictions that are not captured by standard deal-level variables.

**H2 (Target text adds incremental information).** Including the target's pre-merger 10-K embeddings alongside the acquirer's improves prediction relative to using the acquirer's filings alone.

*Motivation:* Integration is inherently a two-firm problem. The target's business description reveals organizational structures, product lines, and operational practices that the acquirer must absorb. A model using both firms' filings should capture acquirer–target mismatch better than one using only the acquirer's text.

**H3 (Item 1 is more informative than Item 7).** Embeddings of Item 1 (Business Description) are more predictive of integration difficulty than embeddings of Item 7 (MD&A).

*Motivation:* Item 1 describes the firm's core operations, products, and organizational structure — factors directly relevant to operational integration. Item 7, while informative about financial performance and risk, may contain less signal about the practical compatibility of two organizations.

**H4 (Pre-merger habit predicts but embeddings add signal).** Acquirers that already use integration-difficulty language in their pre-merger 10-Ks are more likely to report post-merger integration problems, but embeddings capture additional signal beyond this baseline.

*Motivation:* Serial acquirers with prior integration experience may have standing disclosure language. Controlling for this "habit" isolates the incremental predictive content of the text embeddings.

## 3. Sample Construction

### 3.1 Deal Selection

| Filter | Deals |
|--------|------:|
| SDC M&A universe, announced 2000–2004 | 162,985 |
| Completed, both firms US, both with CIK | 901 |
| With pre-merger 10-K sections extracted | 619 |
| With post-merger labels constructed | 723 |
| In final estimation sample (embeddings + labels) | 295 |

**Source:** Thomson SDC Platinum. Completed (status = C) US public–public mergers announced 2000–2004. Both acquirer and target must have a CIK linkable via WRDS SEC Analytics or CUSIP-CIK crosswalk.

### 3.2 Deals by Announcement Year

| Year | Deals (full) | Labeled |
|------|-------------:|--------:|
| 2000 | 252 | — |
| 2001 | 215 | — |
| 2002 | 132 | — |
| 2003 | 152 | — |
| 2004 | 150 | — |
| **Total** | **901** | **723** |

### 3.3 Industry Composition (2-digit SIC)

| Rank | Acquirer SIC | Count | Target SIC | Count |
|------|-------------|------:|-----------|------:|
| 1 | 73 (Business Services) | 165 | 73 (Business Services) | 197 |
| 2 | 60 (Depository Institutions) | 158 | 60 (Depository Institutions) | 158 |
| 3 | 38 (Instruments) | 63 | 36 (Electronic Equipment) | 58 |
| 4 | 28 (Chemicals) | 57 | 38 (Instruments) | 55 |
| 5 | 36 (Electronic Equipment) | 53 | 28 (Chemicals) | 50 |

- **Same 2-digit SIC (acquirer = target):** 68.0% of deals

### 3.4 Deal Characteristics

| Variable | N | Mean | Median | SD | Min | Max |
|----------|--:|-----:|-------:|---:|----:|----:|
| Transaction value ($M) | 886 | 1,598 | 184 | 7,324 | 0.6 | 164,747 |
| 1-week premium (%) | 803 | 300.0 | 30.9 | 7,406 | -80.0 | 209,900 |

*Note:* Premium outliers likely reflect data entry issues in SDC; the median of 30.9% is consistent with the literature. Transaction value is highly right-skewed (mean >> median) due to mega-deals such as AOL–Time Warner ($165B).

## 4. Label Construction

Following Hoberg & Phillips (2018, p. 17), we flag integration difficulty in the acquirer's post-merger 10-Ks (years +1 to +3 after deal completion). A paragraph is a "hit" if it contains at least one word from **each** of three lists:

- **Acquisition:** merger, mergers, merged, acquisition, acquisitions, acquired
- **Integration:** integration, integrate, integrating
- **Difficulty:** challenge, challenging, difficulties, difficulty, inability, failure, unsuccessful, "substantial expense"

Paragraphs in the risk-factor section (Item 1A) are excluded.

### Label Summary (N = 723 deals)

| Variable | Mean | SD | Min | Max |
|----------|-----:|---:|----:|----:|
| Integration-difficulty dummy | 0.271 | 0.445 | 0 | 1 |
| Hit paragraphs per deal | 0.63 | 1.31 | 0 | 9 |
| Total paragraphs per deal | 394 | 279 | 1 | 1,929 |
| Intensity (hits / paragraphs) | 0.0014 | 0.0028 | 0 | 0.023 |
| Pre-merger habit dummy | 0.152 | 0.359 | 0 | 1 |
| Pre-merger intensity | 0.0010 | 0.0029 | 0 | 0.030 |

- **27.1%** of deals have at least one integration-difficulty hit in post-merger 10-Ks.
- **15.2%** of acquirers already use integration-difficulty language before the deal ("habit").

### Label Rate by Announcement Year

| Year | Hit Rate |
|------|--------:|
| 2000 | 27.1% |
| 2001 | 30.4% |
| 2002 | 33.9% |
| 2003 | 30.6% |
| 2004 | 13.5% |

The lower rate for 2004 may reflect a shorter post-merger observation window or vintage effects.

## 5. Text Data

### 5.1 Section Extraction

Pre-merger 10-K sections were extracted for 619 deals (764 with acquirer sections, 772 with target sections).

| Section | Role | Paragraphs |
|---------|------|----------:|
| Item 1 (Business) | Acquirer pre-merger | 75,403 |
| Item 7 (MD&A) | Acquirer pre-merger | 57,733 |
| Item 1 (Business) | Target pre-merger | 74,078 |
| Item 7 (MD&A) | Target pre-merger | 56,618 |
| Item 1 (Business) | Acquirer post-merger | 140,726 |
| Item 7 (MD&A) | Acquirer post-merger | 144,384 |

### 5.2 Embedding

Each section is embedded using ChronoBERT (`manelalab/chrono-bert-v1-{y}1231`), where the vintage year = announcement year - 1 (no look-ahead). Paragraphs are chunked to 1,024 tokens and mean-pooled to produce one embedding vector per section per deal.

## 6. Preliminary Results

We fit logistic regression (L2-regularized) on five input configurations, using a 75/25 time-based train/test split (seed = 0). Test set: N = 74, with 15 positive cases.

| Model | AUC | Description |
|-------|----:|-------------|
| Baseline: pre-merger habit | 0.656 | Acquirer used integration language pre-deal |
| Baseline: habit + intensity | 0.632 | Habit dummy + pre-merger intensity |
| Baseline: same industry | 0.579 | Same 2-digit SIC dummy |
| **Acquirer only** | **0.637** | Acquirer Item 1 + Item 7 embeddings |
| **Item 1 only** | **0.584** | Acquirer + target Item 1 |
| **Item 7 (MD&A) only** | **0.575** | Acquirer + target Item 7 |
| **Both sections** | **0.590** | Acquirer + target, Item 1 + Item 7 |
| **Acquirer Item 1 only** | **0.567** | Acquirer Item 1 only |

### Preliminary Interpretation

- The pre-merger habit baseline (AUC = 0.656) is competitive, suggesting acquirers with prior integration disclosures are more likely to report future difficulties (consistent with H4).
- Embedding-based models currently underperform the habit baseline. The acquirer-only model (AUC = 0.637) comes closest.
- Adding target text does not clearly help (acquirer-only > both), which is inconsistent with H2 — though the small test set (N = 74, 15 positives) limits power.
- H3 is not supported in this preliminary run: Item 1 and Item 7 perform similarly.
- All models are well above chance (AUC = 0.50), but the small test sample makes these differences statistically unreliable.

## 7. Next Steps

1. **Expand the sample.** Increase the number of deals with complete embeddings (currently 295 of 723 labeled deals).
2. **Combine embeddings with baselines.** Test models that augment habit/industry baselines with embedding features.
3. **Dimensionality reduction.** Apply PCA or use CLS-token embeddings to reduce the feature space relative to the sample size.
4. **Robustness.** Cross-validation, different label windows, alternative label definitions.
5. **Interpretability.** Examine which embedding dimensions or text features drive predictions.

---

*Data sources: Thomson SDC Platinum, SEC EDGAR (10-K filings), WRDS SEC Analytics (CIK–GVKEY link). ChronoBERT models from Manela Lab.*
