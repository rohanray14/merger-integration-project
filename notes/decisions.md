# Decisions log

- 2026-10-09: Pilot sample = completed public-public deals announced 2000-2004 (first five ChronoBERT vintages). Model vintage = announce year - 1.
- 2026-10-09: Label = H&P 2018 p.17 keyword rule on acquirer 10-Ks 1-3 years post-completion, excluding risk factors. Window length to confirm with Gordon/Will.
- 2026-10-16: Pilot 2 — expand sample to 2000–2007. Train on 75% random from 2000–2004, two test sets: 25% holdout from 2000–2004 (test A) and matched-size random from 2005–2007 (test B).
- 2026-10-16: Pilot 2 — add cosine similarity and difference vector (acq−tgt) as alternative representations alongside concatenation. All use same ChronoBERT mean-pooled embeddings.
- 2026-10-16: Pilot 2 — add announcement returns (acquirer CARs) as second outcome measure, following H&P Table VI.
- 2026-10-16: Keep all-4-sections requirement so every model is evaluated on the exact same set of deals.
