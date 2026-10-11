"""Step 2. Map gvkey -> CIK; for each deal find the last 10-K of each firm filed before the announcement date, and the acquirer's 10-Ks for label.post_years after completion. Output data/interim/filings.parquet."""
from integrability.config import load
from integrability.counts import log_step


def main():
    cfg = load()
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
