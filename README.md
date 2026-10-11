# Merger integrability pilot

See `AGENTS.md` for the goal, rules and pipeline.

## Setup
1. `uv venv && source .venv/bin/activate && uv pip install -e .`
   (or `python -m venv .venv` + `pip install -e .`)
2. Link the data folder to Dropbox (macOS):
   `ln -s ~/Dropbox/<your-folder> data`
   Inside it, create `raw/`, `interim/`, `processed/`.
3. Copy the SDC file into `data/raw/`. Do not edit it.
4. Edit `config.yaml` if any path differs.

## Run
```
python scripts/01_select_deals.py
python scripts/02_link_filings.py
...
```
Deal counts per step land in `outputs/sample_counts.csv`.
