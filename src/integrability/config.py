"""Load config.yaml from the project root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # make every path absolute relative to the project root
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def vintage_repo(cfg: dict, announce_year: int) -> str:
    """ChronoBERT version a deal may use: dated the year before announcement."""
    return cfg["model"]["family"].format(y=announce_year - cfg["model"]["lag"])
