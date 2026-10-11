"""Append deals-in / deals-out for each pipeline step to outputs/sample_counts.csv."""
import csv
from datetime import datetime
from .config import ROOT

FILE = ROOT / "outputs" / "sample_counts.csv"


def log_step(step: str, n_in: int, n_out: int, note: str = "") -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not FILE.exists()
    with open(FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "step", "deals_in", "deals_out", "dropped", "note"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), step, n_in, n_out, n_in - n_out, note])
    print(f"[{step}] {n_in} -> {n_out} ({n_in - n_out} dropped) {note}")
