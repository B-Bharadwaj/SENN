# senn/logging_metrics.py
from __future__ import annotations
import csv
import os
from typing import Dict, Any

METRICS_HEADER = ["arch_id", "generation", "val_accuracy", "param_count", "flops"]

def init_metrics_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)

def append_metrics_row(path: str, row: Dict[str, Any]) -> None:
    val = row.get("val_accuracy", row.get("val_acc"))
    if val is None:
        raise KeyError("row must include 'val_accuracy' (or 'val_acc')")

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        w.writerow({
            "arch_id": row["arch_id"],
            "generation": int(row["generation"]),
            "val_accuracy": float(val),
            "param_count": int(row["param_count"]),
            "flops": int(row["flops"]),
        })

