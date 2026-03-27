import json
from pathlib import Path

import pandas as pd
import torch

from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evolution.dna_builder import build_model_from_dna
from evolution.dna_schema import ArchitectureDNA
from evaluation.train_eval import evaluate, train_one
from utils.utils import get_device
from pruning.prune_model import prune_model


def latest_run_dir(outputs_dir="outputs") -> Path:
    outputs = Path(outputs_dir)
    runs = [p for p in outputs.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise FileNotFoundError(f"No run_* folders found in {outputs_dir}/")
    return max(runs, key=lambda p: p.stat().st_mtime)


def evaluate_best_model():
    cfg = Config()

    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    run_dir = latest_run_dir("outputs")
    best_json_path = run_dir / "best_architecture.json"
    if not best_json_path.exists():
        raise FileNotFoundError(f"Missing {best_json_path}. Did the evolve run save best_architecture.json?")

    with open(best_json_path, "r") as f:
        best_dna_dict = json.load(f)

    best_dna = ArchitectureDNA.from_dict(best_dna_dict)
    arch_id = best_dna.arch_id()

    print("Loaded:", best_json_path)
    print("Best Architecture ID:", arch_id)

    model = build_model_from_dna(best_dna).to(device)

    pruning_pct = 0.10
    if pruning_pct and pruning_pct > 0:
        model = prune_model(model, pruning_percentage=pruning_pct)
        model = model.to(device)
        print(f"Applied pruning: {pruning_pct*100:.1f}%")

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=(best_dna.lr if best_dna.lr is not None else 1e-3),
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
    )

    input_size = (3, 32, 32)

    history = []

    print(f"\n=== Final retrain for {cfg.final_train_epochs} epochs ===")
    for ep in range(cfg.final_train_epochs):
        train_one(model, train_loader, opt, device)
        _, val_acc, _, _ = evaluate(model, val_loader, device, input_size=input_size)
        print(f"Epoch {ep+1}/{cfg.final_train_epochs} | val_acc={val_acc:.4f}")

        history.append({
            "epoch": ep + 1,
            "val_accuracy": float(val_acc),
        })

    _, test_acc, _, _ = evaluate(model, test_loader, device, input_size=input_size)
    print(f"\n=== FINAL TEST ACCURACY (retrained) === {test_acc:.4f}")

    model_path = run_dir / "best_model_retrained.pth"
    torch.save(model.state_dict(), model_path)
    print("Saved:", model_path)

    history_csv_path = run_dir / "final_retrain_history.csv"
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    print("Saved:", history_csv_path)

    results = {
        "run_dir": str(run_dir),
        "arch_id": arch_id,
        "best_architecture_json": str(best_json_path),
        "pruning_percentage": pruning_pct,
        "final_train_epochs": int(cfg.final_train_epochs),
        "final_test_accuracy": float(test_acc),
        "retrained_model_path": str(model_path),
        "history_csv_path": str(history_csv_path),
        "device": str(device),
        "optimizer": "adamw",
        "learning_rate": float(best_dna.lr if best_dna.lr is not None else 1e-3),
        "weight_decay": float(getattr(cfg, "weight_decay", 1e-4)),
    }

    results_json_path = run_dir / "final_retrain_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", results_json_path)


if __name__ == "__main__":
    evaluate_best_model()