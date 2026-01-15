import matplotlib.pyplot as plt
from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evolution.evolve import evolve
from models.model_space import build_model
from evaluation.train_eval import evaluate
from utils.utils import set_seed, get_device, ensure_dir
import torch
torch.backends.cudnn.benchmark = True

def plot_history(history, out_path):
    gens = [h["gen"]+1 for h in history]
    accs = [h["best_acc"] for h in history]
    plt.figure()
    plt.plot(gens, accs)
    plt.xlabel("Generation")
    plt.ylabel("Best Val Accuracy")
    plt.title("SENN: Accuracy vs Generation")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dir("outputs")

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)

    best_genome, best_fit, (best_acc, best_params), history = evolve(cfg, train_loader, val_loader, device)

    print("\n=== BEST GENOME ===")
    print(best_genome)
    print(f"Best fitness: {best_fit:.6f} | best val acc: {best_acc:.4f} | params: {best_params}")

    plot_history(history, "outputs/acc_vs_gen.png")
    print("Saved: outputs/acc_vs_gen.png")

    # Optional: evaluate best on test
    model = build_model(best_genome, num_classes=cfg.num_classes).to(device)
    # NOTE: For a fair test, you’d normally retrain best model longer.
    # But in scope, you can report quick test after brief training (or retrain for a few more epochs).
    _, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"Quick (untrained) test acc (expected low): {test_acc:.4f}")
    print("Tip: retrain best model for 10-20 epochs, then compute final test acc + confusion matrix.")

if __name__ == "__main__":
    main()
