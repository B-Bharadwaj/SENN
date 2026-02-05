import json
import torch
from evolution.dna_builder import build_model_from_dna
from evaluation.train_eval import evaluate, train_one
from utils.utils import get_device
from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evolution.dna_schema import ArchitectureDNA
from pruning.prune_model import prune_model 

def evaluate_best_model():
    # Load the best architecture DNA from the saved file
    with open("outputs/best_architecture.json", "r") as f:
        best_dna_dict = json.load(f)

    # Rebuild the model from the DNA
    best_dna = ArchitectureDNA.from_dict(best_dna_dict)
    
    # Print the best architecture ID before starting
    print(f"Best Architecture ID: {best_dna.arch_id()}")

    # Set up device (CUDA if available, else CPU)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model from the best DNA
    model = build_model_from_dna(best_dna).to(device)

    # **Apply pruning to the model here** (e.g., pruning 20% of the weights)
    model = prune_model(model, pruning_percentage=0.1)  # Adjust the pruning percentage as needed

    # Load the training, validation, and test loaders
    cfg = Config()  # Make sure to use your config file to define parameters
    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)

    # Define the optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=(best_dna.lr if best_dna.lr is not None else 1e-3),
        weight_decay=getattr(cfg, "weight_decay", 1e-4)
    )

    # Final retrain or evaluate
    print(f"=== Final retrain for {cfg.final_train_epochs} epochs ===")
    for ep in range(cfg.final_train_epochs):
        train_one(model, train_loader, opt, device)
        _, val_acc, _, _ = evaluate(model, val_loader, device, input_size=(3, 32, 32))
        print(f"Epoch {ep+1}/{cfg.final_train_epochs} | val_acc={val_acc:.4f}")

    # Final test
    _, test_acc, y_true, y_pred = evaluate(model, test_loader, device, input_size=(3, 32, 32))
    print(f"\n=== FINAL TEST ACCURACY (retrained) === {test_acc:.4f}")

    # Save final weights
    torch.save(model.state_dict(), "outputs/best_phase1_model.pth")
    print("Saved: outputs/best_phase1_model.pth")

if __name__ == "__main__":
    evaluate_best_model()
