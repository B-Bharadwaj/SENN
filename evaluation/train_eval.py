import torch
import torch.nn as nn
from tqdm import tqdm

def train_one(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    all_preds, all_y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        all_preds.append(pred.cpu())
        all_y.append(y.cpu())
        correct += int((pred == y).sum().item())
        total += x.size(0)
    acc = correct / total
    return total_loss / total, acc, torch.cat(all_y), torch.cat(all_preds)
