import copy, random
import torch
from models.model_space import random_genome, build_model, ACTS, POOLS
from evaluation.train_eval import train_one, evaluate
from utils.utils import count_params

def fitness(acc: float, n_params: int, lam: float) -> float:
    return acc - lam * n_params

def init_population(P: int):
    return [random_genome() for _ in range(P)]

def mutate(genome):
    g = copy.deepcopy(genome)

    # pick mutation type
    m = random.choice([
        "add_block", "remove_block", "change_filters", "change_kernel",
        "toggle_bn", "change_drop", "change_act", "change_pool", "change_lr"
    ])

    blocks = g["blocks"]

    if m == "add_block" and len(blocks) < 6:
        blocks.insert(random.randint(0, len(blocks)), {
            "out": random.choice([16, 32, 64, 96, 128]),
            "k": random.choice([3, 5]),
            "act": random.choice(ACTS),
            "bn": random.choice([True, False]),
            "drop": random.choice([0.0, 0.1, 0.2, 0.3]),
            "pool": random.choice(POOLS),
        })

    elif m == "remove_block" and len(blocks) > 2:
        del blocks[random.randrange(len(blocks))]

    else:
        b = blocks[random.randrange(len(blocks))]
        if m == "change_filters":
            b["out"] = random.choice([16, 32, 48, 64, 96, 128])
        elif m == "change_kernel":
            b["k"] = random.choice([3, 5])
        elif m == "toggle_bn":
            b["bn"] = not b["bn"]
        elif m == "change_drop":
            b["drop"] = random.choice([0.0, 0.1, 0.2, 0.3])
        elif m == "change_act":
            b["act"] = random.choice(ACTS)
        elif m == "change_pool":
            b["pool"] = random.choice(POOLS)
        elif m == "change_lr":
            g["opt"]["lr"] = float(g["opt"]["lr"]) * random.choice([0.5, 0.8, 1.25, 1.5])
            g["opt"]["lr"] = max(1e-5, min(3e-2, g["opt"]["lr"]))

    return g

def evolve(cfg, train_loader, val_loader, device):
    pop = init_population(cfg.population_size)
    history = []

    best = None
    best_fit = -1e9
    best_metrics = None

    for gen in range(cfg.generations):
        scored = []
        print(f"\n=== Generation {gen+1}/{cfg.generations} ===")

        for i, genome in enumerate(pop):
            model = build_model(genome, num_classes=cfg.num_classes).to(device)
            print("Model device:", next(model.parameters()).device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=genome["opt"]["lr"],
                weight_decay=genome["opt"]["weight_decay"]
            )

            for _ in range(cfg.train_epochs):
                train_one(model, train_loader, opt, device)

            _, val_acc, y_true, y_pred = evaluate(model, val_loader, device)
            n_params = count_params(model)
            fit = fitness(val_acc, n_params, cfg.size_penalty_lambda)

            scored.append((fit, val_acc, n_params, genome))
            print(f"Model {i+1}: acc={val_acc:.4f} params={n_params} fit={fit:.6f}")

        scored.sort(key=lambda x: x[0], reverse=True)
        top_fit, top_acc, top_params, top_genome = scored[0]
        history.append({"gen": gen, "best_acc": top_acc, "best_fit": top_fit, "best_params": top_params})

        if top_fit > best_fit:
            best_fit = top_fit
            best = copy.deepcopy(top_genome)
            best_metrics = (top_acc, top_params)

        elites = [g for (_, _, _, g) in scored[:cfg.elite_k]]

        # next population: elites + mutated children
        new_pop = [copy.deepcopy(e) for e in elites]
        while len(new_pop) < cfg.population_size:
            parent = random.choice(elites)
            child = mutate(parent)
            new_pop.append(child)

        pop = new_pop

    return best, best_fit, best_metrics, history
