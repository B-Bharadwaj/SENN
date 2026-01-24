# 🚀 SENN – Self-Evolving Neural Network **Automatic CNN Architecture Search via Evolutionary Intelligence**

## 📌 Overview

**SENN (Self-Evolving Neural Network)** is a research-grade evolutionary framework that **automatically discovers, optimizes, and compresses convolutional neural network (CNN) architectures** for image classification.

Instead of manually designing neural architectures, SENN represents each CNN as a **genetic individual encoded by a deterministic Architecture DNA** and evolves architectures over multiple generations using a **mutation-driven evolutionary process** combined with rigorous selection and evaluation.

The evolutionary process in SENN is based on:

- Mutation-based architecture evolution  
- Selection of high-performing architectures  
- Multi-objective optimization (accuracy vs efficiency)  
- Strict architecture reproducibility via DNA  
- Structured pruning for model compression  

Manual architecture design is entirely avoided.  
Genetic crossover and weight inheritance are **intentionally excluded** to preserve architectural validity, deterministic reconstruction, and implementation robustness.

As evolution progresses, SENN discovers CNN architectures that achieve **strong accuracy while remaining parameter- and computation-efficient**, without human intervention.


---


## ✨ Key Features

- 🔬 Mutation-driven Neural Architecture Search (NAS) 
- 🧬 Deterministic Architecture DNA (JSON-based genotype) 
- ⚖️ Multi-objective optimization using Pareto dominance  
- 🔁 Structured pruning for efficiency improvement
- ✂️ Safe model construction with pre-training validation     
- 📊 Full lineage and mutation tracking   
- 📈 Reproducible architecture reconstruction 
- 🖥️ Optional Streamlit dashboard for live monitoring

---

## 🧠 Core Idea

SENN evolves CNN architectures instead of hand-designing them.

High-level workflow:

```bash
Generate architectures
→ Train briefly
→ Evaluate
→ Select
→ Mutate DNA
→ Next generation
→ Repeat
```

Over generations, the population improves just like biological evolution — discovering architectures that balance **performance and efficiency**.

---

## 📂 Dataset & Preprocessing

### Dataset
- **Primary:** CIFAR-10  
  - 32×32 RGB images  
  - 10 classes  
- **Optional Extension:** CIFAR-100  

### Preprocessing Pipeline
- Tensor conversion  
- Normalization  
- Optional data augmentation:
  - Random crop
  - Horizontal flip  

### Data Splits
- Training set  
- Validation set  
- Test set  

---

## 🧬 Architecture Search Space (CNN DNA)

Each CNN is encoded as a **genotype (architecture DNA)** stored in JSON format.

### CNN Constraints

| Component | Options |
|--------|--------|
| Conv layers | 2 – 6 |
| Filters | 16 / 32 / 64 / 128 |
| Kernel sizes | 3×3, 5×5 |
| Activations | ReLU, LeakyReLU |
| Pooling | MaxPool, AvgPool, None |
| Normalization | Optional BatchNorm |
| Regularization | Optional Dropout |
| Head | Global Average Pooling + Dense |
| Model size | Small–medium CNNs |

The constrained search space ensures **valid, trainable architectures** while allowing rich diversity.

---

## ⚙️ Evolution Configuration

| Parameter | Typical Value |
|--------|--------|
| Population size | 8–12 |
| Generations | 10–20 |
| Survivors | Top-K / Pareto front |
| Training per model | 2–3 epochs |
| Total models evaluated | 100+ |

Short training during evolution allows efficient evaluation of many architectures.

---

## 🏋️ Training Strategy

### During Evolution
- Few epochs (2–3)
- Goal: estimate architectural potential
- Prevents overfitting and saves compute

### After Evolution
- Best architecture(s) fully trained
- 30–50 epochs
- Final evaluation on test set

---

## 📐 Fitness & Evaluation Metrics

SENN uses **multi-objective evaluation**.

### Primary Metrics
- Validation accuracy  
- Validation loss  

### Efficiency Metrics
- Number of parameters  
- FLOPs  
- Inference latency (optional)  

### Fitness Logic
- Early generations: weighted fitness  
- Later generations: Pareto optimization  

This prevents evolution from favoring large, inefficient models.

---

## 🏆 Selection Mechanisms

### Basic Selection
- Rank by fitness
- Select top-K models

### Advanced Selection
- Pareto front extraction
- **NSGA-II**
  - Non-dominated sorting
  - Crowding distance for diversity

Selected models become **parents** for the next generation.

---

## 🔁 Mutation Engine (Core Evolution)

Mutation introduces controlled randomness.

### Structural Mutations
- Add / remove convolution layers  
- Increase / decrease filters  
- Change kernel sizes  
- Toggle BatchNorm / Dropout  
- Change pooling strategy  
- Modify dense layer size  
- Adjust learning rate  

All mutations are **constraint-aware**, ensuring valid CNNs.

---

## ✂️ Pruning (Model Compression)

SENN integrates pruning for efficiency.

### Pruning Strategies
- Filter/channel reduction via mutation  
- L1-norm based channel pruning  
- Post-training pruning on survivors  

Result: **smaller, faster models with minimal accuracy loss**.

---

## 🔄 Full Evolution Loop

### Initialize population
→ Train

→ Measure (accuracy, params, FLOPs, latency)

→ Select (Pareto / NSGA-II)

→ Mutate DNA

→ Prune

→ Validate

→ Next generation


Repeated for **N generations**.

---

## 🏁 Final Model Selection

At the end of evolution:

- Extract Pareto-optimal architectures  
- Fully train best candidates  
- Evaluate on test set  

### Example Result
- CIFAR-10 accuracy: ~77–80%+  
- Reduced parameters and FLOPs vs baseline CNN  

---

## 📁 Outputs & Artifacts

### Model Files
- `best_model.pth`  
- `best_arch.json`  

### Logs
- `evolution_metrics.csv`  
- `lineage.csv` (parent → child)  
- Mutation history  

### Visualizations
- Accuracy vs generation  
- Pareto fronts  
- Params/FLOPs vs accuracy  
- Confusion matrix  
- Training curves  

---

## 🖥️ Dashboard & Demo (Optional)

A **Streamlit dashboard** provides:

- Live evolution progress  
- Best architecture summary  
- Pareto front visualization  
- Architecture comparison table  
- Download links for models & DNA  

This transforms SENN from a research prototype into a **usable system**.

---

## 🛠️ Phase-Wise Implementation Plan

### Phase 0 – Baseline Evolution (MVP)
- Population
- Mutation
- Selection
- Training loop

### Phase 1 – Architecture DNA
- JSON genotype
- Safe model builder
- Logging

### Phase 2 – Multi-Objective Optimization
- Pareto fronts
- NSGA-II

### Phase 3 – Efficiency Metrics
- Params
- FLOPs
- Latency

### Phase 4 – Pruning
- Structured compression

### Phase 5 – Dashboard
- Visualization & interaction





