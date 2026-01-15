# 🚀 SENN – Self-Evolving Neural Network **Automatic CNN Architecture Search via Evolutionary Intelligence**

## 📌 Overview

**SENN (Self-Evolving Neural Network)** is a research-grade evolutionary framework that **automatically designs, optimizes, and compresses convolutional neural networks (CNNs)** for image classification.

Instead of manually designing neural architectures, SENN treats each CNN as a **genetic individual** and evolves it over multiple generations using principles inspired by **biological evolution**:

- Mutation  
- Crossover  
- Selection  
- Multi-objective optimization  
- Weight inheritance  
- Structured pruning  

The system progressively discovers CNN architectures that achieve **high accuracy with low computational cost**, without human intervention.

---

## ✨ Key Features

- 🔬 Neural Architecture Search (NAS) via evolution  
- 🧬 CNNs represented as genetic DNA (JSON genotype)  
- ⚖️ Multi-objective optimization (accuracy vs efficiency)  
- 🔁 Weight inheritance for faster convergence  
- ✂️ Structured pruning for model compression  
- 📊 Pareto-optimal selection (NSGA-II)  
- 📈 Rich logging, visualization, and lineage tracking  
- 🖥️ Optional Streamlit dashboard for live monitoring  

---

## 🧠 Core Idea

SENN evolves CNN architectures instead of hand-designing them.

High-level workflow:

```bash
Generate architectures
→ Train briefly
→ Evaluate
→ Select best
→ Mutate / Crossover
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

## 🔀 Crossover (Genetic Recombination)

Crossover combines two parent architectures:

- Early convolution blocks from Parent A  
- Later blocks from Parent B  
- Head inherited from one parent  

This encourages exploration beyond local optima.

---

## 🧠 Weight Inheritance (Warm Start)

To reduce training cost:

- Layers with identical shapes inherit parent weights  
- New or modified layers are randomly initialized  

This significantly accelerates convergence and mimics biological inheritance.

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

→ Mutate + Crossover

→ Weight inheritance

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

### Phase 4 – Crossover
- Genetic recombination

### Phase 5 – Weight Inheritance
- Warm-start children

### Phase 6 – Pruning
- Structured compression

### Phase 7 – Dashboard
- Visualization & interaction

### Phase 8 – Dataset Extension (Optional)
- CIFAR-100
- Custom datasets




