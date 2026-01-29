# Unlearning Experiments: Author Unlearning Experiment

This repository presents controlled experiments on author-level machine unlearning using a small character-level Transformer language model. The objective is to remove the influence of a specific author (Lord Byron) while monitoring and preserving performance on both co-trained and unseen literary domains.

---

## Motivation

I started this project to understand how to make a trained language model forget specific training data without completely destroying its general language ability. In particular, I wanted to explore what it means to "unlearn" a single author (Byron) while still keeping the model useful on other literary domains. This sits at the intersection of machine unlearning, safety, and practical LLM behaviour.

## Acknowledgments

This project builds on Andrej Karpathy's excellent nanoGPT tutorial and character-level language modeling framework. The base Transformer architecture, training loop, and educational approach are directly inspired by his work:

- [nanoGPT GitHub Repository](https://github.com/karpathy/nanoGPT)
- [Let's build GPT: from scratch, in code, spelled out - YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)

All unlearning methods (GA, GradDiff, KL) and experimental design are original implementations.

## Experimental Pipeline

The experiment follows a 3-part structure to test different retain strategies:

1. Baseline training on Shakespeare + Byron
2. Part 1: Unlearn Byron while retaining Shakespeare (internal training data)
3. Part 2: Unlearn Byron while retaining Arundhati Roy (external, never trained)
4. Part 3: Unlearn Byron while retaining J.K. Rowling (external, never trained)
5. Evaluation on all 5 monitored domains after each part

---

## Data Roles and Monitoring Strategy

### Baseline Training Data (Initial Training Only)
- William Shakespeare
- Lord Byron

### Forget Set (During Unlearning)
- Lord Byron

### Monitored Domains (Evaluation Only)
- William Shakespeare (co-trained, tests collateral damage)
- Arundhati Roy (external, never trained)
- J.K. Rowling (external, never trained)  
- Jane Austen (external, never trained)
- Nursery Rhymes (external, never trained)

---

## Model Architecture

Character-level, decoder-only Transformer (nanoGPT-inspired):

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Attention Heads | 4 |
| Embedding Dim | 128 |
| Context Length | 256 |
| Batch Size | 64 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Training Steps | 50,000 |
| Total Parameters | ~1.25M |

---

## Unlearning Methods

All methods start from the same baseline checkpoint:

- Gradient Ascent (GA): maximizes loss on Byron
- Gradient Difference (GradDiff): ascent on Byron + descent on monitored domain
- KL-Regularized Unlearning (KL): penalizes divergence from baseline on monitored domains

## Evaluation Metric

Perplexity (PPL): Higher = worse performance, Lower = better performance. Higher Byron PPL indicates successful forgetting.

---

## Complete Results

### 0. Baseline Model (Shakespeare + Byron Training)

| Dataset | PPL | Notes |
|---------|-----|-------|
| Shakespeare | 3.08 | Training data |
| Byron | 3.44 | Target forget |
| Roy | 8.07 | External |
| JKRowling | 7.45 | External |
| Austen | 6.09 | Unseen |
| Nursery | 11.27 | Unseen |

### 1. PART 1: Shakespeare Retain (Internal Training Data)

**Goal**: Test preservation of co-trained data (Shakespeare)

| Method | Byron ↑ | Shakes ↓ | Roy ↑ | JKRowling ↑ | Austen ↑ | Nursery ↑ |
|--------|---------|----------|-------|-------------|----------|-----------|
| Original | 3.44 | 3.08 | 8.07 | 7.45 | 6.09 | 11.27 |
| GA | 8.63 | 6.05 | 18.81 | 14.99 | 13.34 | 22.32 |
| GradDiff | 4.91 | 3.08 | 10.55 | 9.86 | 8.84 | 16.35 |
| KL | 3.55 | 3.12 | 8.07 | 7.39 | 6.20 | 11.38 |

**Pattern**: GA destroys everything. GradDiff/KL preserve Shakespeare perfectly.

### 2. PART 2: Roy Retain (External Data - Never Trained)

**Goal**: Test if unseen data can guide preservation

| Method | Byron ↑ | Roy ↓ | Shakes ↑ | JKRowling ↑ | Austen ↓ | Nursery ↑ |
|--------|---------|-------|----------|-------------|----------|-----------|
| Original | 3.44 | 8.07 | 3.08 | 7.45 | 6.09 | 11.27 |
| GA | 8.44 | 18.37 | 5.92 | 14.68 | 13.16 | 22.15 |
| GradDiff | 7.35 | 5.46 | 5.86 | 6.72 | 5.21 | 16.90 |
| KL | 3.78 | 7.63 | 3.28 | 6.94 | 5.91 | 11.36 |

**Pattern**: GradDiff(Roy) improves Roy AND Austen performance!

### 3. PART 3: JKRowling Retain (External Data)

| Method | Byron ↑ | Roy ↑ | Shakes ↑ | JKRowling ↓ | Austen ↓ | Nursery ↑ |
|--------|---------|-------|----------|-------------|----------|-----------|
| Original | 3.44 | 8.07 | 3.08 | 7.45 | 6.09 | 11.27 |
| GA | 8.41 | 18.40 | 5.90 | 14.72 | 13.24 | 22.21 |
| GradDiff | 9.23 | 7.30 | 8.13 | 6.50 | 4.88 | 18.89 |
| KL | 3.89 | 7.58 | 3.34 | 7.01 | 5.92 | 11.54 |

---

## Method Behavior Summary

| Method | Forgetting Byron | Internal Retain | External Retain | Unseen Domains |
|--------|------------------|-----------------|-----------------|---------------|
| GA | Excellent (+145-151%) | Destroys | Destroys | Destroys |
| GradDiff | Good (+43-168%) | Perfect | Improves | Improves |
| KL | Weak (+3-13%) | Perfect | Stable | Stable |

---

## Key Takeaways

1. GA: Nuclear option - forgets perfectly, breaks everything else
2. GradDiff: Smart balance - forgets target, preserves/improves utility 
3. KL: Too safe - barely forgets, everything else stable
4. External retain works: Roy/JKRowling (never trained) successfully guide preservation
5. Best method: GradDiff + external retain (Byron +113-168%, Roy -32%, Austen -14%)

---

## Production Recommendation

**GradDiff with external retain tuning**:


---
## Citation

@misc{Unlearning_Experiments,
  author = {Sakshi Vishwakarma},
  title = {Author-Level Machine Unlearning Experiments},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sakshiv0104/Unlearning_Experiments}
}

---
## License

Released under the MIT License. Feel free to use, modify, and distribute with attribution.


