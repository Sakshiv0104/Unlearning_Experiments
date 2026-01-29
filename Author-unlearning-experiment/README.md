# Unlearning Experiments: Author Unlearning Experiment

This repository presents controlled experiments on **author-level machine unlearning** using a small character-level Transformer language model.  
The objective is to **remove the influence of a specific author (Lord Byron)** while **monitoring and preserving performance** on both co-trained and unseen literary domains.

---

## Motivation

I started this project to understand **how to make a trained language model forget specific training data** without completely destroying its general language ability. In particular, I wanted to explore what it means to "unlearn" a single author (Byron) while still keeping the model useful on other literary domains. This sits at the intersection of machine unlearning, safety, and practical LLM behaviour.

## Acknowledgments

This project builds on **Andrej Karpathy's excellent nanoGPT tutorial** and character-level language modeling framework. The base Transformer architecture, training loop, and educational approach are directly inspired by his work. I highly recommend checking out his resources:

- [nanoGPT GitHub Repository](https://github.com/karpathy/nanoGPT)
- [Let's build GPT: from scratch, in code, spelled out - YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)

All unlearning methods (GA, GradDiff, KL) and experimental design are my own implementation and adaptation.

## Experimental Pipeline

1. **Baseline training** on Shakespeare + Byron  
2. **Unlearning** to remove Byron from the trained model  
3. **Evaluation** on monitored domains using perplexity  

---

## Data Roles and Monitoring Strategy

### Baseline Training Data
Used **only during initial training**:
- **William Shakespeare**
- **Lord Byron**

---

### Forget Set
Used **during unlearning**:
- **Lord Byron**

---

### Monitored Domains (Evaluation Only)

The following datasets are evaluated **after unlearning** to measure forgetting effectiveness and side effects:

- **William Shakespeare** *(co-trained, monitored for collateral damage)*  
- **Arundhati Roy** *(external, unseen during training)*  
- **J.K. Rowling** *(external, unseen during training)*  
- **Jane Austen** *(external, unseen during training)*  
- **Nursery Rhymes** *(external, unseen during training)*  

Although Shakespeare is part of the training data, it is treated purely as a **monitored domain during evaluation**.  
Any degradation on Shakespeare indicates **collateral damage** caused by unlearning Byron.

---

## Model Architecture

The model is a **character-level, decoder-only Transformer**, inspired by Andrej Karpathy’s **nanoGPT** framework.

**Architecture details**:
- Token and positional embeddings
- Multi-head self-attention
- Feed-forward MLP blocks
- Residual connections with LayerNorm
- Character-level vocabulary
- Autoregressive next-token prediction

The model is intentionally small to allow **transparent analysis of unlearning dynamics**.

---

## Hyperparameters

| Parameter        | Value      |
|------------------|------------|
| Layers           | 6          |
| Attention Heads  | 4          |
| Embedding Dim    | 128        |
| Context Length   | 256        |
| Batch Size       | 64         |
| Dropout          | 0.1        |
| Optimizer        | AdamW      |
| Learning Rate    | 3e-4       |
| Training Steps   | 50,000     |
| Parameters       | ~1.25M     |

---


## Unlearning Methods

All methods start from the **same baseline checkpoint**.

- **Gradient Ascent (GA)**: maximizes loss on Byron  
- **Gradient Difference (GradDiff)**: ascent on Byron + descent on a monitored domain  
- **KL-Regularized Unlearning (KL)**: penalizes divergence from the baseline on monitored domains  

---

## Evaluation Metric

All results are reported using **perplexity (PPL)**.

- Higher PPL → worse performance  
- Lower PPL → better performance  

---

## Results: Perplexity Across Domains

| Method    | Byron ↑ | Shakespeare | Roy | Rowling | Austen | Nursery |
|-----------|---------|-------------|-----|---------|--------|---------|
| Baseline  | 32      | 29          | 45  | 48      | 42     | 40      |
| GA        | **120** | 85          | 110 | 118     | 97     | 102     |
| GradDiff | **78**  | 36          | 47  | 50      | 45     | 44      |
| KL        | 55      | 31          | **44** | **46** | **41** | **39** |

*(↑ Higher values are desirable only for the forget domain)*

---

## Key Insights

- **Gradient Ascent (GA)** produces the strongest forgetting of Byron but causes severe degradation across all monitored domains, including Shakespeare.
- **Gradient Difference (GradDiff)** achieves a better balance: Byron is forgotten while performance on Shakespeare and unseen authors is largely preserved.
- **KL-based unlearning** is the most stable method, preserving behaviour on monitored domains, but may under-unlearn Byron unless carefully tuned.
- Monitoring **Shakespeare** helps distinguish *targeted forgetting* from *global model damage*, while external authors test generalization stability.

---

## Takeaway

These experiments show that:
- Targeted author unlearning is feasible in small language models
- Forgetting strength and model utility are fundamentally in tension
- Explicit monitoring of both co-trained and unseen domains is essential
- Gradient Difference methods offer the most practical trade-off

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


