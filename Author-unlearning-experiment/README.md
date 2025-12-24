# Unlearning Experiments: Author Unlearning Experiment

## Motivation

I started this project to understand **how to make a trained language model forget specific training data** without completely destroying its general language ability. In particular, I wanted to explore what it means to "unlearn" a single author (Byron) while still keeping the model useful on other literary domains. This sits at the intersection of machine unlearning, safety, and practical LLM behaviour.

## Acknowledgments

This project builds on **Andrej Karpathy's excellent nanoGPT tutorial** and character-level language modeling framework. The base Transformer architecture, training loop, and educational approach are directly inspired by his work. I highly recommend checking out his resources:

- [nanoGPT GitHub Repository](https://github.com/karpathy/nanoGPT)
- [Let's build GPT: from scratch, in code, spelled out - YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)

All unlearning methods (GA, GradDiff, KL) and experimental design are my own implementation and adaptation.

## Data

I work with several literary text sources, grouped by their role in training and evaluation:

- **Training data (for the baseline model)**  
  - Shakespeare  
  - Byron  

- **Forget set**  
  - **Byron**: Author whose style/content I explicitly try to remove from the model.

- **External retain / monitoring domains**  
  - Arundhati **Roy**  
  - **J.K. Rowling**  
  - **Jane Austen**  
  - **Nursery Rhymes**

Shakespeare and Byron are used to train the baseline model. Byron is later treated as the **forget set**. Roy, Rowling, Austen, and Nursery Rhymes are **not** part of the original training data; they are used as **retain-style domains** and **test sets** to see how much general ability the model preserves while Byron is being unlearned.

## Model

The model is a **character-level Transformer language model** inspired by Andrej Karpathy's nanoGPT-style architecture:

- Decoder-only Transformer
- Multi-head self-attention
- Feed-forward MLP blocks
- Learned token and position embeddings
- LayerNorm + residual connections
- Character-level vocabulary

### Hyperparameters

Baseline hyperparameters used for the core experiments:

- `batch_size`: **64**
- `block_size`: **256**
- `n_embd`: **128**
- `n_head`: **4**
- `n_layer`: **6**
- `dropout`: **0.1**
- `learning_rate`: **3e-4**
- `max_iters`: **50,000**
- Parameters: ~**1.25M**
- Final training loss: ~**1.14**
- Final validation loss: ~**1.47**

The baseline checkpoint (Shakespeare + Byron) is saved as:

- `shakespeare_byron_baseline.pt` – this is the starting point for all unlearning methods.

## Unlearning Methods

I experiment with three gradient-based unlearning strategies, all starting from the same baseline model:

1. **Gradient Ascent (GA)**  
   - Maximize cross-entropy loss on the **forget set** (Byron).  
   - Pure "push away" behaviour: the model is explicitly trained to become worse on Byron.

2. **Gradient Difference (GradDiff)**  
   - Combine **gradient ascent** on the forget set with **gradient descent** on a retain-style dataset (e.g., Roy or J.K. Rowling).  
   - The retain term encourages the model to stay good on external domains while forgetting Byron.

3. **KL-Based Unlearning (KL)**  
   - Use a frozen **reference model** (the original baseline) and add a **KL divergence regularizer** on the retain-style data.  
   - This tries to push the model away from Byron while keeping it close to the original behaviour on retain data.

All methods are implemented at the **character level** and trained with standard optimizers (AdamW) plus gradient clipping for stability.

## Metric: Perplexity

The main quantitative metric is **perplexity (PPL)**, computed on fixed text windows for each domain:

- **Lower PPL** = model is more confident / better at predicting that domain.
- **Higher PPL** = model is worse on that domain (potentially indicating successful forgetting for the target author, or harmful degradation elsewhere).

I compute PPL on:

- **Forget domain**: Byron  
- **Original training companion**: Shakespeare  
- **External / retain-style domains**: Roy, J.K. Rowling, Austen, Nursery Rhymes  

This gives a **matrix of PPL values** for each method × dataset, which is stored in CSV form and visualized with line plots.

## Results and Insights

### Forgetting Byron

- **Gradient Ascent (GA)**  
  - Byron PPL increases significantly compared to the original model.  
  - This indicates **strong forgetting** of the target author.  
  - However, GA also raises PPL noticeably on Shakespeare and most external domains, showing substantial damage to overall utility.

- **Gradient Difference (GradDiff)**  
  - Byron PPL increases (successful forgetting), though usually less aggressively than GA.  
  - PPL on external domains (Roy, J.K. Rowling, Austen) is much better preserved.  
  - This method often achieves the **best trade-off** between forgetting Byron and keeping the model useful on other text.

- **KL-Based Unlearning (KL)**  
  - Byron PPL increases only mildly, meaning the forgetting effect is weaker unless hyperparameters (like `beta`) are carefully tuned.  
  - In return, PPL on retain-style domains is very stable, making KL comparatively **conservative but safe**.

### Retain-Style Performance (Roy / J.K. Rowling)

When using **Roy** or **J.K. Rowling** as retain-style domains in the loss:

- **GradDiff** tends to keep their PPL relatively close to the original model while still increasing PPL on Byron.  
- **GA** often inflates PPL on both forget and retain domains, showing that pure ascent is too destructive for fine-grained control.  
- **KL** keeps retain-domain PPL low but requires careful tuning to achieve meaningful forgetting of Byron.

### Generalization to Other Authors

On **Austen** and **Nursery Rhymes**:

- GA frequently harms performance across the board, suggesting **catastrophic degradation**.
- GradDiff usually maintains the best balance, with moderate forgetting and relatively good generalization.
- KL remains safest but can under-unlearn if the regularization is too strong.

## Summary

In this project, I:

- Trained a **character-level Transformer** on Shakespeare + Byron (based on Andrej Karpathy's nanoGPT framework).  
- Designed and implemented three **unlearning methods** (GA, GradDiff, KL).  
- Evaluated them using **perplexity** on multiple literary domains.  
- Observed a clear trade-off:
  - GA → strong forgetting, poor utility.  
  - GradDiff → best balance between forgetting and retention.  
  - KL → stable but sometimes too gentle, requiring tuning.

The code, data descriptions, model details, and result CSVs/plots in this repository are meant to serve as a small, reproducible testbed for **author-level machine unlearning** in language models.

## Citation

If you use this work or build upon these experiments, please cite this repository:

@misc{Unlearning_Experiments,
author = {Sakshi Vishwakarma},
title = {Unlearning Experiments: Author_Unlearning_Experiment},
year = {2025},
publisher = {GitHub},
url = {https://github.com/Sakshiv0104/Unlearning_Experiments/tree/main/Author-unlearning-experiment}
}

## License

This project is released under the MIT License. Feel free to use, modify, and distribute with attribution.

---

**Built with inspiration from Andrej Karpathy's nanoGPT. Unlearning methodology and experiments designed independently.**



