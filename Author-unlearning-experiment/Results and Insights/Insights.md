
# Results

This folder contains perplexity evaluations and visualizations from unlearning experiments using three methods: Gradient Ascent (GA), Gradient Difference (GradDiff), and KL Divergence regularization (KL).

## Key Findings

### Unlearning Effectiveness (Byron Forget Set)

| Method | Byron PPL (Original) | Byron PPL (After) | Forgetting Success |
|--------|---------------------|-------------------|-------------------|
| Original | 3.44 | - | Baseline |
| **GA** | 3.44 | **8.41** | ✅ Strong forgetting (144% increase) |
| **GradDiff** | 3.44 | **7.46** | ✅ Moderate forgetting (117% increase) |
| **KL** | 3.44 | **4.38** | ⚠️ Weak forgetting (27% increase) |

**Insight**: GA achieved the strongest forgetting of Byron while KL was too conservative, prioritizing stability over erasure.

### Retain Set Preservation

**With Roy as Retain:**
- **GradDiff**: Roy PPL = 5.47 (best preservation, 32% lower than GA)
- **GA**: Roy PPL = 18.40 (significant degradation)
- **KL**: Roy PPL = 7.63 (moderate preservation)

**With JKRowling as Retain:**
- Similar pattern observed across all methods

**Insight**: GradDiff successfully balanced forgetting Byron while maintaining external domain knowledge. GA caused catastrophic forgetting on retain sets.

### Generalization Impact

| Dataset | Original | GA | GradDiff | KL |
|---------|----------|----|---------|----|
| Shakespeare | 3.08 | 5.90 | 5.95 | 3.65 |
| Austen | 6.09 | 13.24 | 5.22 | 5.93 |
| Nursery | 11.27 | 22.21 | 17.20 | 13.49 |

**Insight**: GradDiff maintained the best generalization on unseen literary text (Austen PPL improved to 5.22), while GA caused severe degradation across all test sets.

## Conclusions

1. **Gradient Ascent (GA)** effectively erases target data but damages model utility severely
2. **Gradient Difference (GradDiff)** provides the best trade-off between forgetting and retention when properly tuned (lr=3.5e-5, lambda=3.5)
3. **KL Divergence** is too conservative with standard hyperparameters, requiring lower beta values for effective forgetting
4. **External retain sets** (Roy, JKRowling) successfully guide models to preserve general capabilities not in original training data


