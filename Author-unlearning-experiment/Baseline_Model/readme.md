
# Model

This project uses a character-level Transformer language model based on **Andrej Karpathy's nanoGPT architecture**. The base implementation follows the GPT-2 style decoder-only Transformer with multi-head self-attention and feed-forward layers.

## Architecture Changes from Original

The following hyperparameters were modified from Karpathy's tutorial defaults for this experiment:

| Parameter | Karpathy Default | This Experiment |
|-----------|-----------------|-----------------|
| `batch_size` | 16 | **64** |
| `block_size` | 32 | **256** |
| `n_embd` | 64 | **128** |
| `n_head` | 4 | **4** |
| `n_layer` | 4 | **6** |
| `dropout` | 0.0 | **0.1** |
| `learning_rate` | 1e-3 | **3e-4** |
| `max_iters` | 5000 | **50000** |

## Trained Model

**Version**: `shakespeare_byron_baseline.pt`

- **Training Data**: Shakespeare (70.5%) + Byron (29.5%)
- **Total Parameters**: 1.25M
- **Training Loss**: 1.1359
- **Validation Loss**: 1.4695
- **Vocabulary Size**: 108 characters
- **Training Steps**: 50,000 iterations

This baseline model serves as the starting point for all unlearning experiments (GA, GradDiff, KL).
