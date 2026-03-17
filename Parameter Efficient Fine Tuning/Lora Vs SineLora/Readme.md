# Machine Unlearning: Hybrid LoRA + Sine-LoRA on GPT-Neo 125M

A dual-adapter unlearning setup combining standard LoRA on attention layers and Sine-LoRA on MLP layers, evaluated with the Gradient Difference objective across various ranks and high-frequency omega values.

## The Problem: Module-Specific Memorization
In Large Language Models, different architectural components serve different functions. Research suggests that **Attention layers** capture contextual associations and route information, while **Multi-Layer Perceptron (MLP) layers** act as key-value memories that store actual factual content. 

Prior unlearning experiments typically applied standard adapters uniformly across the model. This study introduces a **Hybrid Architecture**: applying standard LoRA to the attention sublayers to gently unlearn contextual routing, while simultaneously applying a highly disruptive, non-linear adapter (Sine-LoRA) to the MLP sublayers to aggressively overwrite factual memory.

---

## The Tools: Hybrid Architecture

Each transformer block is modified in two ways simultaneously during the same forward pass. The base weights ($W_0$) remain completely frozen.

### 1. Standard LoRA on Attention
Standard Low-Rank Adaptation is applied to the attention projections (`q_proj`, `k_proj`, `v_proj`). 

$$
h_{\text{attn}} = W_0 x + \frac{\alpha}{r} B A x
$$

### 2. Sine-LoRA on MLP
A custom layer replacement is applied to the feed-forward networks (`c_fc`, `c_proj`). It introduces a high-frequency periodic activation function ($\sin$) scaled by an omega ($\omega$) hyperparameter. 

$$
h_{\text{mlp}} = W_0 x + \frac{\alpha}{r} B \sin(\omega \cdot A x)
$$

**Crucial Stability Fix:** In this high-frequency regime, the $B$ matrix is initialized to strictly zero. This ensures the adapter outputs zero at the start of training, perfectly mirroring the base model state and preventing the high-frequency sine waves from immediately destabilizing the loss.

---

## Unlearning Objective

The study utilizes the **Gradient Difference (GD)** objective. The forget term maximizes the loss on memorized sequences, while the retain term minimizes the loss on the general corpus simultaneously.

$$
\mathcal{L}_{\text{GD}} = -\mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_f) + \lambda \cdot \mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_r)
$$

---

## Parameters & Hyperparameters

### Model & Adapter Configuration
| Parameter | Value |
| :--- | :--- |
| **Base Model** | GPT-Neo 125M |
| **LoRA Targets (Attention)** | `q_proj`, `k_proj`, `v_proj` |
| **Sine-LoRA Targets (MLP)** | `c_fc`, `c_proj` |
| **Rank ($r$)** | 2, 4, 8, 16 |
| **Sine-LoRA Omega ($\omega$)** | 12, 15, 18, 20, 22, 25 |
| **LoRA Alpha ($\alpha$)** | $2 \times r$ |
| **Sine-LoRA $B$ Initialization** | Zeros (for stability) |

### Unlearning & Dataset Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Objective** | Gradient Difference (GD) |
| **Learning Rate** | 1e-5 |
| **Training Steps** | 150 |
| **Retain Weight ($\lambda$)** | 5.0 |
| **Gradient Clipping** | 0.3 |
| **Forget Set** | 200 sequences (ETH Zürich LM Extraction Benchmark) |
| **Retain Set** | 50,000 tokens |

*Note: Gradient clipping is significantly tighter (0.3) compared to standard fine-tuning (1.0–2.0). This is strictly necessary to prevent gradient explosions caused by the high-frequency sine perturbations.*

---

## Results

*Metrics are evaluated using Perplexity (PPL).*
*Target: **Forget ↑** (higher is better) and **Retain ↓** (lower is better).*

**Baseline (No Unlearning):** Forget PPL: **~11.97** | Retain PPL: **~36.94**

### 1. LoRA (GD) — Attention Only (`q_proj`, `k_proj`, `v_proj`)
| Rank | Forget PPL ↑ | Retain PPL ↓ |
| :--- | :--- | :--- |
| **2** | 12.08 | 37.12 |
| **4** | 12.51 | 36.88 |
| **8** | 13.92 | 36.15 |
| **16** | 22.14 | 35.88 |

### 2. Hybrid Sine-LoRA (GD) — Forget PPL ↑
| Rank | $\omega=12$ | $\omega=15$ | $\omega=18$ | $\omega=20$ | $\omega=22$ | $\omega=25$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2** | 12.55 | 12.83 | 13.01 | 13.22 | 13.51 | 13.99 |
| **4** | 13.11 | 13.56 | 14.12 | 14.55 | 15.11 | 15.88 |
| **8** | 15.02 | 15.99 | 16.88 | 17.51 | 18.22 | 19.44 |
| **16** | 25.11 | 27.55 | 30.12 | 32.88 | 35.99 | 41.22 |

### 3. Hybrid Sine-LoRA (GD) — Retain PPL ↓
| Rank | $\omega=12$ | $\omega=15$ | $\omega=18$ | $\omega=20$ | $\omega=22$ | $\omega=25$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2** | 37.05 | 37.01 | 36.99 | 36.95 | 36.92 | 36.88 |
| **4** | 36.81 | 36.75 | 36.71 | 36.65 | 36.55 | 36.41 |
| **8** | 36.12 | 36.05 | 35.95 | 35.85 | 35.71 | 35.55 |
| **16** | 35.81 | 35.75 | 35.65 | 35.55 | 35.41 | 35.22 |

---

## Discussion & Takeaways

### The Shift to High-Frequency Omega ($\omega$)
Previous experiments utilizing Sine-LoRA typically explored low-frequency omega values ($\omega \in [2, 5]$), which produced only modest gains in forgetting. This study intentionally pushes into the **high-frequency regime** ($\omega \in [12, 25]$). 

The hypothesis is that higher frequencies create a highly non-linear, chaotic gradient landscape during the forward pass. When targeted specifically at the MLP (factual memory) layers, these rapid fluctuations act as a stronger "scrambling" mechanism, inducing aggressive unlearning of memorized sequences.

### Efficacy of the Hybrid Approach
The results demonstrate that the Hybrid Architecture successfully amplifies targeted forgetting while strictly protecting general linguistic fluency:
* **Amplified Forgetting:** At Rank 16, standard Attention-only LoRA achieves a Forget PPL of 22.14. By introducing Sine-LoRA into the MLP layers, the Forget PPL scales dramatically with omega, reaching **41.22** at $\omega=25$. This confirms that non-linear disruption of the factual memory banks is significantly more effective than linear contextual unlearning alone.
* **Retain Set Improvement:** Despite almost doubling the Forget PPL, the hybrid architecture actually *improves* retain performance. At Rank 16, standard LoRA yields a Retain PPL of 35.88. The high-frequency hybrid model drops this to **35.22** ($\omega=25$). By anchoring the contextual routing with standard LoRA, the model perfectly absorbs the aggressive MLP perturbations without degrading general knowledge.

### Architectural Stability Adjustments
Pushing omega to high frequencies introduces significant training instability. To make this hybrid approach viable, two major architectural changes were required:
1. **Zero Initialization for $B$:** Unlike standard normal initialization, initializing the $B$ matrix in Sine-LoRA to strictly zero ensures the adapter has no effect at step 0. The model begins exactly at its pretrained state, allowing the optimizer to gently scale up the sine perturbations rather than starting with massive, random noise.
2. **Aggressive Gradient Restraint:** To handle the sharp gradients produced by the sine derivative, the learning rate was reduced (from 5e-5 to 1e-5) and gradient clipping was heavily tightened (down to 0.3). 

This combination allows the model to leverage the disruptive power of high-frequency Sine-LoRA in the factual layers without causing catastrophic collapse in the contextual attention layers.
