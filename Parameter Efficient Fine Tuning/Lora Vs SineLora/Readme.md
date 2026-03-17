# Machine Unlearning: Hybrid LoRA + Sine-LoRA-MLP on GPT-Neo 125M

A dual-adapter unlearning setup combining standard LoRA on attention layers and Sine-LoRA on MLP layers, evaluated with the Gradient Difference objective across various ranks and high-frequency omega values.

## Motivation

This notebook tests a hybrid architectural approach to parameter-efficient machine unlearning. The hypothesis is that attention and MLP layers play fundamentally different roles in memorization: attention captures contextual associations and routing, while MLP layers act as key-value memories storing factual content. 

By applying a standard linear adapter (LoRA) to the attention sublayer and a highly non-linear, periodic adapter (Sine-LoRA) to the MLP sublayer simultaneously, this setup aims to achieve stronger, more controlled unlearning than utilizing a single adapter type across all layers. This experiment specifically evaluates the efficacy of high-frequency perturbations ($\omega \in [12, 25]$) within the MLP layers.

---

## Architecture

Each transformer block is modified in two ways simultaneously during the forward pass. The base weights ($W_0$) remain completely frozen.

### 1. Standard LoRA on Attention (via PEFT)
Targets: `q_proj`, `k_proj`, `v_proj`

$$
h_{\text{attn}} = W_0 x + \frac{\alpha}{r} \cdot B A x
$$

### 2. Sine-LoRA on MLP (Custom Layer Replacement)
Targets: `c_fc`, `c_proj`

$$
h_{\text{mlp}} = W_0 x + \frac{\alpha}{r} \cdot \sin(\omega \cdot A B) x
$$

Where $A \in \mathbb{R}^{d_{\text{in}} \times r}$, $B \in \mathbb{R}^{r \times d_{\text{out}}}$, and $\omega$ is the frequency hyperparameter. Both adapters share the same rank $r$ and scaling $\alpha = 2r$. 

**Stability Fix:** The $B$ matrix in the Sine-LoRA adapter is initialized to zero so the model starts exactly at the base model state, preventing the high-frequency sine waves from destabilizing the early training steps.

---

## Unlearning Objective — Gradient Difference (GD)

The forget term maximizes the cross-entropy loss on memorized sequences ($\mathcal{D}_f$), while the retain term minimizes the loss on the general corpus ($\mathcal{D}_r$) simultaneously.

$$
\mathcal{L}_{\text{GD}} = -\mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_f) + \lambda \cdot \mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_r)
$$

---

## Parameters & Hyperparameters

### Model Configuration
| Parameter | Value |
| :--- | :--- |
| **Base model** | GPT-Neo 125M |
| **LoRA targets (attention)** | `q_proj`, `k_proj`, `v_proj` |
| **Sine-LoRA targets (MLP)** | `c_fc`, `c_proj` |
| **$\alpha$** | $2 \times r$ |
| **LoRA dropout** | 0.05 |
| **Sine-LoRA B init** | Zeros (for stability) |
| **Optimizer** | AdamW |
| **Seed** | 42 |

### Dataset Configuration
| Parameter | Value |
| :--- | :--- |
| **Forget set** | 200 sequences × 200 tokens |
| **Retain set** | 50,000 tokens |
| **Source** | ETH Zürich LM Extraction Benchmark |
| **Batch size** | 4 |
| **Sequence length** | 200 tokens |
| **PPL window / stride** | 512 / 256 tokens |

### Experiment Grid & GD Hyperparameters
| Hyperparameter | Values |
| :--- | :--- |
| **Rank $r$** | 2, 4, 8, 16 |
| **Omega $\omega$ (Sine-LoRA)** | 12, 15, 18, 20, 22, 25 |
| **Learning Rate** | 1e-5 |
| **Training Steps** | 150 |
| **Retain Weight ($\lambda$)** | 5.0 |
| **Gradient Clipping** | 0.3 |

*Note: Gradient clipping is set very tight (0.3) compared to standard fine-tuning. This is strictly necessary to prevent gradient explosions caused by the high-frequency sine perturbations in the MLP layers.*

---

## Results

*Metrics are evaluated using Perplexity (PPL).*
*Target: **Forget ↑** (higher is better) and **Retain ↓** (lower is better).*

**Baseline (No Unlearning):** Forget PPL: **11.97** | Retain PPL: **36.94**

### 1. LoRA (GD) — Attention Only (`q_proj`, `k_proj`, `v_proj`)
| Rank | Forget PPL ↑ | Retain PPL ↓ |
| :--- | :--- | :--- |
| 2 | 12.08 | 37.12 |
| 4 | 12.51 | 36.88 |
| 8 | 13.92 | 36.15 |
| 16 | 22.14 | 35.88 |

### 2. Hybrid Sine-LoRA (GD) — Forget PPL ↑
| Rank | $\omega=12$ | $\omega=15$ | $\omega=18$ | $\omega=20$ | $\omega=22$ | $\omega=25$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | 12.55 | 12.83 | 13.01 | 13.22 | 13.51 | 13.99 |
| 4 | 13.11 | 13.56 | 14.12 | 14.55 | 15.11 | 15.88 |
| 8 | 15.02 | 15.99 | 16.88 | 17.51 | 18.22 | 19.44 |
| 16 | 25.11 | 27.55 | 30.12 | 32.88 | 35.99 | 41.22 |

### 3. Hybrid Sine-LoRA (GD) — Retain PPL ↓
| Rank | $\omega=12$ | $\omega=15$ | $\omega=18$ | $\omega=20$ | $\omega=22$ | $\omega=25$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | 37.05 | 37.01 | 36.99 | 36.95 | 36.92 | 36.88 |
| 4 | 36.81 | 36.75 | 36.71 | 36.65 | 36.55 | 36.41 |
| 8 | 36.12 | 36.05 | 35.95 | 35.85 | 35.71 | 35.55 |
| 16 | 35.81 | 35.75 | 35.65 | 35.55 | 35.41 | 35.22 |

---

## Discussion & Analysis

The data generated from this specific notebook validates the hybrid architecture approach and demonstrates clear scaling behaviors within the high-frequency omega regime. 

### 1. Superiority of the Hybrid Architecture
Targeting both the contextual routing (attention) and the factual memory (MLP) yields significantly stronger unlearning than adapting attention modules alone. 
* At Rank 16, the standard Attention-only LoRA reaches a Forget PPL of 22.14. 
* By integrating the non-linear Sine-LoRA into the MLP layers, the Forget PPL at the highest frequency ($\omega=25$) nearly doubles to **41.22**. 
This confirms the hypothesis that the factual storage inside the MLP layers requires targeted, highly disruptive adaptation to successfully "overwrite" memorized data.

### 2. Efficacy of High-Frequency Perturbations
Within the hybrid architecture, increasing the frequency of the sine perturbation ($\omega$) directly and consistently scales the unlearning efficacy across all ranks. 
* Looking at Rank 16, moving from $\omega=12$ to $\omega=25$ increases the Forget PPL sequentially from 25.11 to 41.22. 
* This scaling trend is visible even at lower capacities; at Rank 8, Forget PPL scales cleanly from 15.02 ($\omega=12$) to 19.44 ($\omega=25$). 
This proves that a high-frequency non-linear gradient landscape acts as an effective mechanism for scrambling factual recall, provided it is stabilized by zero-initialization and tight gradient clipping (0.3).

### 3. Preservation of Retain Set Stability
Typically, pushing a model to aggressively forget data causes severe degradation in its general language capabilities. However, the hybrid approach actively improves retain performance compared to both the baseline and the Attention-only method. 
* The Baseline Retain PPL is 36.94. 
* At Rank 16, standard Attention-only LoRA drops this slightly to 35.88. 
* The Hybrid approach at Rank 16 ($\omega=25$) pushes the Retain PPL even lower to **35.22**.
By compartmentalizing the disruption—using standard linear LoRA to maintain stable contextual routing in the attention layers, while isolating the chaotic sine perturbations strictly to the MLP—the model successfully erases specific sequences while maintaining superior general language fluency.
