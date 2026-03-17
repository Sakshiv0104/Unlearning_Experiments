# Machine Unlearning via LoRA and QLoRA on GPT-Neo 125M

> An empirical study comparing parameter-efficient unlearning methods across adapter ranks and quantization precisions on a causal language model.

---

## Why Am I Doing This?

Large language models trained on web-scale corpora have been shown to **memorize verbatim sequences** from their training data. This is not just a theoretical concern — models can be prompted to reproduce private text, copyrighted content, or sensitive personal information word for word. As LLMs get deployed in real-world systems, this creates genuine legal and ethical exposure.

The intuitive fix is to retrain the model from scratch after removing the offending data. But for models with hundreds of millions to billions of parameters, full retraining is prohibitively expensive. This is the problem **machine unlearning** is designed to solve.

---

## What is Machine Unlearning?

Machine unlearning is the task of making a trained model **forget specific data** — reducing its ability to reproduce or leverage that data — without retraining from scratch and without significantly degrading performance on everything else it has learned.

Formally, given a model $\theta$ trained on dataset $\mathcal{D} = \mathcal{D}_f \cup \mathcal{D}_r$, the goal is to produce a model $\theta^*$ such that:

- $\theta^*$ behaves **as if it was never trained on** $\mathcal{D}_f$ (the **forget set**)
- $\theta^*$ retains its performance on $\mathcal{D}_r$ (the **retain set**)

This is evaluated through **perplexity (PPL)**:

$$\text{PPL}(\mathcal{D}_f) \uparrow \qquad \text{and} \qquad \text{PPL}(\mathcal{D}_r) \downarrow$$

A higher forget PPL means the model is no longer fluent on the forgotten sequences. A lower retain PPL means it has preserved its general language ability.

Perplexity is computed using a strided sliding window:

$$\text{PPL} = \exp\!\left(\frac{\sum_i \text{NLL}_i}{\sum_i T_i}\right)$$

where $\text{NLL}_i$ is the negative log-likelihood and $T_i$ is the number of predicted tokens per window. Window size is fixed at 512 tokens with a stride of 256.

---

## Unlearning Using PEFT

Traditional unlearning fine-tunes the entire model, which is expensive and risks **catastrophic forgetting** — the model may lose general ability while trying to forget a small subset of data. **Parameter-Efficient Fine-Tuning (PEFT)** offers a more controlled alternative.

PEFT methods freeze the base model's weights entirely and introduce a small number of trainable parameters on top. The unlearning signal is applied **only through these lightweight parameters**, which means:

- The base model's knowledge is largely protected by construction
- The trainable parameter count is a small fraction of total parameters
- The adapter can be merged back into the base model after training with no inference overhead

This makes PEFT a natural fit for unlearning — the adapter acts as a targeted surgical instrument rather than a blunt full-model update.

---

## LoRA and QLoRA

### LoRA — Low-Rank Adaptation

LoRA (Hu et al., 2022) injects trainable low-rank matrices alongside the frozen weights of selected linear layers. For a pretrained weight $W_0 \in \mathbb{R}^{d \times k}$, the adapted output is:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} \cdot B A x$$

where:
- $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ are the trainable adapter matrices
- $r$ is the **rank** — a small integer that controls the adapter's capacity
- $\alpha$ is a scaling factor (set to $2r$ throughout this work)
- $W_0$ is completely **frozen**; only $A$ and $B$ receive gradient updates

The rank $r$ is the central hyperparameter. A higher rank gives the adapter more expressive capacity to induce forgetting, but also risks greater collateral damage to the retain set. In this work, adapters are inserted into the **query projection (`q_proj`) and value projection (`v_proj`)** of every attention layer in GPT-Neo 125M.

### QLoRA — Quantized LoRA

QLoRA (Dettmers et al., 2023) extends LoRA by loading the base model in **quantized precision** (4-bit or 8-bit) while keeping the adapter weights in full precision (FP32). This dramatically reduces GPU memory. The core idea is that quantization affects only the frozen base model — the trainable adapter matrices remain in full precision and receive gradients normally.

The 4-bit variant uses **NF4 (NormalFloat4)** quantization with double quantization for additional memory savings, and FP16 as the compute dtype for numerical stability during the forward pass.

### Why Compare LoRA vs QLoRA?

LoRA and QLoRA are architecturally identical in their adapter structure, but differ in one critical way: **the precision of the frozen base weights**. When the unlearning gradient flows through the adapter, it implicitly depends on how accurately the frozen base represents the original model. Quantization introduces approximation error into those frozen weights, which may attenuate or distort the unlearning signal.

Comparing them across multiple bit-widths answers the question: **does quantization hurt unlearning, and if so, how much and under which conditions?** This is practically important because QLoRA is the de facto standard for memory-efficient fine-tuning, and understanding whether it remains viable for unlearning has direct implications for privacy-preserving deployment.

---

## Unlearning Objectives

Three gradient-based unlearning objectives are implemented and compared. All three operate on the same forget set $\mathcal{D}_f$ and retain set $\mathcal{D}_r$, but differ in how they protect general model performance.

The cross-entropy loss used throughout is:

$$\mathcal{L}_{\text{CE}}(\theta;\, \mathcal{D}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} m_t^{(i)} \cdot \log p_\theta\!\left(x_t^{(i)} \mid x_{<t}^{(i)}\right)$$

where $m_t \in \{0,1\}$ is the attention mask that excludes padding tokens from the loss.

---

### Gradient Ascent (GA)

The simplest approach: **reverse the training signal** on the forget set. Instead of minimizing CE loss, we maximize it — pushing the model to assign lower probability to memorized sequences.

$$\mathcal{L}_{\text{GA}} = -\mathcal{L}_{\text{CE}}(\theta;\, \mathcal{D}_f)$$

GA has no explicit mechanism to protect the retain set. The only implicit protection comes from the low-rank adapter — since $W_0$ is frozen, the adapter has limited capacity to cause widespread damage. Still, at high ranks, GA can destabilize the model significantly.

---

### Gradient Difference (GD)

GD augments GA with an explicit retain regularization term. At each training step, a batch from the forget set and a batch from the retain set are both sampled:

$$\mathcal{L}_{\text{GD}} = -\mathcal{L}_{\text{CE}}(\theta;\, \mathcal{D}_f) + \lambda \cdot \mathcal{L}_{\text{CE}}(\theta;\, \mathcal{D}_r)$$

The retain term pulls the model toward better performance on $\mathcal{D}_r$ simultaneously with forgetting $\mathcal{D}_f$. The hyperparameter $\lambda$ controls the balance — a higher $\lambda$ prioritizes retain quality, a lower $\lambda$ prioritizes forgetting strength.

---

### KL Divergence (KL)

KL unlearning uses the **original base model as a reference** $\theta_{\text{ref}}$ and adds a penalty for distributional drift on the retain set. Rather than just minimizing CE loss on the retain set, the model is penalized for deviating from the base model's token-level output distribution:

$$\mathcal{L}_{\text{KL}} = -\mathcal{L}_{\text{CE}}(\theta;\, \mathcal{D}_f) + \beta \cdot D_{\text{KL}}\!\left(p_{\theta_{\text{ref}}} \;\|\; p_\theta\right)\bigg|_{\mathcal{D}_r}$$

The token-level KL divergence is masked over non-padding positions:

$$D_{\text{KL}}\big|_{\mathcal{D}_r} = \frac{\displaystyle\sum_{i,t} m_t^{(i)} \sum_v p_{\theta_{\text{ref}},t}^{(i,v)} \log \frac{p_{\theta_{\text{ref}},t}^{(i,v)}}{p_{\theta,t}^{(i,v)}}}{\displaystyle\sum_{i,t} m_t^{(i)}}$$

The reference model $\theta_{\text{ref}}$ is the frozen base (no LoRA) and is never updated. $\beta$ controls how tightly the unlearned model must track the base model on retain data. KL provides the strongest theoretical guarantee of retain preservation among the three methods.

---

## Parameters & Hyperparameters

### Model Configuration

| Parameter | Value |
|---|---|
| Base model | GPT-Neo 125M (`EleutherAI/gpt-neo-125m`) |
| Adapter target modules | `q_proj`, `v_proj` |
| LoRA $\alpha$ | $2 \times r$ |
| LoRA dropout | 0.05 |
| Optimizer | AdamW |
| Random seed | 42 |

### Dataset

| Parameter | Value |
|---|---|
| Forget set size | 200 sequences × 200 tokens |
| Forget source | ETH Zürich LM Extraction Benchmark (prefix + suffix pairs) |
| Retain corpus | 50,000 tokens (fixed slice) |
| Train sequence length | 200 tokens |
| Batch size | 4 |
| PPL window size | 512 tokens |
| PPL stride | 256 tokens |

### Experiment Grids

| Setting | Ranks Tested | Bits Tested |
|---|---|---|
| LoRA | 2, 4, 8, 16, 32 | — (full precision) |
| QLoRA | 2, 4, 8, 16, 32 | 4-bit (NF4), 8-bit (INT8), 16-bit (FP16), 32-bit (FP32) |

### Unlearning Method Hyperparameters

| Method | Learning Rate | Steps | Key Hyperparameter | Gradient Clip |
|---|---|---|---|---|
| GA | `1e-5` | 300 | — | 1.0 |
| GD | `5e-5` | 150 | $\lambda = 5.0$ (retain weight) | 2.0 |
| KL | `1e-5` | 400 | $\beta = 5.0$ (KL penalty) | 1.0 |

---

## Results

### Baseline PPL (No Unlearning)

| | Forget PPL | Retain PPL |
|---|---|---|
| GPT-Neo 125M (base) | ~11.97 | ~36.94 |

---

### LoRA Results

| Rank | GA Forget ↑ | GA Retain ↓ | GD Forget ↑ | GD Retain ↓ | KL Forget ↑ | KL Retain ↓ |
|---|---|---|---|---|---|---|
| 2  | 11.97 | 38.04 | 12.06 | 36.95 | 12.12 | 38.05 |
| 4  | 12.31 | 38.07 | 12.77 | 36.45 | 12.46 | 38.09 |
| 8  | 12.90 | 38.26 | 14.44 | 35.96 | 13.78 | 38.18 |
| 16 | 16.42 | 41.25 | 26.53 | 35.82 | 17.18 | 38.40 |
| 32 | 33.24 | 53.24 | 83.79 | 36.21 | 33.00 | 38.80 |

---

### QLoRA Results — Forget PPL ↑

#### GA
| Rank | 4-bit | 8-bit | 16-bit | 32-bit |
|---|---|---|---|---|
| 2  | 21.51 | 12.96 | 12.01 | 11.98 |
| 4  | 21.52 | 12.97 | 12.34 | 12.24 |
| 8  | 21.53 | 12.98 | 13.35 | 13.08 |
| 16 | 21.55 | 13.00 | 16.58 | 16.11 |
| 32 | 21.57 | 13.42 | 29.97 | 35.33 |

#### GD
| Rank | 4-bit | 8-bit | 16-bit | 32-bit |
|---|---|---|---|---|
| 2  | 21.51 | 13.01 | 12.11 | 12.24 |
| 4  | 21.51 | 12.93 | 12.76 | 12.74 |
| 8  | 21.51 | 13.01 | 14.34 | 15.94 |
| 16 | 21.53 | 15.90 | 24.95 | 25.82 |
| 32 | 21.54 | 38.31 | 79.92 | 93.25 |

#### KL
| Rank | 4-bit | 8-bit | 16-bit | 32-bit |
|---|---|---|---|---|
| 2  | 21.51 | 13.00 | 12.16 | 12.15 |
| 4  | 21.51 | 12.97 | 12.62 | 12.53 |
| 8  | 21.51 | 12.94 | 13.74 | 13.72 |
| 16 | 21.51 | 13.03 | 17.16 | 17.07 |
| 32 | 21.52 | 15.73 | 26.02 | 32.25 |

### QLoRA Results — Retain PPL ↓

#### GA
| Rank | 4-bit | 8-bit | 16-bit | 32-bit |
|---|---|---|---|---|
| 2  | 55.49 | 40.41 | 38.05 | 38.04 |
| 4  | 55.51 | 40.49 | 38.11 | 38.06 |
| 8  | 55.52 | 40.45 | 38.53 | 38.32 |
| 16 | 55.53 | 40.45 | 41.66 | 41.22 |
| 32 | 55.55 | 40.57 | 51.42 | 54.92 |

#### GD
| Rank | 4-bit | 8-bit | 16-bit | 32-bit |
|---|---|---|---|---|
| 2  | 55.48 | 40.42 | 36.98 | 36.94 |
| 4  | 55.45 | 40.44 | 36.46 | 36.43 |
| 8  | 55.45 | 40.45 | 35.99 | 36.04 |
| 16 | 55.40 | 39.16 | 35.92 | 35.82 |
| 32 | 55.30 | 38.18 | 36.40 | 36.36 |

#### KL
| Rank | 4-bit | 8-bit | 16-bit | 32-bit |
|---|---|---|---|---|
| 2  | 55.49 | 40.43 | 38.03 | 38.05 |
| 4  | 55.49 | 40.48 | 38.06 | 38.08 |
| 8  | 55.48 | 40.50 | 38.15 | 38.16 |
| 16 | 55.46 | 40.45 | 38.38 | 38.39 |
| 32 | 55.44 | 40.65 | 38.85 | 38.80 |

---

## Discussion

### LoRA vs QLoRA — General Observations

The most striking pattern across all results is the **sharp collapse in unlearning effectiveness at 4-bit quantization**. Across every method and every rank, 4-bit QLoRA produces forget PPL values hovering around 21.5 — barely above the base model baseline of ~11.97 — while simultaneously inflating retain PPL to ~55. This means 4-bit QLoRA neither forgets the target sequences nor preserves general fluency. It fails on both criteria simultaneously.

The likely cause is that NF4 quantization introduces sufficient approximation error into the frozen base weights that the gradient signal flowing back through the adapter becomes noisy and incoherent. The adapter cannot compensate for the distorted forward pass, and the unlearning objective fails to take hold in any direction.

At **8-bit**, results recover substantially and align more closely with full-precision LoRA — though a consistent gap remains, especially at higher ranks. This suggests 8-bit quantization preserves enough base weight precision for the unlearning gradient to be informative. The retain PPL at 8-bit (~40) is still worse than full LoRA (~36–38) across all methods, indicating a residual penalty from quantization noise even at this precision.

At **16-bit and 32-bit**, QLoRA results closely mirror standard LoRA. The small residual differences are attributable to the different model loading path (`device_map="auto"` vs explicit `.to(device)`) rather than quantization itself. This confirms that at these precisions, quantization is not a meaningful variable.

**Practical implication:** if memory constraints require quantization, 8-bit is the minimum viable precision for PEFT-based unlearning. 4-bit quantization actively breaks the unlearning pipeline and should be avoided for this use case.

---

### LoRA vs QLoRA — Method-by-Method Analysis

#### Gradient Ascent (GA)

In standard LoRA, GA shows clean rank-scaling: forget PPL climbs steadily from 11.97 at rank 2 to 33.24 at rank 32, with retain PPL rising in tandem (38.04 → 53.24). The absence of retain regularization makes GA increasingly destructive at higher ranks — it forgets aggressively but without control.

In QLoRA, the rank-scaling behavior is almost entirely suppressed at 4-bit. Forget PPL barely moves (11.98 → 21.57) regardless of rank, and retain PPL is uniformly elevated at ~55. At 8-bit, weak rank-scaling reappears; at 16/32-bit it closely mirrors standard LoRA. Notably, QLoRA's retain degradation at high ranks is comparable to LoRA (54.92 vs 53.24 at rank 32, 32-bit), suggesting quantization neither consistently helps nor hurts retain stability for GA — it primarily attenuates the forget signal.

GA is the most sensitive method to quantization because it has no retain anchor. Without regularization, even small distortions in the gradient from quantization noise are enough to prevent meaningful forgetting while still causing retain degradation.

#### Gradient Difference (GD)

GD is the most aggressive unlearner in standard LoRA. At rank 32, forget PPL reaches 83.79 — far the highest across all methods — while retain PPL stays controlled at 36.21, demonstrating that the explicit retain loss successfully anchors general ability even as forgetting is pushed hard.

In QLoRA, GD shows the most dramatic quantization sensitivity on the forget side: at 4-bit, forget PPL is completely flat (~21.51–21.54), and even at 8-bit the rank-32 forgetting only reaches 38.31 vs 83.79 in LoRA. However, at 32-bit QLoRA, GD actually **exceeds** LoRA's forgetting strength (93.25 vs 83.79). This anomaly likely arises from differences in how FP32 tensors interact with the retain regularization when loaded via `device_map="auto"` vs standard loading — a subtle but real implementation difference.

GD's key strength under quantization is **retain stability**. Because the retain loss explicitly counteracts quantization-induced degradation, GD's retain PPL at 8-bit (~38–40) is better than GA (~40–41) and comparable to KL. At 16/32-bit, GD actually produces *lower* retain PPL than the base model at higher ranks (e.g., 35.82 at rank 16), suggesting the retain regularization is actively improving general fluency beyond baseline.

#### KL Divergence (KL)

KL is the most conservative method across both settings. In standard LoRA, forget PPL reaches 33.00 at rank 32 with retain PPL of 38.80 — the tightest retain control of any method, because the KL penalty explicitly anchors the model to the base model's full output distribution on retain data.

In QLoRA, KL behaves similarly to GA in quantization sensitivity — 4-bit nearly completely suppresses forgetting, and 8-bit recovers most of the signal. However, KL's retain PPL at 8-bit (~40.43–40.65) is slightly worse than GD (~38–40) despite KL's theoretically stronger retain guarantee. The likely cause is a **precision mismatch**: the KL penalty is computed between the full-precision reference model and the quantized unlearning model. When the two models differ in precision, the KL term measures something different from what it would in a matched-precision setting, slightly reducing its effectiveness as a retain anchor.

At 16-bit and 32-bit, this mismatch disappears and KL QLoRA closely matches standard LoRA — steady, controlled forgetting with minimal retain degradation. KL remains the safest choice when stability and predictability matter more than maximizing forgetting strength.

---

## References

- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- Carlini et al. (2023). *Quantifying Memorization Across Neural Language Models.* ICLR 2023.
- Yao et al. (2023). *Large Language Model Unlearning.* arXiv:2310.10683.
- Maini et al. (2024). *TOFU: A Task of Fictitious Unlearning for LLMs.* arXiv:2401.06121.
- ETH Zürich LM Extraction Benchmark: https://github.com/ethz-spylab/lm-extraction-benchmark-data
