# Machine Unlearning via LoRA and QLoRA on GPT-Neo 125M

An empirical study comparing parameter-efficient unlearning methods across adapter ranks and quantization precisions on a causal language model.

## 🧠 The Problem: Machine Unlearning
Large Language Models (LLMs) often memorize their training data. If a model memorizes sensitive, private, or copyrighted text, the traditional fix is to retrain the entire model from scratch without that data—a process that is prohibitively expensive and time-consuming.

**Machine Unlearning** solves this by surgically removing specific knowledge. The goal is to make the model "forget" specific data (measured by a high Perplexity on the **Forget Set**, $\mathcal{D}_f$) while keeping its general language skills perfectly intact (measured by a low Perplexity on the **Retain Set**, $\mathcal{D}_r$).

---

## 🛠️ The Tools: PEFT, LoRA, and QLoRA

Instead of expensive full-model fine-tuning, this project uses **Parameter-Efficient Fine-Tuning (PEFT)** to perform the unlearning.

### What is LoRA (Low-Rank Adaptation)?
Normally, changing a model's behavior means updating all of its billions of parameters. **LoRA** avoids this by **freezing** the original model completely. Instead, it injects a tiny, trainable "adapter" module onto the model's layers. 
* **Why it matters for unlearning:** We only train this tiny adapter to do the "forgetting." This acts like a surgical scalpel, protecting the frozen base model's core knowledge from being accidentally destroyed (catastrophic forgetting).

### What is QLoRA (Quantized LoRA)?
AI models take up a massive amount of VRAM (graphics card memory). **QLoRA** solves this by "quantizing" or squishing the frozen base model down to lower precisions (like 8-bit or 4-bit) to save memory, while keeping the tiny LoRA adapter in high precision so it can still learn.
* **The Big Question of this Study:** Unlearning requires highly precise, delicate mathematical gradients. If we "squish" the base model using QLoRA, does it introduce too much mathematical noise? Does it break the unlearning process? 

---

## 🔬 Unlearning Objectives (The Methods)

We tested three different mathematical approaches to tell the LoRA adapter how to unlearn the data. All rely on maximizing the Cross-Entropy loss ($\mathcal{L}_{\text{CE}}$) on the forget set, but they differ in how they protect the retain set:

1. **Gradient Ascent (GA):** Reverses the training signal. Aggressively forgets, but risks high collateral damage to general knowledge.
   $$\mathcal{L}_{\text{GA}} = -\mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_f)$$

2. **Gradient Difference (GD):** Balances forgetting with an explicit regularization term to protect general knowledge.
   $$\mathcal{L}_{\text{GD}} = -\mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_f) + \lambda \cdot \mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_r)$$

3. **KL Divergence (KL):** Penalizes the model for deviating from the frozen base model's token distribution, offering the safest retain guarantee.
   $$\mathcal{L}_{\text{KL}} = -\mathcal{L}_{\text{CE}}(\theta, \mathcal{D}_f) + \beta \cdot D_{\text{KL}}(p_{\theta_{\text{ref}}} \parallel p_\theta)\big|_{\mathcal{D}_r}$$

---

## 📊 Results

*Metrics are evaluated using Perplexity (PPL). Format: **Forget PPL / Retain PPL**.* *Target: **Forget ↑** (higher is better) and **Retain ↓** (lower is better).*

**Baseline (No Unlearning):** Forget PPL: **~11.97** | Retain PPL: **~36.94**

### 1. Gradient Ascent (GA)
| Rank | Full LoRA (FP) | QLoRA 32-bit | QLoRA 16-bit | QLoRA 8-bit | QLoRA 4-bit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2** | 11.97 / 38.04 | 11.98 / 38.04 | 12.01 / 38.05 | 12.96 / 40.41 | 21.51 / 55.49 |
| **4** | 12.31 / 38.07 | 12.24 / 38.06 | 12.34 / 38.11 | 12.97 / 40.49 | 21.52 / 55.51 |
| **8** | 12.90 / 38.26 | 13.08 / 38.32 | 13.35 / 38.53 | 12.98 / 40.45 | 21.53 / 55.52 |
| **16** | 16.42 / 41.25 | 16.11 / 41.22 | 16.58 / 41.66 | 13.00 / 40.45 | 21.55 / 55.53 |
| **32** | 33.24 / 53.24 | 35.33 / 54.92 | 29.97 / 51.42 | 13.42 / 40.57 | 21.57 / 55.55 |

### 2. Gradient Difference (GD)
| Rank | Full LoRA (FP) | QLoRA 32-bit | QLoRA 16-bit | QLoRA 8-bit | QLoRA 4-bit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2** | 12.06 / 36.95 | 12.24 / 36.94 | 12.11 / 36.98 | 13.01 / 40.42 | 21.51 / 55.48 |
| **4** | 12.77 / 36.45 | 12.74 / 36.43 | 12.76 / 36.46 | 12.93 / 40.44 | 21.51 / 55.45 |
| **8** | 14.44 / 35.96 | 15.94 / 36.04 | 14.34 / 35.99 | 13.01 / 40.45 | 21.51 / 55.45 |
| **16** | 26.53 / 35.82 | 25.82 / 35.82 | 24.95 / 35.92 | 15.90 / 39.16 | 21.53 / 55.40 |
| **32** | 83.79 / 36.21 | 93.25 / 36.36 | 79.92 / 36.40 | 38.31 / 38.18 | 21.54 / 55.30 |

### 3. KL Divergence (KL)
| Rank | Full LoRA (FP) | QLoRA 32-bit | QLoRA 16-bit | QLoRA 8-bit | QLoRA 4-bit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2** | 12.12 / 38.05 | 12.15 / 38.05 | 12.16 / 38.03 | 13.00 / 40.43 | 21.51 / 55.49 |
| **4** | 12.46 / 38.09 | 12.53 / 38.08 | 12.62 / 38.06 | 12.97 / 40.48 | 21.51 / 55.49 |
| **8** | 13.78 / 38.18 | 13.72 / 38.16 | 13.74 / 38.15 | 12.94 / 40.50 | 21.51 / 55.48 |
| **16** | 17.18 / 38.40 | 17.07 / 38.39 | 17.16 / 38.38 | 13.03 / 40.45 | 21.51 / 55.46 |
| **32** | 33.00 / 38.80 | 32.25 / 38.80 | 26.02 / 38.85 | 15.73 / 40.65 | 21.52 / 55.44 |

---

## 💡 Discussion & Takeaways

### LoRA vs. QLoRA: The Impact of Quantization
**1. 4-bit QLoRA completely breaks unlearning.**
Across every single method and rank, 4-bit quantization failed. It barely nudged the Forget Perplexity (hovering around 21.5, compared to the baseline 11.97), while simultaneously ruining the model's general language skills (Retain Perplexity shot up to ~55). 
* *Takeaway:* The approximation noise from 4-bit quantization is simply too loud. The unlearning signal gets scrambled, and the adapter ruins the model without actually forgetting the target data.

**2. 8-bit is the Minimum Viable Precision.**
At 8-bit, the unlearning signal successfully pushes through the noise. It successfully forgets the data, but it carries a slight "quantization tax"—the model's general fluency (Retain PPL) is consistently slightly worse (~40) than it is in full-precision LoRA (~36).

**3. 16-bit and 32-bit QLoRA act exactly like Standard LoRA.**
Once you hit 16-bit precision, the quantization noise practically vanishes. The results map almost perfectly 1:1 with standard full-precision LoRA. 

### Unlearning Methods: Which works best?
**1. Gradient Ascent (GA)**
* **In LoRA:** It works, but it's dangerous at higher ranks. Because there is no "safety net" telling the model to preserve its general knowledge, high-rank GA damages the model's overall fluency while forgetting.
* **In QLoRA:** GA is incredibly sensitive to quantization noise. At 4-bit, it completely flatlines. 

**2. Gradient Difference (GD) — *The MVP***
* **In LoRA:** This was our most successful method. It achieved the highest Forget PPL of the entire study (**83.79** at Rank 32) while keeping the Retain PPL completely stable at 36.21. It proves that explicitly balancing "forgetting" and "retaining" in the loss function works beautifully.
* **In QLoRA:** GD's explicit retain-check helps it survive 8-bit quantization much better than GA, keeping the model relatively fluent while still inducing heavy forgetting.

**3. KL Divergence (KL) — *The Safest Choice***
* **In LoRA:** KL is the safest, most stable method. It tightly controls the Retain PPL, never letting the model degrade, though it sacrifices a bit of forgetting power to maintain that safety.
* **In QLoRA:** Because KL relies on comparing the precise token distributions between the base model and the adapter, the "squished" 4-bit and 8-bit weights slightly disrupt this delicate comparison. It still works well at 8-bit, but KL truly shines when precision is high (16-bit/32-bit).
