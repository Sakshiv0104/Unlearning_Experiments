# Data

This project uses literary text datasets from multiple authors for machine unlearning experiments. The training data consists of **Shakespeare** and **Byron**, where the model learns both writing styles. For unlearning experiments, **Byron is designated as the forget set** (to be removed from the model's memory), while external datasets serve as retain sets to preserve general language capabilities. The retain sets tested include **Roy (Arundhati Roy)** and **JK Rowling**, which were not part of the original training data. Additional datasets from **Jane Austen** and **Nursery Rhymes** are included as holdout test sets to monitor the model's generalization performance and ensure unlearning does not cause catastrophic forgetting of general language patterns.

## Dataset Summary

| Dataset | Role | Description |
|---------|------|-------------|
| Shakespeare | Training (70.5%) | Part of original training data |
| Byron | Training (29.5%) **(Forget Set)** | Target for unlearning |
| Roy | **Retain Set** (external) | Preserve general capability |
| JK Rowling | **Retain Set** (external) | Preserve general capability |
| Austen | Test Set | Monitor generalization |
| Nursery Rhymes | Test Set | Monitor generalization |

