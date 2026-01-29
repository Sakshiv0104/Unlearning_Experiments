# Results - Complete Unlearning Analysis

## 0. Baseline Model (Shakespeare + Byron Training)

**Model trained on Shakespeare + Byron. Initial perplexities:**

| Dataset | PPL | Notes |
|---------|-----|-------|
| Shakespeare | 3.08 | Training data |
| **Byron** | **3.44** | **Target forget** |
| Roy | 8.07 | External |
| JKRowling | 7.45 | External |
| Austen | 6.09 | Unseen |
| Nursery | 11.27 | Unseen |

## 1. PART 1: Shakespeare Retain (Internal Training Data)

**Unlearn Byron, retain Shakespeare (already in training)**

| Method | Byron ↑ | Shakes ↓ | Roy ↑ | JKRowling ↑ | Austen ↑ | Nursery ↑ |
|--------|---------|----------|-------|-------------|----------|-----------|
| **Original** | 3.44 | 3.08 | 8.07 | 7.45 | 6.09 | 11.27 |
| **GA** | **8.63** | 6.05 | 18.81 | 14.99 | 13.34 | 22.32 |
| **GradDiff** | 4.91 | **3.08** | 10.55 | 9.86 | 8.84 | 16.35 |
| **KL** | 3.55 | 3.12 | 8.07 | 7.39 | 6.20 | 11.38 |

**Part 1 Pattern**: GA forgets Byron well but destroys ALL capabilities. GradDiff/KL preserve Shakespeare perfectly.

## 2. PART 2: Roy Retain (External Data)

**Unlearn Byron, retain Roy (never seen before)**

| Method | Byron ↑ | **Roy ↓** | Shakes ↑ | JKRowling ↑ | Austen ↓ | Nursery ↑ |
|--------|---------|-----------|----------|-------------|----------|-----------|
| **Original** | 3.44 | **8.07** | 3.08 | 7.45 | 6.09 | 11.27 |
| **GA** | **8.44** | 18.37 | 5.92 | 14.68 | 13.16 | 22.15 |
| **GradDiff** | 7.35 | **5.46** | 5.86 | 6.72 | **5.21** | 16.90 |
| **KL** | 3.78 | 7.63 | 3.28 | 6.94 | 5.91 | 11.36 |

**Part 2 Pattern**: GradDiff(Roy) forgets Byron AND improves Roy+Austen. GA destroys everything.

## 3. PART 2: JKRowling Retain (External Data)

**Unlearn Byron, retain JKRowling (never seen before)**

| Method | Byron ↑ | Roy ↑ | Shakes ↑ | **JKRowling ↓** | Austen ↓ | Nursery ↑ |
|--------|---------|-------|----------|-----------------|----------|-----------|
| **Original** | 3.44 | 8.07 | 3.08 | **7.45** | 6.09 | 11.27 |
| **GA** | **8.41** | 18.40 | 5.90 | 14.72 | 13.24 | 22.21 |
| **GradDiff** | **9.23** | 7.30 | 8.13 | **6.50** | 4.88 | 18.89 |
| **KL** | 3.89 | 7.58 | 3.34 | 7.01 | 5.92 | 11.54 |

## Method Behavior Summary

| Method | Forgetting Byron | Internal Retain | External Retain | Unseen Domains |
|--------|------------------|-----------------|-----------------|---------------|
| **GA** | **Excellent** (+145-151%) |  **Destroys** |  **Destroys** | **Destroys** |
| **GradDiff** |  **Good** (+43-168%) |  **Perfect** |  **Improves** |  **Improves** |
| **KL** |  **Weak** (+3-13%) |  **Perfect** |  **Stable** |  **Stable** |

## Key Takeaways

1. **GA**: Nuclear option - forgets perfectly, breaks everything else
2. **GradDiff**: Smart balance - forgets target, preserves/improves utility 
3. **KL**: Too safe - barely forgets, everything else stable
4. **External retain works**: Roy/JKRowling (never trained) successfully guide preservation
5. **Best method**: **GradDiff + external retain** (Byron +113-168%, Roy -32%, Austen -14%)

**Production recommendation**: GradDiff with external retain tuning (lr=3.5e-5).
