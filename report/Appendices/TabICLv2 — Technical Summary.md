# TabICLv2 — Technical Summary

Paper: **TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model**

## 1. Core Idea

TabICLv2 is a **tabular foundation model based on in-context learning (ICL)**.

At inference time it receives:

\[
(X_{\text{train}}, y_{\text{train}}, X_{\text{test}})
\]

and predicts:

\[
\hat y_{\text{test}}
\]

in a **single forward pass**, without gradient-based training or fine-tuning on the target dataset.

It is pretrained on approximately **35 million procedurally generated synthetic tabular tasks**, allowing it to learn how to infer relationships between variables from examples provided in context.

---

# 2. Architecture

Overall pipeline:

```text
X
│
├─ Repeated feature grouping
│
├─ Target-aware embeddings
│
▼
Column Transformer (TF_col)
│
▼
Row Transformer (TF_row)
│
▼
512-D row embeddings
│
▼
Dataset / ICL Transformer (TF_ICL)
│
▼
Prediction MLP
│
├─ Classification → logits
└─ Regression → 999 quantiles
```

## Feature Grouping

Features are grouped using circular offsets:

\[
(j,\;j+1,\;j+3)\pmod m
\]

before being projected into embeddings.

Purpose:

- introduce local feature interactions;
- prevent similar marginal feature distributions from collapsing into similar representations.

---

## Target-Aware Embeddings

For labelled training rows, the target is embedded directly into feature representations:

\[
E[i,j]
=
E_X[i,j] + E_y[y_i]
\]

Thus feature encoding can depend on the observed relationship between \(X\) and \(y\).

Test rows do not receive target embeddings.

---

# 3. Column Transformer — `TF_col`

Processes each feature across observations.

Uses a **Set Transformer** with inducing vectors rather than full row-to-row attention.

Configuration:

```text
Layers: 3
Embedding dimension: 128
Heads: 8
Inducing vectors: 128
Attention: QASSMax
```

Complexity is approximately:

\[
O(nk)
\]

instead of ordinary:

\[
O(n^2)
\]

for column-level attention.

---

# 4. Row Transformer — `TF_row`

Combines feature representations into a single representation for each row.

Configuration:

```text
Layers: 3
Embedding dimension: 128
Heads: 8
CLS tokens: 4
Positional encoding: RoPE
```

Four CLS tokens are concatenated:

\[
4\times128 = 512
\]

producing a **512-dimensional embedding per observation**.

---

# 5. Dataset Transformer — `TF_ICL`

Performs the actual dataset-level in-context learning.

Inputs are row embeddings from:

```text
training rows + y_train
test rows
```

Configuration:

```text
Layers: 12
Embedding dimension: 512
Heads: 8
Attention: QASSMax
```

The Transformer learns relationships between labelled training observations and unlabelled test observations.

Prediction head:

```text
2-layer MLP
Hidden dimension: 1024
```

---

# 6. QASSMax

TabICLv2 introduces **Query-Aware Scalable Softmax (QASSMax)**.

Standard attention can suffer from attention dilution as dataset size \(n\) becomes large.

QASSMax adjusts query magnitude using:

1. dataset size, approximately through \(\log n\);
2. a learned query-dependent gating function.

Conceptually:

\[
\tilde q
=
q
\times f(\log n)
\times g(q)
\]

This allows attention to become sharper for larger datasets while remaining query-dependent.

QASSMax is used in:

```text
TF_col
TF_ICL
```

and is important for generalizing to datasets much larger than those encountered during pretraining.

---

# 7. Loss Functions

Classification and regression are trained as separate models.

## Classification

Uses standard categorical cross-entropy:

\[
\mathcal L_{\text{class}}
=
-\sum_i
\log p_\theta
(y_i\mid X_{\text{train}},y_{\text{train}},x_i)
\]

The native classifier predicts up to **10 classes**.

Tasks with more classes are handled through hierarchical / mixed-radix decomposition.

---

## Regression

TabICLv2 performs **distributional regression**.

The model predicts **999 quantiles**:

\[
\alpha\in
\{0.001,0.002,\ldots,0.999\}
\]

Each quantile uses pinball loss:

\[
\rho_\alpha(u)
=
u(\alpha-\mathbf1[u<0])
\]

where

\[
u=y-\hat q_\alpha
\]

Total loss:

\[
\boxed{
\mathcal L_{\text{reg}}
=
\sum_{\alpha=0.001}^{0.999}
\rho_\alpha(y-\hat q_\alpha)
}
\]

Therefore TabICLv2 does **not directly optimize**:

```text
MSE
RMSE
R²
MAE
MASE
```

Instead, it learns an approximation of the full conditional distribution:

\[
p(Y\mid X,D_{\text{train}})
\]

A point prediction can be obtained by aggregating the predicted quantiles.

---

# 8. Pretraining Data

Pretraining uses **entirely synthetic tabular datasets**.

Real benchmark datasets such as TabArena/TALENT are used for evaluation, not for foundation-model pretraining.

Each synthetic task corresponds to a randomly generated data-generating process.

Approximate training exposure:

\[
\sim35\text{ million synthetic datasets/tasks}
\]

not 35 million individual rows.

---

# 9. Synthetic Dataset Generator

Synthetic tasks are based largely on randomly generated **structural causal models (SCMs)** / directed acyclic graphs.

For each task, the generator samples properties including:

```text
number of observations
number of features
numerical/categorical feature ratio
categorical cardinalities
number of classes
causal/dependency graph structure
feature importance
noise / relationship strength
```

Typical feature count during pretraining:

\[
m\in[2,100]
\]

Classification tasks generally contain:

\[
2-10\text{ classes}
\]

---

# 10. Random Function Families

Relationships between variables are sampled from multiple function families, including:

```text
1. Multilayer perceptrons
2. Tree-ensemble / CatBoost-like functions
3. Discretization / nearest-neighbour functions
4. Gaussian processes
5. Linear functions
6. Quadratic functions
7. Cluster / plateau / EM-like functions
8. Products and compositions of random functions
```

This exposes the model to many possible tabular relationships rather than assuming one universal functional form.

Synthetic datasets may therefore resemble:

\[
y=X\beta
\]

or

\[
y=X^\top AX
\]

or

```text
decision trees
Gaussian processes
clusters
highly nonlinear interactions
mixed categorical/numerical relationships
```

---

# 11. Synthetic Task Filtering

Generated datasets that contain little or no useful predictive structure are filtered.

An **ExtraTrees** model is used as a basic sanity check.

Synthetic problems can be rejected if a simple model cannot significantly outperform a constant predictor.

Approximately:

```text
~35% of generated classification tasks filtered
~25% of generated regression tasks filtered
```

during the first training stage.

---

# 12. Pretraining Curriculum

Training uses progressively larger contexts.

| Stage | Steps | Dataset size |
|---|---:|---:|
| 1 | 500,000 | 1,024 rows |
| 2 | 40,000 | 400–10,240 rows |
| 3 | 10,000 | 400–60,000 rows |

Maximum feature count:

```text
100
```

Stage 1 uses varying train/test proportions; later stages use approximately an 80% training-context split.

---

# 13. Optimisation

Training uses:

```text
Optimizer: Muon
Batch size: 64
LR schedule: cosine decay
Gradient clipping: up to 10 in early stages
```

Approximate compute:

```text
~24.5 H100 GPU-days per classification/regression model
```

---

# 14. Computational Scaling

TabPFN-style cell-level architectures can have complexity approximately:

\[
O(n^2m + nm^2)
\]

TabICLv2 compresses each observation into a 512-D representation before dataset-level attention, reducing this approximately to:

\[
\boxed{
O(n^2 + nm^2)
}
\]

This removes the number-of-features factor \(m\) from the expensive dataset-level \(n^2\) attention term.

---

# 15. Main Improvements over TabICL v1

Key TabICLv2 changes:

```text
1. Richer synthetic pretraining prior
2. Repeated feature grouping
3. Target-aware feature embeddings
4. QASSMax attention
5. Muon-based optimisation
6. Longer-context training curriculum
7. 999-quantile regression
```

---

# 16. Conceptual Interpretation

TabICLv2 is not primarily learning one function:

\[
X\rightarrow y
\]

Instead, during pretraining it repeatedly learns tasks of the form:

```text
Given:
    labelled examples (X_train, y_train)
    unseen observations X_test

Infer:
    the underlying relationship between X and y

Then:
    predict y_test
```

The meta-learning objective is therefore approximately:

\[
\boxed{
D_{\text{train}}
\rightarrow
\text{infer task-specific }f(X)
\rightarrow
\hat y_{\text{test}}
}
\]

The pretrained Transformer acts as a learned tabular learning algorithm.

---

# 17. Important Regression Implication

For regression experiments, remember:

```text
TabICLv2 is NOT trained using MSE.
```

It predicts an entire conditional target distribution through 999 quantiles and minimizes pinball loss.

Therefore evaluation metrics such as:

```text
R²
RMSE
MAE
MASE
```

are downstream evaluation criteria rather than the model's native training objective.

This can partly explain differences between TabICLv2 and conventional regressors such as XGBoost/LightGBM trained directly using squared-error objectives.