# Plan: Add a soft-independence conditioning encoder for conditional flow matching

## Goal

Improve robustness of conditional flow matching for estimating \( p(\theta \mid x) \) when:

- \(x \in \mathbb{R}^D\) is high-dimensional,
- \(\theta\) is low-dimensional,
- training data may be limited,
- a fully-coupled conditioning network overfits or learns unstable dependencies across coordinates of \(x\).

The core idea is to replace a fully entangled conditioner on \(x\) with a **mostly additive encoder**:

\[
c(x) = \sum_{i=1}^D \phi(x_i) + \alpha \psi(x),
\]

and then use this context in the conditional velocity field

\[
v(\theta, t; x) = \tilde v(\theta, t, c(x)).
\]

Here:

- \(\phi(x_i)\) is a shared per-coordinate feature extractor,
- \(\sum_i \phi(x_i)\) encodes approximately independent evidence from each dimension of \(x\),
- \(\psi(x)\) is a residual interaction network,
- \(\alpha\) controls how strongly cross-coordinate interactions are allowed.

---

## Motivation

In this project, the flow is defined in **\(\theta\)-space**:

\[
\frac{d\theta}{dt} = v(\theta, t; x),
\]

and the direct ODE likelihood is computed from the divergence with respect to \(\theta\), not with respect to \(x\).

Therefore, if we want to impose an inductive bias that coordinates of \(x\) are approximately independent, the natural place to do so is in the **conditioning architecture** \(x \mapsto c(x)\), not in the Jacobian of the \(\theta\)-flow.

A mostly additive conditioner gives a soft Naive-Bayes-like bias:

- each coordinate \(x_i\) contributes independent evidence about \(\theta\),
- but a residual interaction term can still capture dependencies when needed.

---

## Proposed model

### 1. Context encoder

Use:

\[
c(x) = c_{\text{add}}(x) + \alpha c_{\text{int}}(x),
\]

with

\[
c_{\text{add}}(x) = \sum_{i=1}^D \phi(x_i),
\qquad
c_{\text{int}}(x) = \psi(x).
\]

Recommended implementation:

- \(\phi: \mathbb{R} \to \mathbb{R}^{d_c}\) is a small shared MLP,
- \(\psi: \mathbb{R}^D \to \mathbb{R}^{d_c}\) is a small residual MLP,
- \(d_c\) is the context dimension.

Optional normalization:

\[
c_{\text{add}}(x) = \frac{1}{D} \sum_{i=1}^D \phi(x_i),
\]

which may improve scale stability when \(D\) changes.

### 2. Conditional velocity

Use the encoded context in the flow-matching velocity model:

\[
v(\theta, t; x) = \tilde v(\theta, t, c(x)).
\]

Recommended implementation:

- concatenate \(\theta\), time embedding \(e(t)\), and context \(c(x)\),
- feed the concatenated vector into a small MLP,
- output the velocity in \(\theta\)-space.

---

## Main design choices

### A. Shared per-coordinate feature extractor

Use one shared network \(\phi\) for all coordinates:

\[
\phi(x_i) = \text{MLP}_{\phi}(x_i).
\]

Advantages:

- strong inductive bias,
- parameter efficient,
- reduces overfitting,
- naturally permutation-tolerant if coordinate order is not meaningful.

### B. Residual interaction branch

Use a small residual network:

\[
\psi(x) = \text{MLP}_{\psi}(x),
\]

and scale it by a small factor \(\alpha\).

Recommended initial choices:

- fixed small \(\alpha\), such as \(0.01\) or \(0.1\),
- or a learnable scalar initialized near zero,
- or a schedule that increases \(\alpha\) during training.

### C. Optional blockwise extension

If coordinates of \(x\) have known structure, replace coordinatewise independence with blockwise independence:

\[
c(x) = \sum_{b=1}^B \phi_b(x^{(b)}) + \alpha \psi(x),
\]

where each block \(x^{(b)}\) contains related coordinates.

This is useful when neighboring coordinates are expected to be correlated.

---

## Training objective

Keep the original flow-matching loss and add a soft penalty on the interaction branch.

### Base loss

\[
\mathcal{L}_{\text{FM}}
\]

This is the current conditional flow-matching training loss.

### Regularized loss

Start with the simplest version:

\[
\mathcal{L}
=
\mathcal{L}_{\text{FM}}
+
\lambda_{\text{int}} \, \| \psi(x) \|^2.
\]

This encourages the model to rely on additive evidence unless residual interactions are truly useful.

Possible alternatives:

1. Penalize the residual context magnitude:
   \[
   \lambda_{\text{int}} \|c_{\text{int}}(x)\|^2
   \]

2. Penalize the interaction weight directly:
   \[
   \lambda_\alpha \alpha^2
   \]

3. Use both.

### Optional stronger regularizer

A more principled but more expensive penalty is to discourage interaction sensitivity across coordinates:

\[
\lambda_{\text{cross}}
\sum_{i \neq j}
\left\|
\frac{\partial^2 v(\theta,t;x)}{\partial x_i \partial x_j}
\right\|^2.
\]

This should be considered only after the simpler residual penalty is tested.

---

## Implementation steps

### Step 1. Refactor conditioning into a separate module

Create a new context encoder module, for example:

- `AdditiveContextEncoder`
- `BlockAdditiveContextEncoder` for later extension

The module should expose:

```python
c = context_encoder(x)