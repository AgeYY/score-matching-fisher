# Dimension Increasing in Johnston & Fusi (2023)

This note focuses **only** on the part of the paper that turns a low-dimensional latent variable into a high-dimensional synthetic dataset.

## Core idea

They start with a low-dimensional latent variable vector:

- latent dimension: usually `D = 5`
- latent samples: typically drawn from a Gaussian distribution

Then they learn a neural network that maps these latent variables into a **high-dimensional, sparse, non-abstract representation**. That learned representation is what they use as the synthetic input dataset for the rest of the paper.

In other words, they do:

```text
low-dimensional latent z  -->  learned encoder  -->  high-dimensional code h
```

where:

- `z in R^D`
- `h in R^500` in their main setup

---

## The model they use

They call this module the **input model** or **standard input**.

It is a **feedforward autoencoder** with a bottleneck-like low-dimensional input and a larger hidden representation.

A practical implementation can be:

```text
Encoder: D -> 100 -> 200 -> 500
Decoder: 500 -> 200 -> 100 -> D
```

The 500-dimensional hidden layer is the representation they want.

After training:

- keep the encoder
- discard or ignore the decoder for downstream use
- use the 500-dimensional representation as the synthetic dataset

---

## What objective do they optimize?

They do **not** just train a normal autoencoder.

They train the model with **two goals at the same time**:

1. **Reconstruction objective**
   - the decoder should reconstruct the original latent variable from the high-dimensional representation
   - this keeps the representation informative

2. **Dimensionality expansion objective**
   - the representation layer should have large embedding dimensionality
   - they measure this using the **participation ratio**
   - during training they encourage this quantity to be large

So the total loss is conceptually:

```text
loss = reconstruction_loss - lambda_dim * participation_ratio(representation)
```

or equivalently

```text
loss = reconstruction_loss + lambda_dim * dimensionality_penalty
```

where the dimensionality penalty is defined so that minimizing the loss increases participation ratio.

The exact sign convention is not important. The important point is:

- reconstruct the latent variable well
- push the representation to spread across many dimensions

---

## Why does this increase dimension?

Without the extra dimensionality term, the network could simply learn a compact low-dimensional code that stays close to the original latent structure.

But they explicitly want something more like an early sensory representation:

- high-dimensional
- sparse
- tangled
- non-abstract

So they encourage the representation to use many directions in activity space while still preserving enough information to reconstruct the original latent variable.

This gives a representation that:

- contains the latent information
- is much more nonlinear and distributed
- is harder to linearly decode in a way that generalizes cleanly

That is exactly the kind of “messy” input they need before showing that multitask learning can recover abstraction.

---

## What is participation ratio?

Participation ratio is their measure of embedding dimensionality.

If `C` is the covariance matrix of representation activity across samples, then a standard form is:

```text
PR(C) = (trace(C)^2) / trace(C^2)
```

Interpretation:

- if activity is concentrated in only a few dimensions, PR is small
- if activity is spread across many dimensions, PR is large

So maximizing PR encourages the representation to occupy a larger effective dimensional subspace.

---

## What does the resulting representation look like?

According to the paper, the learned high-dimensional code has these properties:

- sparse
- conjunctive
- often multi-modal at the single-unit level
- tangled and disordered at the population level
- only about 4% of units active for a given stimulus
- effective dimensionality close to about 190, even though the latent space is only 5-dimensional

So although the representation layer has 500 units, the **effective** dimensionality is around 190 in their main experiment.

---

## Minimal implementation plan in Python

## Step 1: sample latent variables

Generate latent samples:

```python
import numpy as np

N = 50000
D = 5
z = np.random.randn(N, D).astype(np.float32)
```

---

## Step 2: build the autoencoder

In PyTorch, define:

- encoder: `D -> 100 -> 200 -> 500`
- decoder: `500 -> 200 -> 100 -> D`

Use ReLU or another simple nonlinearity.

---

## Step 3: compute the representation

For a batch `x`, let:

```python
h = encoder(x)   # shape: [batch, 500]
x_hat = decoder(h)
```

---

## Step 4: reconstruction loss

Use mean squared error:

```python
recon_loss = ((x_hat - x) ** 2).mean()
```

---

## Step 5: participation-ratio reward

Center the batch representation:

```python
h_centered = h - h.mean(dim=0, keepdim=True)
C = h_centered.T @ h_centered / (h_centered.shape[0] - 1)
```

Then:

```python
trC = torch.trace(C)
trC2 = torch.trace(C @ C)
pr = (trC ** 2) / (trC2 + 1e-8)
```

Use it in the loss, for example:

```python
loss = recon_loss - lambda_dim * pr
```

In practice, you may want to normalize `pr` or tune `lambda_dim`, because the scale of this term can be large.

---

## Step 6: train the autoencoder

Train until:

- reconstruction error is low enough
- participation ratio is high enough
- representation is stable across runs

Track during training:

- reconstruction MSE
- PR of the representation layer
- fraction of active units if you want to compare with the paper

---

## Step 7: freeze encoder and export the synthetic dataset

After training:

```python
with torch.no_grad():
    H = encoder(torch.from_numpy(z).to(device)).cpu().numpy()
```

Now `H` is your synthetic high-dimensional dataset.

Use:

- `z` as the ground-truth latent variables
- `H` as the observed high-dimensional input

---

## Pseudocode summary

```python
for batch in latent_loader:
    h = encoder(batch)
    batch_hat = decoder(h)

    recon = mse(batch_hat, batch)
    pr = participation_ratio(h)

    loss = recon - lambda_dim * pr

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Important practical notes

### 1. Batchwise PR is noisy

Participation ratio is estimated from a minibatch covariance, so it can be noisy.

Two common fixes:

- use large batches
- use an exponential moving average of covariance statistics for logging

### 2. Scaling matters

If `lambda_dim` is too small, the model just becomes a normal autoencoder.

If `lambda_dim` is too large, reconstruction may collapse or become poor.

So this hyperparameter is the main thing to tune.

### 3. Sparsity is not directly forced here

In the paper, the learned representation becomes sparse, but that sparsity comes from the learned model and objective rather than from a simple explicit L1 penalty in the core construction.

So for a faithful reproduction, start without extra sparsity penalties.

### 4. Effective dimension is not the same as layer width

Even though the layer has 500 neurons, the effective dimension is much smaller.

That is expected.

The goal is not to use all 500 dimensions equally, but to create a representation that is much more distributed than the original 5-dimensional latent code.

---

## Clean conceptual summary

Their dimension-increasing method is:

1. sample a low-dimensional latent variable
2. pass it through a learned autoencoder
3. train the autoencoder to both:
   - reconstruct the latent variable
   - maximize representation dimensionality via participation ratio
4. take the high-dimensional hidden layer as the synthetic dataset

So the “dimension increase” is **not** done by a hand-designed random mapping.
It is done by a **learned nonlinear encoder** whose hidden layer is encouraged to be high-dimensional while still preserving the original latent information.

---

## Suggested file/module structure

```text
project/
  data.py              # latent sampling
  models.py            # autoencoder
  losses.py            # reconstruction + participation ratio
  train_input_model.py # training loop
  export_dataset.py    # save latent z and high-d H
  analysis.py          # PR, sparsity, visualization
```

---

## If you want to reproduce only this part

You do not need to implement the multitask model yet.

A minimal reproduction only needs:

- latent sampler
- autoencoder
- PR-based dimensionality objective
- export of the encoder representation

That is the complete “dimension increasing” part of the paper.
