# Plan: implement deterministic OT-CFM with TorchCFM

## Goal

Train a deterministic OT-CFM model using the `torchcfm` package for a sparse bimodal target distribution, so the model pairs source and target samples with an OT joint and then learns the usual affine conditional flow-matching velocity field. TorchCFM exposes this directly as `ExactOptimalTransportConditionalFlowMatcher`, while the base matcher returns the usual `(t, x_t, u_t)` training objects for conditional flow matching.

## Main idea

Use TorchCFM to handle the OT pairing step and conditional path sampling, instead of implementing minibatch OT manually.

At each training step:

1. sample a source minibatch `x0` from a known base distribution, usually standard Gaussian,
2. sample a target minibatch `x1` from the dataset,
3. call `ExactOptimalTransportConditionalFlowMatcher.sample_location_and_conditional_flow(x0, x1)`,
4. get `t`, `x_t`, and target velocity `u_t`,
5. train a neural network `v_theta(x_t, t)` by MSE regression to predict `u_t`.

This matches the package design: TorchCFM defines `ExactOptimalTransportConditionalFlowMatcher` for OT-CFM, and the underlying conditional flow matcher uses the affine Gaussian path with mean `t x1 + (1 - t) x0`, conditional flow `u_t = x1 - x0`, and a helper that returns `t, xt, ut`.

## Scope

This plan targets the **unconditional** case first:

- learn a flow from Gaussian base `x0 ~ N(0, I)` to target data `x1 ~ p_data`,
- generate samples by integrating the learned ODE forward,
- later extend to a conditional model if needed.

This is a good first step because the TorchCFM repo already includes 2D tutorials and unconditional image examples, so the workflow is aligned with how the package is intended to be used.

## Dependencies

Install the core package first:

- `torch`
- `torchcfm`
- optionally `matplotlib` for 2D visualization

Example:

```bash
pip install torch torchcfm matplotlib
```

TorchCFM is packaged as an installable library and includes the OT-CFM loss class directly.

## Files to create

### 1. `data.py`

Purpose:

- generate or load the bimodal dataset,
- return batches of target samples `x1`.

For a first experiment, use a toy 2D dataset with two Gaussian modes.

### 2. `model.py`

Purpose:

- define the velocity network `v_theta(x, t)`,
- use a small MLP plus time embedding.

Recommended structure:

- input: concatenation of `x` and a time embedding of `t`,
- hidden layers: 2 to 4 MLP layers,
- output: same dimension as `x`.

### 3. `train.py`

Purpose:

- instantiate `ExactOptimalTransportConditionalFlowMatcher`,
- run the training loop,
- save checkpoints,
- plot loss and generated samples.

### 4. `sample.py`

Purpose:

- load the trained model,
- integrate the deterministic ODE from Gaussian noise to target samples,
- plot generated points.

## Step-by-step implementation

## Step 1: build a toy bimodal dataset

Start with a very simple target distribution:

- mode 1 centered near `(-2, 0)`,
- mode 2 centered near `(2, 0)`,
- both with small isotropic Gaussian noise.

Why:

- it lets you quickly see whether the model still collapses to the middle,
- it is easy to visualize during debugging.

Success criterion:

- raw target scatter plot clearly shows two separated modes.

## Step 2: define the velocity model

Implement a network

```math
v_\theta(x, t): \mathbb{R}^d \times [0,1] \to \mathbb{R}^d
```

Recommended minimal architecture:

- sinusoidal or simple learned embedding for `t`,
- concatenate time embedding with `x`,
- MLP with `SiLU` activations,
- final linear layer producing velocity in data space.

Initial target setup:

- data dimension `d = 2`,
- hidden size `128`,
- 3 hidden layers.

## Step 3: instantiate TorchCFM OT-CFM

In `train.py`, create:

```python
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.05)
```

Reason:

- this class is the package’s OT-CFM implementation, where the conditional distribution is based on an exact OT joint `π(x0, x1)`.

## Step 4: write the training loop

At each iteration:

1. sample `x1` from the target batch,
2. sample `x0` from standard Gaussian with the same batch size,
3. call:

```python
t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
```

4. predict:

```python
pred = model(xt, t)
```

5. optimize:

```python
loss = mse(pred, ut)
```

This follows the TorchCFM base API, where the matcher samples `x_t` from the affine Gaussian path and computes `u_t = x_1 - x_0`.

## Step 5: sample from the learned deterministic flow

After training, generate samples by solving the learned ODE forward:

```math
\frac{d x_t}{dt} = v_\theta(x_t, t)
```

Implementation choices:

- start with simple Euler integration,
- later switch to `torchdiffeq` if needed,
- use `x(0) ~ N(0, I)` as the base sample.

Success criterion:

- sampled points should separate into two clusters instead of filling the middle.

## Step 6: evaluate whether OT-CFM fixed the mode problem

Compare against a baseline using `ConditionalFlowMatcher` instead of the OT version.

Metrics to check:

- scatter plot of generated samples,
- per-mode sample counts,
- distance between generated cluster centers and target centers,
- optional kernel density estimate in 2D.

Main comparison:

- **baseline CFM** may transport samples in a way that blurs modes,
- **OT-CFM** should preserve the two-mode structure better because it uses an OT joint instead of independent pairing.

## Suggested training settings

For a first 2D run:

- batch size: `256`
- learning rate: `1e-3`
- optimizer: `AdamW`
- training steps: `3000` to `5000`
- path noise `sigma`: start with `0.05`, then test `0.0` and `0.1`

Notes:

- if training is unstable, reduce learning rate to `3e-4`,
- if generated samples are too blurry, try smaller `sigma`,
- if training overfits or oscillates, add weight decay and larger batch size.

## Debugging checklist

### If the model still collapses to the middle

Check:

- whether the velocity network is too small,
- whether `sigma` is too large,
- whether the ODE sampler uses enough time steps,
- whether the dataset really has balanced modes,
- whether the baseline and OT-CFM are being compared under the same settings.

### If training loss decreases but samples look bad

Check:

- time embedding quality,
- numerical integration quality during sampling,
- whether the learned field is smooth enough,
- whether the model is underfitting the target velocity.

### If import or API errors occur

Check:

- installed TorchCFM version,
- correct import path from `torchcfm.conditional_flow_matching`,
- package dependencies from the repo.

## Minimal milestone plan

### Milestone 1

Train OT-CFM on a 2D bimodal toy dataset and generate a scatter plot.

Deliverable:

- one script that trains,
- one saved checkpoint,
- one figure comparing real vs generated samples.

### Milestone 2

Train both:

- `ConditionalFlowMatcher`
- `ExactOptimalTransportConditionalFlowMatcher`

Deliverable:

- side-by-side visual comparison,
- short note on whether OT-CFM reduces the middle-mode artifact.

### Milestone 3

Extend to your real dataset.

Deliverable:

- replace toy `x1` samples with your actual data,
- keep the same training loop,
- adapt the model dimension and architecture.

## Extension to conditional modeling

If your real task is conditional, for example learning `p(theta | x)`, modify the plan like this:

- OT pairing happens in the transported variable space, usually `theta`,
- the condition `x` is fed into the velocity network,
- the model becomes:

```math
v_\theta(\theta_t, t, x)
```

Training sketch:

1. batch `(theta1, x)` from data,
2. sample `theta0` from base Gaussian,
3. use OT-CFM matcher on `(theta0, theta1)`,
4. predict `u_t` from `(theta_t, t, x)`.

That conditional version is not a new TorchCFM algorithm; it is the same OT-CFM training pattern with an augmented network input.

## Final deliverable

The final result should be a clean training pipeline with:

- TorchCFM OT-CFM matcher,
- MLP velocity model,
- toy bimodal benchmark,
- deterministic ODE sampler,
- visual comparison against standard CFM.

## Next action

Implement the unconditional 2D version first, verify that two modes are preserved, then adapt the same structure to your real problem.
