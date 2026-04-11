# Convert the current conditional score-matching code to EDM style

This note describes how to modify the current codebase from **direct conditional score matching**
for a scalar variable `theta` into an **EDM-style conditional denoiser**.

The goal is to keep the overall project structure and most of the current MLP / FiLM backbone,
while changing the **parameterization**, **noise sampling**, and **loss**.

---

## 1. What the current code is doing

### Current model side

In `models.py`, the main conditional score models are:

- `ConditionalScore1D`
- `ConditionalScore1DFiLMPerLayer`

Both currently implement a model with API roughly like:

```python
pred = model(theta_tilde, x, sigma)
```

where `pred` is interpreted as the **score**

```python
s(theta_tilde, x, sigma) \approx ∂/∂theta_tilde log p(theta_tilde | x, sigma)
```

The FiLM model is already the better starting point, because it has:

- an `x` trunk,
- conditioning on `(theta_tilde, sigma)`,
- residual blocks,
- optional layer norm,
- optional bounded FiLM.

### Current trainer side

In `trainers.py`, the most relevant function is:

- `train_score_model_ncsm_continuous(...)`

The current training loop does:

```python
sigma = sample_continuous_geometric_sigmas(...)
eps = torch.randn_like(tb)
theta_tilde = tb + sigma * eps
pred = model(theta_tilde, xb, sigma)
loss = _score_matching_loss(pred, eps, sigma, ...)
```

with

```python
residual = sigma * pred + eps
loss = || residual ||^2
```

This is a standard continuous NCSM parameterization.

---

## 2. What changes in EDM style

The single biggest change is:

> **Do not train the network to output the score directly. Train it to output a denoised estimate of clean `theta`.**

So instead of learning

```python
s(theta_tilde, x, sigma)
```

you learn

```python
D(theta_tilde, x, sigma) ≈ theta
```

and recover the score later by

```python
score = (D(theta_tilde, x, sigma) - theta_tilde) / sigma**2
```

This is the main EDM-style change that should be implemented first.

---

## 3. Recommended migration strategy

Do the migration in two layers:

### Layer A: minimal change

Keep your current FiLM backbone idea, but change:

- output interpretation: **denoiser** instead of **score**,
- sigma sampler: **log-normal** instead of current `uniform_log` / `beta_log`,
- loss: **EDM weighted denoising loss**,
- evaluation: derive score from denoiser.

### Layer B: optional later upgrades

After Layer A works, improve:

- sigma embedding with Fourier features,
- residual scaling such as `(h + update) / sqrt(2)`,
- EMA of model weights,
- better default standardization for `theta` and maybe `x`.

This note focuses on **Layer A first**.

---

## 4. Model changes in `models.py`

## 4.1 Add a new EDM model instead of overwriting the old score model

Do **not** delete the old classes. Keep them as baselines.

Add a new class, for example:

```python
class ConditionalThetaEDM(nn.Module):
    ...
```

and optionally a FiLM version:

```python
class ConditionalThetaEDMFiLM(nn.Module):
    ...
```

The easiest path is:

- keep `ConditionalScore1DFiLMPerLayer` as the internal backbone template,
- add a wrapper around it that applies EDM preconditioning,
- make the wrapper return a **denoised theta estimate**.

---

## 4.2 Separate backbone output from final denoiser output

The new model should have two conceptual parts:

### Backbone

A network `F(...)` that produces a scalar residual-like output.

### EDM wrapper

A wrapper that computes:

```python
D(theta_tilde, x, sigma) = c_skip * theta_tilde + c_out * F(c_in * theta_tilde, x, c_noise)
```

where `c_in`, `c_skip`, `c_out`, `c_noise` are EDM preconditioning terms.

---

## 4.3 EDM preconditioning formulas for scalar `theta`

Add a hyperparameter:

```python
sigma_data: float = 0.5
```

Then define:

```python
sigma2 = sigma ** 2
ns2 = sigma_data ** 2

c_in   = 1.0 / torch.sqrt(sigma2 + ns2)
c_skip = ns2 / (sigma2 + ns2)
c_out  = sigma * sigma_data / torch.sqrt(sigma2 + ns2)
c_noise = 0.25 * torch.log(torch.clamp(sigma, min=1e-8))
```

For your scalar problem, this is the cleanest first adaptation.

---

## 4.4 What the new forward should return

The new `forward()` should return the **denoised estimate**:

```python
def forward(self, theta_tilde, x, sigma):
    sigma = sigma if sigma.ndim == 2 else sigma.unsqueeze(-1)

    sigma2 = sigma ** 2
    ns2 = self.sigma_data ** 2

    c_in = 1.0 / torch.sqrt(sigma2 + ns2)
    c_skip = ns2 / (sigma2 + ns2)
    c_out = sigma * self.sigma_data / torch.sqrt(sigma2 + ns2)
    c_noise = 0.25 * torch.log(torch.clamp(sigma, min=1e-8))

    backbone_out = self.backbone(c_in * theta_tilde, x, c_noise)
    denoised = c_skip * theta_tilde + c_out * backbone_out
    return denoised
```

Important: the backbone should now interpret the third argument as a **noise feature**, not necessarily the raw sigma.

---

## 4.5 Keep a `predict_score()` API

Your downstream code may still expect a score. Preserve that convenience:

```python
@torch.no_grad()
def predict_score(self, theta_tilde, x, sigma_eval):
    self.eval()
    sigma = torch.full((theta_tilde.shape[0], 1), float(sigma_eval), device=theta_tilde.device)
    denoised = self.forward(theta_tilde, x, sigma)
    return (denoised - theta_tilde) / torch.clamp(sigma ** 2, min=1e-8)
```

This lets you train in EDM style while keeping score-based downstream analysis.

---

## 4.6 Minimal backbone reuse strategy

The fastest route is to reuse the existing FiLM architecture with only small edits.

Currently, `ConditionalScore1DFiLMPerLayer` conditions on:

```python
cond = torch.cat([theta_tilde, sigma_feat], dim=-1)
```

For the minimal EDM conversion, that is acceptable if you reinterpret inputs as:

- first scalar = `c_in * theta_tilde`,
- second scalar = `c_noise`.

So the backbone can remain almost unchanged.

A very practical implementation path is:

- rename current class internally to something like `ConditionalThetaBackboneFiLM`, or
- leave it unchanged and instantiate it as a backbone,
- but document clearly that its output is no longer a score.

---

## 4.7 Optional better sigma embedding later

Later, replace the scalar `log_sigma` or `c_noise` feature with a Fourier embedding:

```python
emb = fourier_embed(log_sigma)
emb = mlp(emb)
```

and use that embedding for FiLM.

But this is **not the first thing** to implement.

---

## 5. Trainer changes in `trainers.py`

## 5.1 Keep the old trainer and add a new EDM trainer

Do not overwrite `train_score_model_ncsm_continuous(...)`.
Keep it as a baseline.

Add a new function:

```python
def train_theta_edm_model(...):
    ...
```

This keeps comparisons clean.

---

## 5.2 Replace geometric-log sigma sampling with EDM sigma sampling

Right now you use:

```python
sample_continuous_geometric_sigmas(...)
```

EDM uses a log-normal sigma distribution. Add:

```python
def sample_edm_sigmas(batch_size, P_mean, P_std, device):
    rnd_normal = torch.randn((batch_size, 1), device=device)
    sigma = torch.exp(rnd_normal * P_std + P_mean)
    return sigma
```

Suggested defaults to start:

```python
P_mean = -1.2
P_std = 1.2
sigma_data = 0.5
```

These are the official EDM defaults, but you may tune them for your `theta` scale.

---

## 5.3 Add an EDM denoising loss

Add a new loss helper:

```python
def _edm_theta_loss(denoised, clean_theta, sigma, sigma_data, *, loss_type="mse", huber_delta=1.0):
    weight = (sigma ** 2 + sigma_data ** 2) / torch.clamp((sigma * sigma_data) ** 2, min=1e-8)
    residual = denoised - clean_theta
    return _loss_reduce(weight * residual, torch.zeros_like(residual), loss_type=loss_type, huber_delta=huber_delta)
```

A numerically clearer alternative is:

```python
def _edm_theta_loss(denoised, clean_theta, sigma, sigma_data):
    weight = (sigma ** 2 + sigma_data ** 2) / torch.clamp((sigma * sigma_data) ** 2, min=1e-8)
    return torch.mean(weight * (denoised - clean_theta) ** 2)
```

For the first EDM pass, I recommend plain **MSE**, not Huber.

---

## 5.4 The new training loop

The new training loop should look like:

```python
for tb, xb in loader:
    tb = tb.to(device, non_blocking=True)
    xb = xb.to(device, non_blocking=True)

    sigma = sample_edm_sigmas(tb.shape[0], P_mean, P_std, tb.device)
    eps = torch.randn_like(tb)
    theta_tilde = tb + sigma * eps

    denoised = model(theta_tilde, xb, sigma)
    loss = _edm_theta_loss(denoised, tb, sigma, sigma_data)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(...)
    optimizer.step()
```

The validation loop should mirror this exactly.

---

## 5.5 What to remove from the EDM path

For the EDM path, do **not** use:

- `_score_matching_loss(...)`
- `normalize_by_sigma`
- `sample_continuous_geometric_sigmas(...)`

These belong to the old NCSM path and should remain only for baseline comparison.

---

## 5.6 Suggested trainer signature

A practical function signature is:

```python
def train_theta_edm_model(
    model,
    theta_train,
    x_train,
    epochs,
    batch_size,
    lr,
    device,
    log_every,
    theta_val=None,
    x_val=None,
    early_stopping_patience=30,
    early_stopping_min_delta=1e-4,
    early_stopping_ema_alpha=0.05,
    early_stopping_ema_warmup_epochs=0,
    restore_best=True,
    optimizer_name="adamw",
    weight_decay=1e-4,
    max_grad_norm=1.0,
    lr_scheduler="cosine",
    lr_warmup_frac=0.05,
    loss_type="mse",
    huber_delta=1.0,
    abort_on_nonfinite=True,
    P_mean=-1.2,
    P_std=1.2,
    sigma_data=0.5,
):
    ...
```

This keeps the interface close to your current trainer.

---

## 6. EMA of weights: recommended but second priority

Your trainer already uses EMA smoothing of validation loss for early stopping.
That is different from **EMA of model parameters**.

After the EDM loss is working, add parameter EMA.

### Add helper functions

```python
def init_ema_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def update_ema(model, ema_state, decay):
    with torch.no_grad():
        cur = model.state_dict()
        for k in ema_state:
            ema_state[k].mul_(decay).add_(cur[k].detach(), alpha=1.0 - decay)
```

### Recommended usage

- initialize EMA before training,
- update EMA after every optimizer step,
- optionally evaluate validation loss using EMA weights,
- save best EMA weights, not only raw weights.

Good starting value:

```python
ema_decay = 0.999
```

But again, this is **not** the top priority. The first priority is changing the target/loss.

---

## 7. Standardization recommendations

EDM-style training behaves best when the target scale is well controlled.

For your scalar `theta`, I recommend:

1. standardize training `theta` to roughly zero mean and unit standard deviation,
2. train the EDM model in standardized space,
3. convert back only when needed.

If `x` is poorly scaled across dimensions, standardizing `x` can help too.

This makes `sigma_data = 0.5` a much more sensible default.

---

## 8. Minimal code patch plan

If you want the fewest edits, do the following in order.

### Step 1
Add a new model class:

- `ConditionalThetaEDM`

that wraps a backbone and returns `denoised_theta`.

### Step 2
Add a new sigma sampler:

- `sample_edm_sigmas(...)`

### Step 3
Add a new loss:

- `_edm_theta_loss(...)`

### Step 4
Add a new trainer:

- `train_theta_edm_model(...)`

### Step 5
Add `predict_score()` for the new model:

```python
(denoised - theta_tilde) / sigma**2
```

### Step 6
Only after the above works, add EMA-of-weights and better sigma embeddings.

---

## 9. Suggested class layout

A clean organization is:

```python
class ConditionalThetaEDMBackboneFiLM(nn.Module):
    # nearly the same as current ConditionalScore1DFiLMPerLayer
    # but semantically just a scalar backbone
    ...

class ConditionalThetaEDM(nn.Module):
    def __init__(self, backbone, sigma_data=0.5):
        ...

    def forward(self, theta_tilde, x, sigma):
        ...  # returns denoised theta

    def predict_score(self, theta_tilde, x, sigma_eval):
        ...
```

This avoids confusion between “backbone output” and “final denoised output”.

---

## 10. Suggested initial hyperparameters

If `theta` is standardized:

```python
sigma_data = 0.5
P_mean = -1.2
P_std = 1.2
loss_type = "mse"
optimizer_name = "adamw"
weight_decay = 1e-4
max_grad_norm = 1.0
lr_scheduler = "cosine"
lr_warmup_frac = 0.05
```

Model side:

```python
hidden_dim = 128 or 256
depth = 3 or 4
use_layer_norm = True
zero_out_init = True
gated_film = True
```

The safest first comparison is to use the current FiLM architecture with only the EDM wrapper and EDM trainer changed.

---

## 11. How to compare old vs new fairly

Keep both training paths:

- old: `train_score_model_ncsm_continuous(...)`
- new: `train_theta_edm_model(...)`

Then compare on the same dataset and seed:

- train loss stability,
- validation loss,
- frequency of non-finite events,
- gradient clipping frequency,
- final score accuracy on a toy problem with known analytic score.

This will tell you whether EDM-style denoising actually improves the scalar conditional problem.

---

## 12. Common mistakes to avoid

### Mistake 1
Changing only the sigma sampler but still training the raw score.

That is not the main EDM benefit.

### Mistake 2
Returning the backbone output directly as the denoised theta without EDM preconditioning.

You want:

```python
denoised = c_skip * theta_tilde + c_out * backbone_out
```

not just:

```python
denoised = backbone(theta_tilde, x, sigma)
```

### Mistake 3
Using both EDM loss and old `_score_matching_loss(...)` in the same training path.

Pick one training objective per model.

### Mistake 4
Turning on too many upgrades at once.

First change only:

- model output meaning,
- sigma sampler,
- loss.

Then add EMA and richer sigma embeddings later.

---

## 13. Final recommendation

If you only implement one thing, implement this:

> Add a new EDM-style conditional denoiser model and train it with weighted denoising loss.

Everything else is secondary.

A good first milestone is:

1. `ConditionalThetaEDM` works,
2. `train_theta_edm_model(...)` runs end-to-end,
3. `predict_score()` recovers a reasonable score from the denoiser,
4. old and new training paths can be compared side by side.

Once that works, the next best improvements are:

- EMA of weights,
- better sigma embedding,
- standardized `theta` defaults.

