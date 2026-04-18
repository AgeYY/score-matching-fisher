# Refactor plan: generalize CTSM path code with pluggable schedules

## Goal

Refactor the current CTSM-v implementation so the path is no longer hard-coded to the linear two-endpoint bridge

\[
x_t = (1-t)x_0 + t x_1 + \sigma \sqrt{t(1-t)}\,\varepsilon,
\]

and can instead be selected by a simple interface such as:

```python
prob_path = TwoEndpointBridge(dim=dim, var=2.0, scheduler="linear")
prob_path = TwoEndpointBridge(dim=dim, var=2.0, scheduler="cosine")
```

The refactor should keep the training and evaluation code almost unchanged, while moving all path-specific logic into a path/scheduler module.

---

## Why this refactor is useful

The current code mixes together three concerns:

1. **Bridge geometry**: mean and variance of the two-endpoint bridge.
2. **Schedule choice**: whether the interpolation clock is linear, cosine, or something else.
3. **CTSM-v target construction**: the weighted analytic target used in the loss.

Separating these concerns will make it much easier to:

- add new schedules without rewriting the objective code,
- compare linear vs cosine fairly,
- test raw and weighted targets independently,
- reuse the same objective functions for multiple paths.

---

## Current structure and what should change

### Current path code

The current `TwoSB` implementation is linear-only:

- `marginal_prob()` uses `(1 - t) * x0 + t * x1`,
- the variance is `t * (1 - t) * var`,
- `full_epsilon_target()` uses the corresponding linear closed-form target.

### Current objective code

The current objectives call the path object for:

- `marginal_prob(...)`,
- `full_epsilon_target(...)`.

This is already a good boundary. The main missing piece is that the path object itself is not general yet.

---

## Target design

### 1. Introduce a scheduler abstraction

Create a small scheduler interface with two methods:

```python
class Scheduler(Protocol):
    def value(self, t: torch.Tensor) -> torch.Tensor:
        ...
    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        ...
```

This represents a monotone clock

\[
u = s(t), \qquad u \in [0,1].
\]

Implement at least:

- `LinearScheduler`: \(s(t)=t\)
- `CosineScheduler`: \(s(t)=\frac{1-\cos(\pi t)}{2}\)

The cosine schedule should also provide

\[
s'(t)=\frac{\pi}{2}\sin(\pi t).
\]

### 2. Replace `TwoSB` with a general bridge class

Create a new class such as:

```python
class TwoEndpointBridge:
    def __init__(self, dim, var=2.0, scheduler="linear", eps=1e-12):
        ...
```

This class should implement the generalized path

\[
x_t = (1-s(t))x_0 + s(t)x_1 + \sigma\sqrt{s(t)(1-s(t))}\,\varepsilon.
\]

Inside the class, compute

\[
u=s(t), \qquad \dot u=s'(t).
\]

Then define

\[
\mu_t=(1-u)x_0+u x_1, \qquad
k_t=u(1-u)\sigma^2.
\]

### 3. Keep the objective API stable

The objective code should continue to use the path object through:

- `marginal_prob(x0, x1, t)`
- `full_epsilon_target(epsilon, x0, x1, t, factor=...)`

That way, the objective code does not need to know whether the underlying schedule is linear or cosine.

### 4. Move raw target and weighting into the path class

Inside the path class, define separate methods for:

- `raw_vector_target(...)`
- `time_score_normalization(...)`
- `full_epsilon_target(...)`

This separation is useful because:

- `raw_vector_target` is the mathematically clean object,
- `time_score_normalization` handles weighting,
- `full_epsilon_target` combines them for training.

This makes debugging much easier.

---

## Mathematical rule for generalization

The key idea is to replace linear time \(t\) by a schedule clock \(u=s(t)\).

For the two-endpoint Gaussian bridge, define

\[
\epsilon_t = \frac{x - \mu_t}{\sqrt{k_t}}.
\]

Then the vectorized conditional time score is

\[
\mathrm{vec}(\partial_t \log p_t(x\mid z))
=
-\frac{\dot k_t}{2k_t}\mathbf 1
+ \frac{1}{\sqrt{k_t}}\dot\mu_t \odot \epsilon_t
+ \frac{\dot k_t}{2k_t}\epsilon_t^{\odot 2}.
\]

For the scheduled bridge,

\[
\dot\mu_t = \dot u (x_1-x_0),
\qquad
\dot k_t = \dot u (1-2u)\sigma^2.
\]

So the target is obtained automatically once \(u\) and \(\dot u\) are available from the scheduler.

This means the path-specific formulas should live in the path class, not in the objective file.

---

## Proposed file layout

### Option A: minimal-change refactor

Keep the current file names and only change internals:

- `ctsm_paths.py`
- `ctsm_objectives.py`

This is best if other code already imports these exact modules.

### Option B: transition refactor

Add new files first, then migrate imports later:

- `ctsm_paths_general.py`
- `ctsm_objectives_general.py`

This is safer for experimentation because you can compare old and new behavior side by side.

Recommended workflow:

1. implement the generalized files,
2. verify linear schedule reproduces old behavior,
3. switch imports once confirmed.

---

## Step-by-step implementation plan

### Phase 1: generalize the path module

1. Add scheduler classes:
   - `LinearScheduler`
   - `CosineScheduler`
2. Add a scheduler registry:
   ```python
   SCHEDULER_REGISTRY = {
       "linear": LinearScheduler(),
       "cosine": CosineScheduler(),
   }
   ```
3. Add a `build_scheduler(...)` helper.
4. Implement `TwoEndpointBridge`.
5. Keep a backward-compatible alias:
   ```python
   TwoSB = TwoEndpointBridge
   ```

### Phase 2: keep objective code path-agnostic

1. Update type hints in the objective file to accept `TwoEndpointBridge`.
2. Leave the loss functions structurally unchanged.
3. Ensure all path-specific calculations stay inside the path object.

### Phase 3: verify correctness

Run three checks:

#### Check 1: linear schedule reproduces old behavior
For the same random seed and same inputs, verify that:

- `marginal_prob` matches old `TwoSB`,
- `full_epsilon_target` matches old target,
- training loss values are numerically the same up to tiny tolerance.

#### Check 2: cosine schedule is numerically stable
Verify:

- no NaNs near endpoints,
- `t_eps` and `eps` are enough to prevent division-by-zero,
- training runs with similar tensor shapes and device behavior.

#### Check 3: scalar integration path still works
Verify that

- `estimate_log_ratio_trapz`
- `estimate_log_ratio_trapz_pair`
- `estimate_log_ratio_torch`

still work without modification when using the new path class.

---

## Important implementation details

### Endpoint stability

For cosine and bridge-like schedules, the variance goes to zero at the endpoints. Keep:

- `t_eps` in the objective sampling,
- `eps` inside the path class for clamping denominators.

### Preserve tensor broadcasting

Make sure `t` keeps shape `(batch, 1)` so that:

- schedule outputs broadcast correctly,
- mean/std broadcast against `x0` and `x1`,
- vector targets stay shape-compatible.

### Preserve device and dtype behavior

All helper functions should return tensors on the same device and dtype as inputs.

### Keep the model interface unchanged

The model should still receive `(x_t, t)` or `(x_t, t, m, delta)`. Do not replace `t` by `u` at the model input in the first refactor. That keeps the comparison between schedules conceptually simple.

---

## Suggested tests

### Unit tests

1. **Scheduler tests**
   - linear: `value(t)=t`, `derivative(t)=1`
   - cosine: correct endpoints and derivative

2. **Path tests**
   - `marginal_prob` returns expected mean/std for linear and cosine
   - standard deviation is zero only at the endpoints up to clamp

3. **Target tests**
   - for linear schedule, new target equals old target
   - target tensor has same shape as `x0`

### Smoke tests

1. one forward/backward pass of `ctsm_v_two_sample_loss`
2. one forward/backward pass of `ctsm_v_pair_conditioned_loss`
3. one call to each log-ratio estimator

---

## Recommended migration strategy

1. Keep the old files untouched.
2. Introduce the generalized files.
3. Verify linear schedule reproduces the old results.
4. Add cosine experiments.
5. After validation, either:
   - rename the generalized files to replace the originals, or
   - keep both and use the generalized version as the new default.

---

## Optional next improvements

After the basic refactor works, consider:

1. adding a general `ScheduleBridge` base class,
2. exposing `raw_scalar_target(...)` in addition to vector target,
3. allowing custom callables as schedulers,
4. adding a config-driven constructor:
   ```python
   TwoEndpointBridge.from_config(cfg.path)
   ```
5. separating weighting strategy from path geometry if you later want to compare multiple normalization rules.

---

## Suggested final interface

```python
from fisher.ctsm_paths import TwoEndpointBridge

prob_path = TwoEndpointBridge(
    dim=dim,
    var=2.0,
    scheduler="linear",   # or "cosine"
)

loss = ctsm_v_two_sample_loss(
    model=model,
    prob_path=prob_path,
    x0=x0,
    x1=x1,
)
```

This is the main refactor target: path choice becomes a constructor option, while the rest of the training code remains unchanged.
