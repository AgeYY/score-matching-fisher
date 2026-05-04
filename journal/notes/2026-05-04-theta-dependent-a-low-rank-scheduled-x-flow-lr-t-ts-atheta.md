# 2026-05-04 — Theta-dependent $A(t,\theta)$ low-rank scheduled X-flow (`linear_x_flow_lr_t_ts_atheta`)

## Question / context

Scheduled **linear X-flow** with a **low-rank nonlinear correction** already exists as `linear_x_flow_lr_t_ts`: symmetric **matrix drift $A(t)$** depends only on time, while **$b(\theta)$** depends only on $\theta$, with a two-stage recipe (mean-regression warmup on $b$, then freeze $b$ for flow matching). This note records the sibling method **`linear_x_flow_lr_t_ts_atheta`**, which lets the **base drift matrix depend on both $t$ and $\theta$** while keeping the same correction, likelihood path, and $b$-warmup story.

## Method

Work in **normalized** coordinates $\tilde{x} = (x - m) \oslash s$ as for other linear X-flow variants. The velocity is

$$
v(\tilde{x}, t, \theta) = A(t,\theta)\,\tilde{x} + b(\theta) + U\,h\!\left(U^\top \tilde{x},\, t,\, \theta\right),
$$

where $U \in \mathbb{R}^{d \times r}$ has **orthonormal columns** (fixed after initialization), and $h$ is an MLP on $(U^\top \tilde{x},\, t,\, \theta)$ with values in $\mathbb{R}^r$.

### Symmetric $A(t,\theta)$ from $B(t,\theta)$

A network produces an unconstrained matrix $B(t,\theta) \in \mathbb{R}^{d \times d}$ from the concatenated input $[t,\theta]$ (with $t$ as a scalar channel per sample). The drift uses the **symmetric part**

$$
A(t,\theta) = \frac{1}{2}\left(B(t,\theta) + B(t,\theta)^\top\right).
$$

There is **no** separate diagonal identity offset in the low-rank wrapper configuration used for this method token (same spirit as the `lr_t_ts` linear head: the learnable symmetric part carries the dynamics).

### Offset $b(\theta)$ and two-stage training

The offset **does not** take $t$ as input:

$$
b(\theta,\, t) = b(\theta) \quad\text{(implemented as an MLP on $\theta$ only).}
$$

Training matches `linear_x_flow_lr_t_ts`:

1. **Warmup:** fit $b(\theta)$ by **mean squared error** to targets derived from normalized data at the bridge endpoint (requires `--lxf-low-rank-t-warmup-epochs >= 1`).
2. **Main stage:** **freeze** all parameters of `b_net`, then run **scheduled** flow matching on the full velocity (including $A(t,\theta)$, $U$, and $h$).

Implementation detail: the convergence script toggles `requires_grad` on `model.linear.b_net` parameters so the existing helper `train_low_rank_t_theta_only_b_mean_regression_pretrain_then_freeze_b` applies unchanged (see [`fisher/linear_x_flow.py`](../../fisher/linear_x_flow.py) and [`bin/study_h_decoding_convergence.py`](../../bin/study_h_decoding_convergence.py)).

### Divergence and likelihood

Because the correction enters as $U h(U^\top \tilde{x}, t, \theta)$ with fixed orthonormal $U$, the **divergence** splits as for other low-rank scheduled models:

$$
\nabla_{\tilde{x}} \cdot v = \operatorname{tr}\!\left(A(t,\theta)\right) + \operatorname{tr}\!\left(\frac{\partial h}{\partial z}\right), \qquad z = U^\top \tilde{x}.
$$

Here $\operatorname{tr}(A(t,\theta))$ is the sum of diagonal entries of the **batch** symmetric matrix (since $A$ varies with $\theta$). The trace of $\partial h / \partial z$ is estimated by **Hutchinson** probes (default) or an **exact** rank loop, controlled like other low-rank scheduled LXF methods.

There is **no** closed-form Gaussian endpoint for the composite field; **log-density** follows the same **reverse-time ODE + integrated divergence** path as `linear_x_flow_lr_t_ts` (and analytic-Gaussian $H$ shortcuts remain disabled for this family).

### Relation to `linear_x_flow_lr_t_ts`

| Aspect | `linear_x_flow_lr_t_ts` | `linear_x_flow_lr_t_ts_atheta` |
|--------|-------------------------|--------------------------------|
| Base matrix drift | $A(t)$ from MLP on $t$ only | $A(t,\theta)=\frac12(B+B^\top)$ from MLP on $[t,\theta]$ |
| Offset | $b(\theta)$ | $b(\theta)$ (same warmup + freeze) |
| Low-rank correction | $U h(U^\top x, t, \theta)$ | Same |
| CLI token | `linear_x_flow_lr_t_ts` | `linear_x_flow_lr_t_ts_atheta` (alias `linear-x-flow-lr-t-ts-atheta`) |

### Initialization note ($B$ head)

The `matrix_net` last layer uses a **nonzero** `final_gain` so that, at initialization, $B$ (hence $A$) actually varies with $(t,\theta)$. A zero final gain would zero the last linear layer in `_make_mlp`, making $A\equiv 0$ for all inputs until late optimization—undesirable for a $\theta$-indexed matrix field. The sibling `lr_t_ts` linear head uses zero gain for its scalar/diagonal construction; the **full matrix** head for `atheta` is handled differently on purpose (see `ConditionalTimeThetaMatrixThetaOnlyBLowRankCorrectionLinearXFlowMLP` in code).

## Figure

Schematic split of the velocity into base linear part, $\theta$-only offset, and low-rank correction:

![Velocity decomposition for `linear_x_flow_lr_t_ts_atheta`: $A(t,\theta)x + b(\theta) + U h(U^\top x,t,\theta)$](figs/theta-dependent-a-lr-t-ts-atheta/velocity_split.png)

The caption matches the implemented field in normalized space; training still uses the bridge and scheduled FM exactly as other scheduled LXF rows.

## Reproduction (commands & scripts)

**Core classes:** `ConditionalTimeThetaMatrixThetaOnlyLinearXFlowMLP` (base $A,b$) and `ConditionalTimeThetaMatrixThetaOnlyBLowRankCorrectionLinearXFlowMLP` (adds $U,h$ + divergence / ODE likelihood) in [`fisher/linear_x_flow.py`](../../fisher/linear_x_flow.py).

**H-decoding convergence** (canonical token; alias with hyphens is accepted):

```bash
cd /grad/zeyuan/score-matching-fisher
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py ... \
  --theta-field-method linear_x_flow_lr_t_ts_atheta \
  --lxf-low-rank-dim <r> \
  --lxf-low-rank-t-warmup-epochs 1 \
  --device cuda
```

Replace `...` with your dataset, bins, and budget flags as in other LXF convergence runs. Script entry: [`bin/study_h_decoding_convergence.py`](../../bin/study_h_decoding_convergence.py) (method branch `linear_x_flow_lr_t_ts_atheta`, same training dispatch as `linear_x_flow_lr_t_ts` for the $b$-warmup path).

**Twofig row token** is documented alongside `linear_x_flow_lr_t_ts` in the valid-row comment in [`bin/study_h_decoding_twofig.py`](../../bin/study_h_decoding_twofig.py).

**Unit / smoke tests:**

```bash
mamba run -n geo_diffusion python -m pytest \
  tests/test_linear_x_flow.py::TestLinearXFlow::test_lr_t_ts_atheta_b_independent_of_t_A_depends_on_theta \
  tests/test_linear_x_flow.py::TestLinearXFlow::test_lr_t_ts_atheta_mean_regression_pretrain_then_freeze_b_smoke \
  tests/test_study_h_decoding_convergence_gaussian_network.py -q --tb=short
```

## Artifacts

- **Figure (this note):** `/grad/zeyuan/score-matching-fisher/journal/notes/figs/theta-dependent-a-lr-t-ts-atheta/velocity_split.png`
- **Note source:** `/grad/zeyuan/score-matching-fisher/journal/notes/2026-05-04-theta-dependent-a-low-rank-scheduled-x-flow-lr-t-ts-atheta.md`

No benchmark twofig run is bundled with this documentation entry; add NPZ/SVG paths here when you have a canonical experiment directory.

## Takeaway

`linear_x_flow_lr_t_ts_atheta` is the same **scheduled low-rank X-flow** likelihood machinery as `linear_x_flow_lr_t_ts`, but replaces **$A(t)$** with a **symmetric $A(t,\theta)$** so the linear part of the drift can co-vary with $\theta$ at fixed $t$. The **$b(\theta)$ mean-regression warmup and freeze** behavior is unchanged, which keeps one shared trainer and stable metadata fields (`lxf_low_rank_t_warmup_objective = mean_regression`, second-stage freeze flags) aligned with `lr_t_ts`.
