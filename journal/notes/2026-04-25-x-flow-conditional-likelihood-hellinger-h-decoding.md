# x-flow: conditional $x$-space flow matching for $\log p(x\mid\theta)$ and H-decoding

This note summarizes the **x-flow** method as used in H-decoding convergence studies, in particular via `bin/study_h_decoding_convergence.py` with `--theta-field-method x_flow`. The implementation name inside the H-matrix code is `flow_x_likelihood` (see `fisher/h_matrix.py`).

## Goal

We observe pairs $(\theta, x)$ with $x \in \mathbb{R}^d$ and want **conditional log-likelihoods** $\log p(x\mid\theta)$ from a learned model. Off-diagonal **log-likelihood ratios** between parameter values for the same $x$ then feed the same binned-$\sqrt{H^2}$ / correlation pipeline as other field methods: build a matrix of contrasts, map them to a Hellinger-type summary, and compare to a generative ground truth.

Unlike **`theta_flow`**, the x-flow path **does not** model a prior over $\theta$ or run a $\theta$-space ODE; there is **no** `model_prior` in `flow_x_likelihood` mode.

## Model class: conditional flow in observation space

The network is a **conditional** velocity $v_t(x_t;\theta)$ that, for each fixed $\theta$, defines a flow from a simple base density (typically a standard Gaussian on $\mathbb{R}^d$) to the conditional law $p(x\mid\theta)$. Training follows **flow matching** on $(\theta, x)$: sample $t \sim \mathrm{Unif}(0,1)$, $x_0 \sim \mathcal{N}(0,I)$, interpolate $x_t$ and target velocity on the path (e.g. cosine or VP path from `flow_matching`), and minimize mean squared error between the network output and the path’s conditional velocity, **conditioned on the same** $\theta$ for each example.

Concretely, see `train_conditional_x_flow_model` in `fisher/trainers.py` (optional two-stage pretrain: fix $\theta$ to the training mean for an initial stage, then full conditional training via `--flow-x-two-stage-mean-theta-pretrain` where supported).

Architecture is selected with `--flow-arch` (e.g. MLP, FiLM, or Fourier-$\theta$ features via the `--flow-x-theta-fourier-*` group for `x_flow`, as in the study script’s docstring).

## From velocity to $\log p(x\mid\theta)$

At evaluation time, the trained velocity is wrapped in the **same ODE likelihood machinery** as $\theta$-space flows: integrate from $t=1$ (data) to $t=0$ (base), accumulating **divergence** of the velocity w.r.t. the ODE state $x$ so the instantaneous change-of-variables formula gives $\log p_t$ along the trajectory. The base log-density is the standard normal log-pdf in $d$ dimensions. This yields $\log p(x\mid\theta)$ at the observation $x$ for each conditional $\theta$ of interest.

The estimator loops over (blocks of) rows $i$ and columns $j$ and sets

$$
C_{ij} = \log p(x_i \mid \theta_j),
$$

implemented in `HMatrixEstimator.compute_x_conditional_loglik_matrix` in `fisher/h_matrix.py` (one conditional $\theta$ per column, one observation $x$ per row).

**Hyperparameters** that affect this phase include the **flow ODE** discretization (`--flow-ode-steps` in the study script, internal `flow_ode_steps`) and the training **path** scheduler (e.g. `--flow-scheduler cosine`).

## From $\log p(x\mid\theta)$ to $\Delta L$ and the Hellinger panel

The script reuses the generic construction:

- **Contrast matrix (off-diagonals):**  
  $\Delta L_{ij} = C_{ij} - C_{ii} = \log p(x_i\mid\theta_j) - \log p(x_i\mid\theta_i)$, i.e. subtract the row’s diagonal, implemented as `compute_delta_l` in `fisher/h_matrix.py`.

- **Directed then symmetrized H:** the symmetric Hellinger transform $\psi(u)=1-\operatorname{sech}(u/2)$ is applied to $\Delta L/2$ (clipped) to get a **directed** matrix, then **symmetrized**; this is the “binned $H$” that is compared to Monte Carlo **ground truth** in `study_h_decoding_convergence` (see also `fisher/hellinger_gt.py` and the binned-H / pairwise decoding documentation in the script header).

Intuition: for each $x_i$, large $\Delta L$ between $\theta$ values means the model assigns very different mass to those conditions, in line with a large Hellinger distance in $\theta$ for that $x$.

## How to run (study script)

From the repository root, use the `geo_diffusion` environment and the study’s **theta-field** flag:

- `--theta-field-method x_flow`

together with the usual dataset, binning, and flow training flags documented in `study_h_decoding_convergence.py` (e.g. `--flow-epochs`, `--flow-arch`, `--device cuda` per project `AGENTS.md`).

## Relation to other field methods (one line each)

- **`theta_flow`:** posterior/prior **$\theta$**-space flows; matrix built from $\log p(\theta\mid x)$ contrasts via ODE likelihoods (requires a prior model).
- **`theta_path_integral`:** same FM training as `theta_flow` in $\theta$ space, but H built from a **path integral** of scores along $\theta$ rather than direct flow likelihoods.
- **`x_flow`:** direct **$p(x\mid\theta)$** only; no prior; `field_method="flow_x_likelihood"` in `HMatrixEstimator.run`.

## Code map

| Piece | Location |
| --- | --- |
| CLI and sweep | `bin/study_h_decoding_convergence.py` (docstring + `--theta-field-method`) |
| FM training for $v_t(x_t;\theta)$ | `fisher/trainers.py` — `train_conditional_x_flow_model` |
| $C_{ij}=\log p(x_i\mid\theta_j)$, $\Delta L$, $H$ | `fisher/h_matrix.py` — `HMatrixEstimator` with `field_method="flow_x_likelihood"` |
| Study wires model + estimator | `fisher/shared_fisher_est.py` (trains `ConditionalXFlowVelocity*`, calls estimator with `field_method="flow_x_likelihood"`) |
