# Toy Experiment Plan: Flow Matching with an Analytical Diagonal Gaussian Velocity Prior

## 0. Purpose

We want to evaluate whether an analytical diagonal Gaussian velocity prior improves flow matching when the training data are limited and noisy.

The target distribution will be deliberately chosen to be:

1. close to Gaussian in its first-order and marginal second-order statistics;
2. mildly non-Gaussian;
3. correlated across dimensions;
4. analytically tractable enough that we can evaluate true density or at least true samples.

We compare two methods:

- **Baseline flow matching**: ordinary flow matching with no prior regularization.
- **Regularized flow matching**: ordinary flow matching plus the analytical diagonal Gaussian velocity-prior regularizer.

The prior is estimated from the training data using only empirical mean and empirical diagonal variance.

---

## 1. Main question

Given sparse samples from a mildly non-Gaussian correlated distribution, does adding the analytical diagonal Gaussian velocity prior:

1. improve sample quality?
2. improve held-out likelihood?
3. reduce overfitting?
4. preserve true non-Gaussian and correlated structure rather than collapsing to a Gaussian?

The desired outcome is not that the regularized model becomes Gaussian. The desired outcome is that the prior stabilizes training while still allowing the model to learn non-Gaussian corrections from data.

---

## 2. Synthetic data distribution

### 2.1 Two-dimensional correlated banana distribution

Start with a correlated Gaussian latent variable

$$
Z=
\begin{pmatrix}
Z_1\\
Z_2
\end{pmatrix}
\sim
\mathcal N(0,C),
$$

where

$$
C=
\begin{pmatrix}
1 & \rho\\
\rho & 1
\end{pmatrix}.
$$

Use, for example,

$$
\rho=0.7.
$$

Now define the observed data

$$
X_1=Z_1,
$$

$$
X_2=Z_2+\beta(Z_1^2-1).
$$

The parameter $\beta$ controls the strength of non-Gaussianity. A good starting value is

$$
\beta=0.3.
$$

The subtraction of $1$ keeps the nonlinear term approximately mean-zero because

$$
\mathbb E[Z_1^2]=1.
$$

Therefore, the distribution remains centered near zero, but it becomes non-Gaussian and curved.

This gives a useful toy distribution because:

- it has correlated noise through $\rho$;
- it is not Gaussian due to the quadratic deformation;
- it is still close enough to Gaussian that a Gaussian prior is not absurd;
- the transformation is invertible with unit Jacobian, so the true density can be computed analytically.

---

### 2.2 True density

The transformation is

$$
x_1=z_1,
$$

$$
x_2=z_2+\beta(z_1^2-1).
$$

The inverse is

$$
z_1=x_1,
$$

$$
z_2=x_2-\beta(x_1^2-1).
$$

The Jacobian determinant is

$$
\left|
\det
\frac{\partial x}{\partial z}
\right|
=1.
$$

Therefore,

$$
p_X(x)=p_Z(z(x)).
$$

So the true log density is

$$
\log p_X(x)
=
\log \mathcal N
\left(
\begin{pmatrix}
x_1\\
x_2-\beta(x_1^2-1)
\end{pmatrix}
;
0,C
\right).
$$

This is useful for evaluating likelihood accuracy.

---

### 2.3 Optional higher-dimensional extension

After the two-dimensional experiment works, extend to $d>2$.

One option is:

$$
Z\sim \mathcal N(0,C_d),
$$

where $C_d$ has correlated structure, for example

$$
[C_d]_{ij}=\rho^{|i-j|}.
$$

Then apply a nonlinear deformation only to the first two coordinates:

$$
X_1=Z_1,
$$

$$
X_2=Z_2+\beta(Z_1^2-1),
$$

and for $k\ge 3$,

$$
X_k=Z_k.
$$

This gives a high-dimensional distribution with a localized non-Gaussian component and correlated background noise.

---

## 3. Train, validation, and test sets

Use several sample-size regimes.

Recommended settings:

$$
N_{\mathrm{train}}\in\{128,512,2048\}.
$$

Use a large validation and test set, for example

$$
N_{\mathrm{val}}=10{,}000,
$$

$$
N_{\mathrm{test}}=10{,}000.
$$

The small-sample regime is especially important. The prior should be most useful when $N_{\mathrm{train}}$ is small.

Use multiple random seeds, for example

$$
5
$$

or

$$
10
$$

seeds per setting.

---

## 4. Prior estimation

For each training set, estimate the diagonal Gaussian prior using only the training samples.

Let the training samples be

$$
x^{(1)},\dots,x^{(N)}.
$$

Estimate the prior mean by

$$
\hat\mu
=
\frac{1}{N}
\sum_{n=1}^N x^{(n)}.
$$

Estimate the diagonal variance by

$$
\hat s_i
=
\frac{1}{N-1}
\sum_{n=1}^N
\left(
x_i^{(n)}-\hat\mu_i
\right)^2.
$$

Then the prior is

$$
p_G(x)
=
\mathcal N
\left(
\hat\mu,
\operatorname{diag}(\hat s)
\right).
$$

Important: this prior intentionally ignores off-diagonal covariance.

This is useful because the ground-truth data have correlated noise, while the prior only contains marginal variances. Therefore, if the regularized model learns the correlation well, it means the prior is not simply forcing the model to become diagonal Gaussian.

Use a variance floor for numerical stability:

$$
\hat s_i\leftarrow \max(\hat s_i,s_{\min}),
$$

with for example

$$
s_{\min}=10^{-6}.
$$

---

## 5. Flow matching setup

### 5.1 Base distribution

Use the standard Gaussian base distribution

$$
X_0\sim \mathcal N(0,I).
$$

The model learns a time-dependent velocity field

$$
v_\theta(x,t).
$$

During sampling, we solve the ODE

$$
\frac{dX_t}{dt}=v_\theta(X_t,t)
$$

from $t=0$ to $t=1$, starting from

$$
X_0\sim \mathcal N(0,I).
$$

---

### 5.2 Use the `flow_matching` package

Use the official `flow_matching` package for the standard flow matching component.

The package convention for affine paths is

$$
X_t=\alpha_t X_1+\sigma_t X_0,
$$

where $X_0$ is the source noise and $X_1$ is the target data.

This corresponds to the notation from the previous note by setting

$$
a_t=\sigma_t,
$$

$$
b_t=\alpha_t.
$$

Use `AffineProbPath` with an affine scheduler.

For the first experiment, use the conditional optimal transport scheduler, which corresponds to the simple linear path

$$
\alpha_t=t,
$$

$$
\sigma_t=1-t.
$$

Then

$$
\dot\alpha_t=1,
$$

$$
\dot\sigma_t=-1.
$$

Later, repeat the experiment with another scheduler, such as a cosine or VP-style scheduler, to test whether the prior still helps under a different affine path.

---

## 6. Baseline method: ordinary flow matching

For the baseline, use the standard flow matching loss:

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{t,X_0,X_1}
\left[
\left\|
v_\theta(X_t,t)-U_t
\right\|^2
\right],
$$

where

$$
X_t=\alpha_tX_1+\sigma_tX_0,
$$

and

$$
U_t
=
\dot\alpha_tX_1+\dot\sigma_tX_0.
$$

In the `flow_matching` package, the path object returns both $X_t$ and its target velocity.

For the baseline,

$$
\lambda_0=0.
$$

---

## 7. Regularized method: analytical diagonal Gaussian velocity prior

For the regularized model, use the same standard flow matching loss, plus the analytical Gaussian velocity-prior loss.

The prior endpoint distribution is

$$
X_1^G\sim
\mathcal N
\left(
\hat\mu,
\operatorname{diag}(\hat s)
\right).
$$

Under the affine path,

$$
X_t^G=\alpha_tX_1^G+\sigma_tX_0.
$$

The marginal distribution of $X_t^G$ is

$$
X_t^G
\sim
\mathcal N
\left(
\alpha_t\hat\mu,
\sigma_t^2I+\alpha_t^2\operatorname{diag}(\hat s)
\right).
$$

The analytical Gaussian-prior velocity field is

$$
v_G(x,t)
=
\dot\alpha_t\hat\mu
+
\frac{
\sigma_t\dot\sigma_t+\alpha_t\dot\alpha_t\hat s
}{
\sigma_t^2+\alpha_t^2\hat s
}
\odot
(x-\alpha_t\hat\mu).
$$

The prior loss is

$$
\mathcal L_{\mathrm{prior}}(\theta)
=
\mathbb E_{t,X_t^G}
\left[
\left\|
v_\theta(X_t^G,t)-v_G(X_t^G,t)
\right\|^2
\right].
$$

The total loss is

$$
\mathcal L(\theta)
=
\mathcal L_{\mathrm{FM}}(\theta)
+
\lambda_0
\mathcal L_{\mathrm{prior}}(\theta).
$$

For each training run, $\lambda_0$ is fixed.

Recommended main comparison:

$$
\lambda_0=0
$$

versus

$$
\lambda_0=0.1.
$$

Optional ablation:

$$
\lambda_0\in\{0.001,0.01,0.1,1.0\}.
$$

Each value should be treated as a separate fixed-objective training run, not as a time-varying schedule.

---

## 8. Model architecture

Use the same architecture for the baseline and regularized models.

A reasonable first model is a small MLP:

$$
(x,t)\mapsto v_\theta(x,t).
$$

Recommended architecture:

- input: $x$ concatenated with a time embedding of $t$;
- time embedding: sinusoidal or Fourier features;
- hidden layers: 3 to 5 layers;
- hidden width: 128 or 256;
- activation: SiLU or GELU;
- output dimension: $d$.

Keep the architecture intentionally modest. The goal is to evaluate whether the prior improves estimation, not whether a huge model can memorize the dataset.

---

## 9. Training protocol

Use identical training settings for the baseline and regularized model.

Recommended settings:

- optimizer: AdamW;
- learning rate: $10^{-3}$ or $3\times 10^{-4}$;
- batch size: 256;
- training steps: 20,000 to 100,000 depending on dataset size;
- time sampling: $t\sim \mathrm{Uniform}(\epsilon,1-\epsilon)$;
- default $\epsilon=10^{-4}$;
- loss norm: mean squared error per dimension.

Use the same random seed for corresponding baseline and regularized runs.

For example, for seed $k$:

1. generate one training set;
2. estimate $\hat\mu$ and $\hat s$ from that training set;
3. train the baseline model;
4. train the regularized model;
5. compare both models on the same validation and test sets.

This paired-seed design reduces variance in the comparison.

---

## 10. Evaluation metrics

### 10.1 Visual sample quality

For the 2D experiment, generate samples from each trained model and plot:

1. true data samples;
2. baseline generated samples;
3. regularized generated samples.

Recommended plots:

- scatter plots;
- kernel density contours;
- learned sample contours over true density contours;
- vector field visualization at selected time points.

The regularized model should look smoother in small-data regimes but should still preserve the banana-shaped non-Gaussian structure.

---

### 10.2 Held-out likelihood

Use the ODE likelihood computation to estimate

$$
\log p_\theta(x)
$$

on held-out test samples.

Report

$$
\mathbb E_{x\sim p_{\mathrm{test}}}
[
\log p_\theta(x)
].
$$

Since the banana distribution has known true density, also compute

$$
\mathbb E_{x\sim p_{\mathrm{test}}}
[
\log p_{\mathrm{true}}(x)
].
$$

The true value gives an upper reference point. The learned model will usually be below it.

Compare:

$$
\Delta_{\mathrm{NLL}}
=
-\mathbb E[\log p_\theta(x)]
+
\mathbb E[\log p_{\mathrm{true}}(x)].
$$

Lower $\Delta_{\mathrm{NLL}}$ is better.

---

### 10.3 Distribution distance between generated and true samples

Generate a large set of model samples:

$$
\hat x^{(1)},\dots,\hat x^{(M)}\sim p_\theta.
$$

Compare them to true test samples using:

1. sliced Wasserstein distance;
2. MMD with RBF kernel;
3. energy distance.

These metrics do not require model likelihoods and are often more stable than ODE likelihood estimates.

---

### 10.4 Moment and correlation recovery

Compare generated samples to true samples using:

Mean error:

$$
\|\hat m-m_{\mathrm{true}}\|_2.
$$

Diagonal variance error:

$$
\|\operatorname{diag}(\hat C)-\operatorname{diag}(C_{\mathrm{true}})\|_2.
$$

Full covariance error:

$$
\|\hat C-C_{\mathrm{true}}\|_F.
$$

Correlation error:

$$
|\hat\rho-\rho_{\mathrm{true}}|.
$$

This is important because the prior only contains diagonal covariance. The model should still recover off-diagonal correlation from data.

---

### 10.5 Non-Gaussian structure recovery

For the banana distribution, the key non-Gaussian structure is

$$
X_2
\approx
Z_2+\beta(X_1^2-1).
$$

Estimate the quadratic relationship by regressing generated samples using

$$
x_2
=
c_0+c_1x_1+c_2(x_1^2-1)+\eta.
$$

Compare the estimated coefficient $\hat c_2$ to the true value $\beta$.

Report

$$
|\hat c_2-\beta|.
$$

This metric directly tests whether the regularized model preserves the non-Gaussian curvature rather than collapsing toward a simple Gaussian.

---

### 10.6 Overfitting diagnostics

Compare train and test likelihoods:

$$
\mathbb E_{x\sim p_{\mathrm{train}}}[\log p_\theta(x)]
-
\mathbb E_{x\sim p_{\mathrm{test}}}[\log p_\theta(x)].
$$

A large gap indicates overfitting.

The hypothesis is that the regularized model should reduce this gap, especially for small $N_{\mathrm{train}}$.

---

## 11. Main experimental table

For each $N_{\mathrm{train}}$, report:

| Method | $N_{\mathrm{train}}$ | Test NLL | Sliced Wasserstein | MMD | Mean Error | Cov Error | Corr Error | Banana Coef Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FM baseline | 128 | | | | | | | |
| FM + prior | 128 | | | | | | | |
| FM baseline | 512 | | | | | | | |
| FM + prior | 512 | | | | | | | |
| FM baseline | 2048 | | | | | | | |
| FM + prior | 2048 | | | | | | | |

Report mean and standard error across random seeds.

---

## 12. Suggested figures

### Figure 1: Data distribution

Show true samples from the banana distribution.

Panels:

1. latent correlated Gaussian $Z$;
2. transformed observed data $X$;
3. diagonal Gaussian prior fit to training data.

This makes clear that the prior is useful but misspecified.

---

### Figure 2: Generated samples

Compare generated samples:

1. true data;
2. baseline flow matching;
3. regularized flow matching.

Show results for small $N_{\mathrm{train}}$, for example $N=128$.

---

### Figure 3: Density contours

Overlay learned density contours with true density contours.

This is useful in 2D because the true density is known.

---

### Figure 4: Metric versus sample size

Plot each evaluation metric as a function of $N_{\mathrm{train}}$.

Expected pattern:

- regularization helps most at small $N$;
- the gap shrinks as $N$ increases;
- if $\lambda_0$ is too large, non-Gaussian recovery becomes worse.

---

### Figure 5: Lambda ablation

Plot performance versus fixed $\lambda_0$.

Use

$$
\lambda_0\in\{0,0.001,0.01,0.1,1.0\}.
$$

This tests the bias-variance tradeoff.

---

## 13. Expected outcomes

### Expected benefit

The regularized model should improve:

1. held-out likelihood;
2. sample smoothness;
3. covariance estimation stability;
4. distribution distances;
5. overfitting gap.

This should be most visible for small training sets.

---

### Expected risk

If $\lambda_0$ is too large, the model may become too close to the diagonal Gaussian prior.

Symptoms:

1. reduced banana curvature;
2. underestimated correlation;
3. worse MMD or sliced Wasserstein despite better mean and variance;
4. generated samples look too elliptical.

Therefore, evaluation must include both Gaussian-like statistics and non-Gaussian structure metrics.

---

## 14. Important controls

### Control 1: Same architecture and optimizer

The baseline and regularized models must use the same architecture, optimizer, batch size, and training steps.

---

### Control 2: Same train/test split

For each random seed, train both methods on the same dataset.

---

### Control 3: Same affine path

Use the same affine probability path for both methods.

---

### Control 4: Same prior estimate

The prior estimate must be computed only from the training set.

Do not estimate $\hat\mu$ or $\hat s$ using validation or test data.

---

### Control 5: No full covariance prior

The main experiment should use a diagonal Gaussian prior only.

This makes the experiment stricter because the true data have correlated noise. The model must learn correlation from data, not from the prior.

---

## 15. Possible extensions

### Extension 1: Full covariance analytical Gaussian prior

After the diagonal prior experiment, compare against a full covariance prior:

$$
p_G(x)=\mathcal N(\hat\mu,\hat C).
$$

This tests whether including correlation in the prior helps or over-regularizes.

---

### Extension 2: Misspecified prior

Manually perturb the prior:

$$
\mu_{\mathrm{prior}}=\hat\mu+\delta_\mu,
$$

or

$$
s_{\mathrm{prior}}=c\hat s.
$$

This tests robustness to prior misspecification.

---

### Extension 3: Stronger non-Gaussianity

Increase $\beta$:

$$
\beta\in\{0.1,0.3,0.6,1.0\}.
$$

The prior should help when the distribution is close to Gaussian, but may hurt when the true distribution is far from Gaussian.

---

### Extension 4: Stronger correlation

Vary

$$
\rho\in\{0,0.3,0.7,0.9\}.
$$

This tests whether a diagonal prior interferes with learning off-diagonal covariance.

---

## 16. Minimal experiment checklist

1. Generate banana-distribution data.
2. Split into train, validation, and test sets.
3. Estimate $\hat\mu$ and $\hat s$ from the training set.
4. Train baseline flow matching with $\lambda_0=0$.
5. Train regularized flow matching with fixed $\lambda_0>0$.
6. Generate samples from both models using the same ODE solver.
7. Compute held-out likelihood using the ODE likelihood method.
8. Compute sample-based distribution distances.
9. Compute mean, variance, covariance, correlation, and banana coefficient errors.
10. Repeat across multiple seeds and sample sizes.
11. Report mean and standard error.
12. Visualize samples and density contours.

---

## 17. Success criterion

The method is successful if, in the small-data regime, the regularized model improves held-out distribution estimation while preserving the true non-Gaussian and correlated structure.

More specifically, compared with the baseline, the regularized model should have:

1. lower test NLL;
2. lower sliced Wasserstein or MMD;
3. smaller train-test likelihood gap;
4. comparable or better correlation recovery;
5. comparable or better banana coefficient recovery.

If the regularized model only improves mean and variance but destroys correlation or banana curvature, then the prior is too strong or too misspecified.
