# Conditional Flow Matching with a KNN-Kernel Gaussian Velocity Prior

## 1. Goal

We have paired data

$$
\{(x_n,\theta_n)\}_{n=1}^N,
$$

where

$$
x_n\in\mathbb R^d,
\qquad
\theta_n\in\mathbb R^q.
$$

The goal is to learn the conditional distribution

$$
p(x\mid\theta)
$$

using conditional flow matching.

The baseline model learns a conditional velocity field

$$
v_\phi(x,t,\theta).
$$

The regularized model also learns

$$
v_\phi(x,t,\theta),
$$

but it is encouraged to stay close to a local diagonal Gaussian prior

$$
p_G(x\mid\theta)
=
\mathcal N(\mu(\theta),\operatorname{diag}(s(\theta))).
$$

The key idea is:

1. estimate the local conditional mean $\mu(\theta)$ and diagonal variance $s(\theta)$ using KNN-kernel smoothing;
2. construct the analytical Gaussian-prior velocity field $v_G(x,t,\theta)$;
3. regularize the learned conditional velocity field toward $v_G(x,t,\theta)$.

This lets the prior stabilize training while still allowing the learned model to capture correlation and non-Gaussian structure from data.

---

## 2. Toy dataset

We construct a synthetic conditional distribution

$$
p(x\mid\theta)
$$

that is close to Gaussian but not exactly Gaussian.

The condition variable is one-dimensional and periodic:

$$
\theta\in[0,2\pi).
$$

For each sample, draw

$$
\theta\sim\mathrm{Uniform}(0,2\pi).
$$

The observation is two-dimensional:

$$
x\in\mathbb R^2.
$$

---

## 3. Conditional mean

Let the conditional mean move around a circle:

$$
m(\theta)
=
\begin{pmatrix}
2\cos\theta\\
2\sin\theta
\end{pmatrix}.
$$

This makes the conditional distribution change smoothly with $\theta$.

---

## 4. Conditional correlated Gaussian noise

Generate a latent Gaussian noise variable

$$
z\mid\theta\sim\mathcal N(0,C(\theta)).
$$

Let

$$
C(\theta)
=
\begin{pmatrix}
\sigma_1^2(\theta)
&
\rho\sigma_1(\theta)\sigma_2(\theta)
\\
\rho\sigma_1(\theta)\sigma_2(\theta)
&
\sigma_2^2(\theta)
\end{pmatrix}.
$$

Use, for example,

$$
\sigma_1(\theta)=0.4+0.2\sin\theta,
$$

$$
\sigma_2(\theta)=0.3+0.1\cos\theta,
$$

and

$$
\rho=0.6.
$$

This gives $\theta$-dependent variance and correlated noise.

---

## 5. Conditional non-Gaussian deformation

Define the observed data by

$$
x_1
=
m_1(\theta)+z_1,
$$

$$
x_2
=
m_2(\theta)+z_2+\beta\left(z_1^2-\sigma_1^2(\theta)\right).
$$

Use, for example,

$$
\beta=0.4.
$$

The term

$$
z_1^2-\sigma_1^2(\theta)
$$

has approximately zero mean conditional on $\theta$, so it introduces curvature without strongly shifting the conditional mean.

Therefore, the true conditional distribution has:

1. $\theta$-dependent mean;
2. $\theta$-dependent marginal variance;
3. correlated noise;
4. mild non-Gaussian banana-shaped curvature.

This is useful because the diagonal Gaussian prior is helpful but misspecified. It can capture local mean and marginal variance, but it cannot directly capture correlation or non-Gaussian curvature.

---

## 6. Dataset splits

Use several training sample sizes:

$$
N_{\mathrm{train}}\in\{128,512,2048\}.
$$

Use larger validation and test sets:

$$
N_{\mathrm{val}}=5000,
$$

$$
N_{\mathrm{test}}=5000.
$$

Repeat experiments over multiple random seeds, for example $5$ or $10$ seeds.

The small-data regime is especially important because the prior should help most when conditional samples are sparse.

---

## 7. KNN-kernel estimate of the conditional Gaussian prior

For any query condition $\theta^*$, estimate

$$
p_G(x\mid\theta^*)
=
\mathcal N
\left(
\hat\mu(\theta^*),
\operatorname{diag}(\hat s(\theta^*))
\right).
$$

The prior is estimated only from the training data.

---

### 7.1 K nearest neighbors in condition space

Find the $K$ nearest training conditions to $\theta^*$:

$$
\mathcal N_K(\theta^*)
=
\{n_1,\dots,n_K\}.
$$

Because $\theta$ is periodic, use circular distance:

$$
d_{\mathrm{circ}}(\theta_n,\theta^*)
=
\min
\left(
|\theta_n-\theta^*|,
2\pi-|\theta_n-\theta^*|
\right).
$$

Use, for example,

$$
K=64
$$

or

$$
K=128.
$$

---

### 7.2 Kernel weights

For each neighbor, assign a Gaussian kernel weight:

$$
w_n(\theta^*)
=
\exp
\left(
-\frac{
d_{\mathrm{circ}}(\theta_n,\theta^*)^2
}{
2h(\theta^*)^2
}
\right),
\qquad
n\in\mathcal N_K(\theta^*).
$$

Use an adaptive bandwidth:

$$
h(\theta^*)=d_K(\theta^*),
$$

where $d_K(\theta^*)$ is the distance from $\theta^*$ to its $K$-th nearest neighbor.

Normalize the weights:

$$
\bar w_n(\theta^*)
=
\frac{
w_n(\theta^*)
}{
\sum_{m\in\mathcal N_K(\theta^*)}w_m(\theta^*)
}.
$$

Then

$$
\sum_{n\in\mathcal N_K(\theta^*)}
\bar w_n(\theta^*)=1.
$$

---

### 7.3 Conditional mean estimate

Estimate the local conditional mean by

$$
\hat\mu(\theta^*)
=
\sum_{n\in\mathcal N_K(\theta^*)}
\bar w_n(\theta^*)x_n.
$$

---

### 7.4 Conditional diagonal variance estimate

Estimate the local diagonal variance by

$$
\hat s(\theta^*)
=
\sum_{n\in\mathcal N_K(\theta^*)}
\bar w_n(\theta^*)
\left(
x_n-\hat\mu(\theta^*)
\right)^2.
$$

The square is elementwise.

If $K$ is small, use the weighted variance correction:

$$
\hat s_{\mathrm{corrected}}(\theta^*)
=
\frac{
\sum_{n\in\mathcal N_K(\theta^*)}
\bar w_n(\theta^*)
\left(
x_n-\hat\mu(\theta^*)
\right)^2
}{
1-\sum_{n\in\mathcal N_K(\theta^*)}\bar w_n(\theta^*)^2
}.
$$

Then apply a variance floor:

$$
\hat s_i(\theta^*)\leftarrow
\max
\left(
\hat s_i(\theta^*),
s_{\min}
\right).
$$

For example,

$$
s_{\min}=10^{-6}.
$$

---

## 8. Conditional flow matching baseline

Use base distribution

$$
x_0\sim\mathcal N(0,I).
$$

For a training pair $(x_1,\theta)$, define an affine probability path

$$
x_t=\alpha_t x_1+\sigma_t x_0.
$$

The target velocity is

$$
u_t=\dot\alpha_t x_1+\dot\sigma_t x_0.
$$

The velocity model takes $(x_t,t,\theta)$ as input:

$$
v_\phi(x_t,t,\theta).
$$

The baseline conditional flow matching loss is

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E_{x_1,\theta,x_0,t}
\left[
\left\|
v_\phi(x_t,t,\theta)-u_t
\right\|^2
\right].
$$

This model has no prior regularization.

---

## 9. Analytical conditional Gaussian velocity prior

For a given $\theta$, the KNN-kernel prior is

$$
p_G(x\mid\theta)
=
\mathcal N
\left(
\hat\mu(\theta),
\operatorname{diag}(\hat s(\theta))
\right).
$$

Under the affine path,

$$
x_t^G=\alpha_t x_1^G+\sigma_t x_0,
$$

where

$$
x_1^G\mid\theta
\sim
\mathcal N
\left(
\hat\mu(\theta),
\operatorname{diag}(\hat s(\theta))
\right).
$$

Therefore,

$$
x_t^G\mid\theta
\sim
\mathcal N
\left(
\alpha_t\hat\mu(\theta),
\sigma_t^2I+\alpha_t^2\operatorname{diag}(\hat s(\theta))
\right).
$$

The analytical Gaussian-prior velocity field is

$$
\boxed{
v_G(x,t,\theta)
=
\dot\alpha_t\hat\mu(\theta)
+
\frac{
\sigma_t\dot\sigma_t+\alpha_t\dot\alpha_t\hat s(\theta)
}{
\sigma_t^2+\alpha_t^2\hat s(\theta)
}
\odot
\left(
x-\alpha_t\hat\mu(\theta)
\right)
}
$$

where all operations involving $\hat s(\theta)$ are elementwise.

This is the conditional mean velocity field induced by the local diagonal Gaussian prior.

---

## 10. Prior regularization loss

The prior regularization loss is

$$
\mathcal L_{\mathrm{prior}}
=
\mathbb E_{\theta,t,x_t^G}
\left[
\left\|
v_\phi(x_t^G,t,\theta)
-
v_G(x_t^G,t,\theta)
\right\|^2
\right].
$$

Here

$$
x_t^G\mid\theta
\sim
\mathcal N
\left(
\alpha_t\hat\mu(\theta),
\sigma_t^2I+\alpha_t^2\operatorname{diag}(\hat s(\theta))
\right).
$$

The prior loss always uses the analytical velocity field $v_G$, not noisy endpoint velocities.

---

## 11. Full training objective

The regularized conditional flow matching loss is

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{FM}}
+
\lambda_0\mathcal L_{\mathrm{prior}}.
}
$$

Here $\lambda_0$ is fixed during training.

The baseline corresponds to

$$
\lambda_0=0.
$$

For the main regularized experiment, use for example

$$
\lambda_0=0.1.
$$

For an ablation, test

$$
\lambda_0\in\{0,0.001,0.01,0.1,1.0\}.
$$

Each $\lambda_0$ should be treated as a separate fixed-objective training run.

---

## 12. Sampling $\theta$ for the prior loss

The simplest choice is to use $\theta$ values from the training minibatch:

$$
\theta\sim p_{\mathrm{train}}(\theta).
$$

This regularizes the model in regions where data exist.

An optional extension is to sample

$$
\theta\sim\mathrm{Uniform}(0,2\pi)
$$

and then estimate $\hat\mu(\theta)$ and $\hat s(\theta)$ by KNN-kernel smoothing.

For the first experiment, use training-minibatch $\theta$ values because this avoids unreliable priors in regions with little data.

---

## 13. Methods to compare

Use the same velocity-network architecture and training settings for all methods.

### Method A: Baseline conditional flow matching

$$
\mathcal L
=
\mathcal L_{\mathrm{FM}}.
$$

### Method B: Conditional flow matching with KNN Gaussian velocity prior

$$
\mathcal L
=
\mathcal L_{\mathrm{FM}}
+
\lambda_0\mathcal L_{\mathrm{prior}}.
$$

### Optional Method C: Conditional flow matching with global Gaussian prior

Estimate one global diagonal Gaussian prior:

$$
p_G(x)
=
\mathcal N(\hat\mu,\operatorname{diag}(\hat s)).
$$

This ignores $\theta$.

This baseline checks whether the conditional KNN prior is actually useful.

### Optional Method D: Conditional flow matching with oracle Gaussian prior

Use the true conditional mean and true conditional marginal variance from the synthetic generator.

This is an oracle control. It tells us the best-case effect of this type of prior.

---

## 14. Evaluation conditions

Evaluate the learned conditional distribution at fixed test conditions, for example:

$$
\theta^*
\in
\left\{
0,
\frac{\pi}{4},
\frac{\pi}{2},
\pi,
\frac{3\pi}{2}
\right\}.
$$

For each $\theta^*$, generate samples from the learned conditional model:

$$
x\sim p_\phi(x\mid\theta^*).
$$

Compare these samples against true simulator samples:

$$
x\sim p_{\mathrm{true}}(x\mid\theta^*).
$$

---

## 15. Evaluation metrics

### 15.1 Conditional sample visualization

For each fixed $\theta^*$, plot:

1. true samples;
2. baseline generated samples;
3. prior-regularized generated samples;
4. local KNN diagonal Gaussian prior ellipse.

This will show whether the learned model captures correlation and non-Gaussian curvature beyond the diagonal Gaussian prior.

---

### 15.2 Conditional mean error

Compute

$$
\left\|
\hat m_\phi(\theta^*)-m_{\mathrm{true}}(\theta^*)
\right\|_2.
$$

Here $\hat m_\phi(\theta^*)$ is the sample mean of generated samples.

---

### 15.3 Conditional covariance error

Compute the generated covariance

$$
\hat C_\phi(\theta^*).
$$

Compare it to the true conditional covariance:

$$
\left\|
\hat C_\phi(\theta^*)-C_{\mathrm{true}}(\theta^*)
\right\|_F.
$$

This is important because the prior is diagonal, while the true distribution has correlated noise.

---

### 15.4 Conditional correlation recovery

Compute the generated correlation

$$
\hat\rho_\phi(\theta^*).
$$

Compare it against the true correlation:

$$
\left|
\hat\rho_\phi(\theta^*)-\rho_{\mathrm{true}}(\theta^*)
\right|.
$$

The regularized model should still learn correlation from the data.

---

### 15.5 Non-Gaussian curvature recovery

At each fixed $\theta^*$, regress generated samples using

$$
x_2
=
c_0+c_1x_1+c_2(x_1^2-\bar v_1)+\eta.
$$

The coefficient $c_2$ measures banana curvature.

Compare

$$
|\hat c_2-\beta|.
$$

This checks whether the Gaussian prior destroys non-Gaussian structure.

---

### 15.6 Sample-based distribution distance

For each $\theta^*$, compute a sample-based distance between generated and true conditional samples:

1. maximum mean discrepancy;
2. sliced Wasserstein distance;
3. energy distance.

Then average across test conditions:

$$
\frac{1}{M}
\sum_{j=1}^M
D
\left(
p_\phi(x\mid\theta_j^*),
p_{\mathrm{true}}(x\mid\theta_j^*)
\right).
$$

---

## 16. Expected outcomes

The KNN-prior regularized model should help most when $N_{\mathrm{train}}$ is small.

Expected pattern:

1. baseline conditional flow matching may overfit local samples;
2. baseline may produce unstable conditional distributions in sparse regions;
3. KNN-prior regularized flow matching should recover smoother conditional samples;
4. regularization should improve conditional mean and marginal variance estimation;
5. if $\lambda_0$ is not too large, the model should still learn correlation and banana curvature from data.

The prior becomes harmful if it forces the model too close to the local diagonal Gaussian.

Symptoms of too strong prior:

1. generated samples become too elliptical;
2. conditional correlation is underestimated;
3. banana curvature is reduced;
4. sample-based distribution distances get worse even if mean and variance improve.

---

## 17. Main experimental table

For each training size, report mean and standard error across seeds.

| Method | $N_{\mathrm{train}}$ | Mean Error | Cov Error | Corr Error | Curvature Error | MMD | Sliced Wasserstein |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline CFM | 128 | | | | | | |
| CFM + KNN Prior | 128 | | | | | | |
| Baseline CFM | 512 | | | | | | |
| CFM + KNN Prior | 512 | | | | | | |
| Baseline CFM | 2048 | | | | | | |
| CFM + KNN Prior | 2048 | | | | | | |

---

## 18. Suggested figures

### Figure 1: Toy data generator

Show samples from several fixed $\theta$ values.

This illustrates how the conditional distribution moves with $\theta$.

### Figure 2: KNN Gaussian prior

For several $\theta^*$ values, show:

1. true conditional samples;
2. KNN diagonal Gaussian prior ellipse.

This confirms that the prior captures mean and marginal scale, but not correlation or curvature.

### Figure 3: Generated conditional samples

Compare:

1. true conditional samples;
2. baseline conditional flow matching;
3. prior-regularized conditional flow matching.

Use a small-data setting such as

$$
N_{\mathrm{train}}=128.
$$

### Figure 4: Performance versus sample size

Plot metrics as a function of $N_{\mathrm{train}}$.

The prior should help more in the small-data regime.

### Figure 5: Fixed $\lambda_0$ ablation

Compare

$$
\lambda_0\in\{0,0.001,0.01,0.1,1.0\}.
$$

This visualizes the bias-variance tradeoff.

---

## 19. Important controls

### Control 1: Same training set

For each seed, the baseline and regularized models should use the same training set.

### Control 2: Same architecture

Use the same velocity-network architecture across methods.

### Control 3: Same optimizer and training steps

Use the same optimizer, learning rate, batch size, and number of training steps.

### Control 4: Prior estimated only from training data

Do not use validation or test data to estimate $\hat\mu(\theta)$ or $\hat s(\theta)$.

### Control 5: Diagonal prior only

The main experiment should use a diagonal Gaussian prior, even though the true distribution has correlations.

This makes the experiment stricter: the model must learn correlation from the data, not from the prior.

---

## 20. Summary

The experiment tests whether a KNN-kernel local Gaussian prior can improve conditional flow matching.

The complete pipeline is:

1. Generate conditional toy data $(x,\theta)$ with $\theta$-dependent mean, variance, correlation, and non-Gaussian curvature.
2. Estimate local diagonal Gaussian prior

$$
p_G(x\mid\theta)
=
\mathcal N
\left(
\hat\mu(\theta),
\operatorname{diag}(\hat s(\theta))
\right)
$$

using KNN-kernel smoothing.

3. Train baseline conditional flow matching using

$$
\mathcal L_{\mathrm{FM}}.
$$

4. Train regularized conditional flow matching using

$$
\mathcal L_{\mathrm{FM}}
+
\lambda_0\mathcal L_{\mathrm{prior}}.
$$

5. Evaluate conditional samples at fixed $\theta^*$ values.
6. Measure conditional mean error, covariance error, correlation recovery, non-Gaussian curvature recovery, and distribution distance.
7. Check whether the KNN Gaussian prior stabilizes learning without destroying the correlated and non-Gaussian structure of the true conditional distribution.
