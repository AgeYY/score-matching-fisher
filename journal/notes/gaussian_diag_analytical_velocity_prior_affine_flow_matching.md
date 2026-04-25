# Flow Matching with an Analytical Diagonal Gaussian Velocity Prior under a General Affine Probability Path

## Goal

We want to learn a data distribution $p_{\mathrm{data}}(x)$ using flow matching, while incorporating a prior belief that the target distribution should be close to a diagonal Gaussian distribution

$$
p_G(x)=\mathcal N(\mu,\Sigma),
$$

where

$$
\Sigma=\operatorname{diag}(s_1,\dots,s_d),
\qquad
s_i=\sigma_i^2.
$$

The true data distribution does **not** need to be Gaussian. The Gaussian prior is used only as a soft regularizer on the learned velocity field.

The key point is that we do **not** train the prior term using noisy sampled endpoint velocities. Instead, we always use the **analytical Gaussian-prior velocity field**.

The final objective is

$$
\mathcal L(\theta)
=
\mathcal L_{\mathrm{FM}}(\theta)
+
\lambda_0
\mathcal L_{\mathrm{prior}}(\theta),
$$

where $\lambda_0$ is a fixed regularization coefficient.

---

## 1. General affine probability path

Let the base distribution be

$$
X_0\sim \mathcal N(0,I),
$$

and let the data endpoint be

$$
X_1\sim p_{\mathrm{data}}.
$$

We use a general affine probability path

$$
X_t=a_tX_0+b_tX_1,
\qquad
t\in[0,1],
$$

where $a_t$ and $b_t$ are differentiable scalar schedules.

Usually the boundary conditions are

$$
a_0=1,\qquad b_0=0,
$$

and

$$
a_1=0,\qquad b_1=1.
$$

The instantaneous velocity along this path is

$$
U_t
=
\frac{dX_t}{dt}
=
\dot a_t X_0+\dot b_t X_1.
$$

The standard flow matching loss is

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{t,X_0,X_1}
\left[
\left\|
v_\theta(X_t,t)-U_t
\right\|^2
\right].
$$

Equivalently,

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{t,X_0,X_1}
\left[
\left\|
v_\theta(a_tX_0+b_tX_1,t)
-
\left(\dot a_tX_0+\dot b_tX_1\right)
\right\|^2
\right].
$$

This term fits the empirical data distribution.

---

## 2. Diagonal Gaussian prior endpoint distribution

Now define the Gaussian prior endpoint distribution

$$
X_1^G\sim \mathcal N(\mu,\operatorname{diag}(s)).
$$

Under the same affine path,

$$
X_t^G=a_tX_0+b_tX_1^G.
$$

Since $X_0$ and $X_1^G$ are independent Gaussians, $X_t^G$ is also Gaussian.

Its mean is

$$
m_t^G
=
\mathbb E[X_t^G]
=
b_t\mu.
$$

Its covariance is

$$
C_t^G
=
\operatorname{Cov}(X_t^G)
=
a_t^2I+b_t^2\operatorname{diag}(s).
$$

Because the covariance is diagonal, the coordinate-wise variance is

$$
c_i(t)
=
a_t^2+b_t^2s_i.
$$

Therefore,

$$
X_t^G
\sim
\mathcal N
\left(
b_t\mu,
\operatorname{diag}(c_1(t),\dots,c_d(t))
\right).
$$

This distribution describes the region of $(x,t)$ space where the Gaussian-prior regularization is applied.

---

## 3. Analytical Gaussian-prior velocity field

The Gaussian-prior velocity field is defined as the conditional mean velocity

$$
v_G(x,t)
=
\mathbb E[U_t^G\mid X_t^G=x],
$$

where

$$
U_t^G
=
\frac{dX_t^G}{dt}
=
\dot a_tX_0+\dot b_tX_1^G.
$$

This is the exact velocity field that ordinary flow matching would recover if the target distribution were exactly the Gaussian prior.

Because $(U_t^G,X_t^G)$ are jointly Gaussian, the conditional expectation is linear:

$$
v_G(x,t)
=
\mathbb E[U_t^G]
+
\operatorname{Cov}(U_t^G,X_t^G)
\operatorname{Cov}(X_t^G)^{-1}
\left(x-\mathbb E[X_t^G]\right).
$$

We compute the terms one by one.

First,

$$
\mathbb E[U_t^G]
=
\dot b_t\mu.
$$

Second,

$$
\operatorname{Cov}(X_t^G)
=
a_t^2I+b_t^2\operatorname{diag}(s).
$$

Third,

$$
\operatorname{Cov}(U_t^G,X_t^G)
=
a_t\dot a_t I
+
b_t\dot b_t\operatorname{diag}(s).
$$

Therefore,

$$
v_G(x,t)
=
\dot b_t\mu
+
\left(
a_t\dot a_t I
+
b_t\dot b_t\operatorname{diag}(s)
\right)
\left(
a_t^2I
+
b_t^2\operatorname{diag}(s)
\right)^{-1}
(x-b_t\mu).
$$

Because the covariance is diagonal, this simplifies coordinate by coordinate to

$$
\boxed{
[v_G(x,t)]_i
=
\dot b_t\mu_i
+
\frac{
a_t\dot a_t+b_t\dot b_t s_i
}{
a_t^2+b_t^2s_i
}
\left(
x_i-b_t\mu_i
\right)
}
$$

for $i=1,\dots,d$.

In vectorized elementwise notation,

$$
\boxed{
v_G(x,t)
=
\dot b_t\mu
+
\frac{
a_t\dot a_t+b_t\dot b_t s
}{
a_t^2+b_t^2s
}
\odot
(x-b_t\mu)
}
$$

where all operations involving $s$ are elementwise.

---

## 4. Analytical velocity-prior regularization

The prior regularization term is defined by matching the model velocity to the analytical Gaussian-prior velocity field:

$$
\mathcal L_{\mathrm{prior}}(\theta)
=
\mathbb E_{t,\,X_t^G}
\left[
\left\|
v_\theta(X_t^G,t)-v_G(X_t^G,t)
\right\|^2
\right].
$$

Here

$$
X_t^G
\sim
\mathcal N
\left(
b_t\mu,
a_t^2I+b_t^2\operatorname{diag}(s)
\right).
$$

This is an **analytical velocity prior**, not a sampled endpoint-velocity prior.

In particular, the regularization target is always

$$
v_G(X_t^G,t),
$$

not

$$
U_t^G=\dot a_tX_0+\dot b_tX_1^G.
$$

The difference is important:

$$
v_G(x,t)
=
\mathbb E[U_t^G\mid X_t^G=x]
$$

is the conditional mean velocity, while $U_t^G$ is a noisy pathwise velocity. The analytical prior uses the lower-variance conditional mean field.

Thus, the regularizer says:

$$
v_\theta(x,t)
\approx
v_G(x,t)
$$

for points $(x,t)$ distributed according to the Gaussian-prior path.

---

## 5. Final regularized objective

The final objective is

$$
\boxed{
\mathcal L(\theta)
=
\mathcal L_{\mathrm{FM}}(\theta)
+
\lambda_0
\mathcal L_{\mathrm{prior}}(\theta)
}
$$

with

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{t,X_0,X_1}
\left[
\left\|
v_\theta(a_tX_0+b_tX_1,t)
-
(\dot a_tX_0+\dot b_tX_1)
\right\|^2
\right],
$$

and

$$
\mathcal L_{\mathrm{prior}}(\theta)
=
\mathbb E_{t,\,X_t^G}
\left[
\left\|
v_\theta(X_t^G,t)-v_G(X_t^G,t)
\right\|^2
\right].
$$

Here $\lambda_0$ is fixed throughout training.

A larger $\lambda_0$ imposes a stronger Gaussian prior.

A smaller $\lambda_0$ allows the model to rely more heavily on the empirical data.

---

## 6. MAP-style interpretation

Flow matching can be viewed as a regression problem for a velocity field.

The ordinary flow matching objective fits empirical velocity targets

$$
U_t=\dot a_tX_0+\dot b_tX_1.
$$

The Gaussian prior defines a reference function

$$
v_G(x,t).
$$

The regularized objective is analogous to a MAP estimator:

$$
\text{data-fitting loss}
+
\text{prior penalty on the velocity field}.
$$

In ridge regression, a parameter vector is encouraged to stay close to a prior mean. Here, the velocity function is encouraged to stay close to the analytical Gaussian-prior velocity field.

The prior belief is therefore

$$
v_\theta(x,t)\approx v_G(x,t),
$$

rather than

$$
p_\theta(x)\equiv p_G(x).
$$

So the learned target distribution can still be non-Gaussian if the data provide evidence for non-Gaussian structure.

---

## 7. Training procedure

### Step 1: Choose the affine path

Choose differentiable schedules $a_t$ and $b_t$ satisfying

$$
a_0=1,\quad b_0=0,\quad a_1=0,\quad b_1=1.
$$

The method does not require the path to be linear. Any differentiable affine path can be used as long as $\dot a_t$ and $\dot b_t$ are available.

---

### Step 2: Specify the diagonal Gaussian prior

Choose the prior mean and diagonal variance:

$$
\mu\in\mathbb R^d,
$$

$$
s=(s_1,\dots,s_d),
\qquad
s_i>0.
$$

This prior can come from domain knowledge, empirical estimates, or a combination of both.

---

### Step 3: Choose a fixed regularization coefficient

Choose a fixed value

$$
\lambda_0>0.
$$

This value is kept constant during training.

---

### Step 4: Compute the data flow matching loss

For each data minibatch, evaluate

$$
X_t=a_tX_0+b_tX_1,
$$

and

$$
U_t=\dot a_tX_0+\dot b_tX_1.
$$

The data loss is

$$
\mathcal L_{\mathrm{FM}}
=
\left\|
v_\theta(X_t,t)-U_t
\right\|^2.
$$

This is the only part of the objective that uses pathwise velocity targets.

---

### Step 5: Compute the analytical velocity-prior loss

For the prior term, evaluate points $X_t^G$ from the Gaussian-prior path distribution

$$
X_t^G
\sim
\mathcal N
\left(
b_t\mu,
a_t^2I+b_t^2\operatorname{diag}(s)
\right).
$$

Then compute the analytical Gaussian-prior velocity

$$
v_G(X_t^G,t)
=
\dot b_t\mu
+
\frac{
a_t\dot a_t+b_t\dot b_t s
}{
a_t^2+b_t^2s
}
\odot
(X_t^G-b_t\mu).
$$

The prior loss is

$$
\mathcal L_{\mathrm{prior}}
=
\left\|
v_\theta(X_t^G,t)-v_G(X_t^G,t)
\right\|^2.
$$

The prior loss always uses the analytical velocity field $v_G$, not noisy sampled endpoint velocities.

---

### Step 6: Optimize the total loss

Update model parameters using

$$
\mathcal L(\theta)
=
\mathcal L_{\mathrm{FM}}(\theta)
+
\lambda_0\mathcal L_{\mathrm{prior}}(\theta).
$$

This trains the model to fit the real data while remaining close to the analytical velocity field implied by the diagonal Gaussian prior.

---

## 8. Loss normalization

In high dimensions, it is often useful to normalize squared norms by the dimension $d$:

$$
\|z\|^2_{\mathrm{mean}}
=
\frac{1}{d}\sum_{i=1}^d z_i^2.
$$

Then the objective becomes

$$
\mathcal L(\theta)
=
\mathbb E
\left[
\frac{1}{d}
\left\|
v_\theta(X_t,t)-U_t
\right\|^2
\right]
+
\lambda_0
\mathbb E
\left[
\frac{1}{d}
\left\|
v_\theta(X_t^G,t)-v_G(X_t^G,t)
\right\|^2
\right].
$$

The same norm convention should be used for both $\mathcal L_{\mathrm{FM}}$ and $\mathcal L_{\mathrm{prior}}$.

---

## 9. Numerical stability

### Positive diagonal variance

The diagonal variance should satisfy

$$
s_i>0.
$$

In practice, one can use a variance floor:

$$
s_i\leftarrow \max(s_i,s_{\min}).
$$

### Positive path variance

The denominator in the analytical prior velocity is

$$
a_t^2+b_t^2s_i.
$$

This should be positive for all $t$ and all $i$.

If $s_i>0$ and the path does not make both $a_t$ and $b_t$ zero at the same time, then this denominator is positive.

### Endpoint stability

Depending on the path schedule, it may be useful to avoid exact endpoint times in numerical optimization:

$$
t\in[\epsilon,1-\epsilon].
$$

This is a numerical choice, not a conceptual requirement.

---

## 10. Summary

Given a diagonal Gaussian prior

$$
p_G(x)=\mathcal N(\mu,\operatorname{diag}(s)),
$$

and a general affine probability path

$$
X_t=a_tX_0+b_tX_1,
$$

the analytical Gaussian-prior velocity field is

$$
\boxed{
v_G(x,t)
=
\dot b_t\mu
+
\frac{
a_t\dot a_t+b_t\dot b_t s
}{
a_t^2+b_t^2s
}
\odot
(x-b_t\mu)
}
$$

where all operations involving $s$ are elementwise.

The regularized flow matching objective is

$$
\boxed{
\mathcal L(\theta)
=
\mathbb E
\left[
\left\|
v_\theta(X_t,t)-U_t
\right\|^2
\right]
+
\lambda_0
\mathbb E
\left[
\left\|
v_\theta(X_t^G,t)-v_G(X_t^G,t)
\right\|^2
\right].
}
$$

The first term fits the data. The second term uses the analytical Gaussian-prior velocity field as a functional prior on $v_\theta$.

This provides a principled way to impose the belief that the target distribution should stay close to a diagonal Gaussian, while still allowing non-Gaussian structure to be learned from data.
