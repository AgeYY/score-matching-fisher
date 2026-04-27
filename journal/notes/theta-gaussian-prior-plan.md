I think your idea is very good. I would polish it as a **Bayesian-scaffolded conditional flow matching** method:

[
p_g(x\mid \theta)
\quad \Longrightarrow \quad
q_0(\theta\mid x)
\propto
p_g(x\mid \theta)p(\theta)
\quad \Longrightarrow \quad
q_\phi(\theta\mid x)
]

where (q_0(\theta\mid x)) is an approximate posterior source, and flow matching learns the correction from (q_0) to the true posterior (p(\theta\mid x)).

The important change is: **do not use (p_g(x\mid \theta)p(\theta)) only to find the most likely (\theta). Use it to construct the full approximate posterior distribution.**

---

## Core idea

You already have:

[
p_g(x\mid \theta)
=================

\mathcal N(x;\mu_g(\theta),\Sigma_g(\theta)).
]

Given an observed (x_i), define an approximate posterior:

[
q_0(\theta\mid x_i)
===================

\frac{
p_g(x_i\mid \theta)p(\theta)
}{
\int p_g(x_i\mid \theta')p(\theta')d\theta'
}.
]

Because (p_g(x\mid \theta)) is periodic or approximately periodic, this posterior can naturally be multimodal:

[
q_0(\theta\mid x_i)
\approx
\text{multiple bumps}.
]

Then use this (q_0(\theta\mid x_i)) as the **source distribution** for flow matching.

Your training pair is:

[
(x_i,\theta_i).
]

Instead of sampling source from a standard Gaussian,

[
\theta_0 \sim \mathcal N(0,I),
]

sample source from the approximate posterior:

[
\theta_0 \sim q_0(\theta\mid x_i).
]

Then train flow matching:

[
\theta_t=(1-t)\theta_0+t\theta_i,
]

[
v_\phi(\theta_t,t,x_i)
\approx
\theta_i-\theta_0.
]

At inference, given a new (x^\ast):

[
\theta_0\sim q_0(\theta\mid x^\ast),
]

then solve:

[
\frac{d\theta_t}{dt}=v_\phi(\theta_t,t,x^\ast),
]

and output (\theta_1\sim q_\phi(\theta\mid x^\ast)).

So the flow learns:

[
q_0(\theta\mid x)
\rightarrow
p(\theta\mid x).
]

This is much easier than:

[
\mathcal N(0,I)
\rightarrow
p(\theta\mid x).
]

---

## Why this helps

The approximate Gaussian likelihood already knows the main periodic structure:

[
p_g(x\mid \theta)\approx p_g(x\mid \theta+T).
]

Therefore (q_0(\theta\mid x)) already has the correct multimodal skeleton. The flow no longer needs to create multiple peaks from scratch. It only needs to learn the residual correction:

[
\text{Gaussian posterior scaffold}
\rightarrow
\text{true posterior}.
]

This should reduce the blurry bridge problem because the source distribution already puts mass near the posterior peaks.

---

## But you need mode-matched coupling

There is one subtle but important issue.

If you simply sample:

[
\theta_0\sim q_0(\theta\mid x_i)
]

and pair it with the target (\theta_i), you may accidentally pair source from one posterior bump with target from another bump.

For example:

[
\theta_0 \text{ from bump 1}
\quad \rightarrow \quad
\theta_i \text{ from bump 3}.
]

That creates cross-mode trajectories again, which can reintroduce blur.

So the better version is:

[
\boxed{
\text{sample } \theta_0 \text{ from the same posterior mode as } \theta_i.
}
]

---

## Polished algorithm

### Stage 1: estimate Gaussian likelihood

You already have:

[
p_g(x\mid \theta)
=================

\mathcal N(x;\mu_g(\theta),\Sigma_g(\theta)).
]

This can be your simple but robust model.

---

### Stage 2: construct approximate posterior for each (x_i)

For each training point (x_i), compute:

[
q_0(\theta\mid x_i)
\propto
p_g(x_i\mid \theta)p(\theta).
]

Because (\theta) is low-dimensional, this is practical. You can evaluate (q_0) on a dense grid of (\theta)-values.

For 1D (\theta):

```text
for each x_i:
    evaluate log p_g(x_i | theta_grid)
    add log p(theta_grid)
    normalize to get q0(theta | x_i)
```

Then find local peaks / posterior modes.

---

### Stage 3: decompose (q_0(\theta\mid x_i)) into modes

Suppose (q_0(\theta\mid x_i)) has modes:

[
q_0(\theta\mid x_i)
===================

\sum_{k=1}^K \pi_{ik}q_{0,k}(\theta\mid x_i).
]

Here (q_{0,k}) is the (k)-th posterior bump.

You do not necessarily need to fit a neural GMM. Since (\theta) is low-dimensional, you can get the modes directly from the posterior grid.

For example:

1. evaluate (q_0(\theta\mid x_i)) on a grid;
2. find local maxima;
3. define basin boundaries at valleys between maxima;
4. each basin is one posterior component.

Then assign the observed target (\theta_i) to its posterior mode:

[
k_i = \operatorname{mode}(\theta_i; q_0(\theta\mid x_i)).
]

---

### Stage 4: sample source from the matched mode

Instead of:

[
\theta_0 \sim q_0(\theta\mid x_i),
]

use:

[
\theta_0 \sim q_{0,k_i}(\theta\mid x_i).
]

So source and target are mode matched:

[
\theta_0 \text{ from mode } k_i
\quad \rightarrow \quad
\theta_i \text{ in mode } k_i.
]

This is the key MM-FM-style idea adapted to your problem.

---

### Stage 5: train conditional flow matching

Use:

[
\theta_t=(1-t)\theta_0+t\theta_i,
]

[
u_t=\theta_i-\theta_0.
]

Train:

[
\mathcal L_{\mathrm{FM}}
========================

\mathbb E
\left[
\left|
v_\phi(\theta_t,t,x_i,k_i)-(\theta_i-\theta_0)
\right|^2
\right].
]

You can include (k_i) as a mode embedding, but I would first try both versions:

[
v_\phi(\theta_t,t,x_i)
]

and

[
v_\phi(\theta_t,t,x_i,k_i).
]

The mode-conditioned version should usually be better if modes are well defined.

---

## Inference

Given a new (x^\ast):

1. Compute approximate posterior:

[
q_0(\theta\mid x^\ast)
\propto
p_g(x^\ast\mid \theta)p(\theta).
]

2. Sample a posterior mode:

[
k\sim \pi_k(x^\ast).
]

3. Sample source:

[
\theta_0\sim q_{0,k}(\theta\mid x^\ast).
]

4. Integrate the conditional flow:

[
\frac{d\theta_t}{dt}
====================

v_\phi(\theta_t,t,x^\ast,k).
]

5. Output:

[
\theta_1\sim q_\phi(\theta\mid x^\ast).
]

So the final posterior sampler is:

[
\boxed{
\theta_0\sim q_0(\theta\mid x)
\quad \xrightarrow{\text{residual flow}}
\quad
\theta_1\sim q_\phi(\theta\mid x)
}
]

---

## Why this is better than standard conditional FM

Standard conditional FM asks the model to learn:

[
\mathcal N(0,I)
\rightarrow
p(\theta\mid x).
]

Your method asks the model to learn:

[
q_0(\theta\mid x)
\rightarrow
p(\theta\mid x).
]

If (q_0) is already sufficiently good, then the flow only needs to learn a small correction.

This has several advantages:

1. **Less mode collapse**
   The posterior source already contains multiple modes.

2. **Less blurry bridge mass**
   Mode-matched coupling avoids transporting samples across posterior modes.

3. **Better sparse-data behavior**
   The Gaussian likelihood provides strong inductive bias.

4. **Better interpretability**
   You can separately inspect (q_0(\theta\mid x)) and the residual correction learned by flow matching.

5. **Easier likelihood computation**
   If you use a CNF-style flow, then:

   [
   \log q_\phi(\theta_1\mid x)
   ===========================

   ## \log q_0(\theta_0\mid x)

   \int_0^1
   \nabla_\theta\cdot v_\phi(\theta_t,t,x),dt.
   ]

   So the base density is no longer standard Gaussian; it is your approximate Bayesian posterior.

---

## The polished method name

You could call it something like:

**Posterior-Scaffolded Flow Matching**

or more specifically:

**Gaussian-Posterior-Scaffolded Conditional Flow Matching**

The method is:

[
p_g(x\mid \theta)
\Rightarrow
q_0(\theta\mid x)
\Rightarrow
\text{mode-matched residual flow}
\Rightarrow
q_\phi(\theta\mid x).
]

---

## My recommended final formulation

I would write the idea as:

> We first construct a tractable approximate posterior using the Gaussian likelihood model,
> [
> q_0(\theta\mid x)\propto p_g(x\mid\theta)p(\theta).
> ]
> Because the likelihood model preserves the periodic structure of (p(x\mid\theta)), this approximate posterior already captures the major multimodal structure of (p(\theta\mid x)). We then use (q_0(\theta\mid x)) as a conditional source distribution for flow matching. For each training pair ((x_i,\theta_i)), we assign (\theta_i) to a mode of (q_0(\theta\mid x_i)), sample (\theta_0) from the same mode, and train a conditional velocity field to transport (\theta_0) to (\theta_i). This converts the difficult problem of learning a multimodal posterior from a standard Gaussian source into a simpler residual correction from a structured approximate posterior source.

That is the clean version of your idea.