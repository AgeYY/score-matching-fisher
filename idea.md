Yes. The main idea I’d recommend is:

**do not estimate (p(x\mid \theta)) itself. Estimate the (\theta)-score directly.**

Let
[
s(x,\theta) ;:=; \nabla_\theta \log p(x\mid \theta),
\qquad
\mathcal I(\theta);=;\mathbb E_{x\sim p(x\mid \theta)}\big[s(x,\theta)s(x,\theta)^\top\big].
]

If your prior (p(\theta)) is uniform on its support, then inside the interior of that support,
[
\nabla_\theta \log p(x\mid \theta)
==================================

\nabla_\theta \log p(\theta\mid x),
]
because
[
\log p(\theta\mid x)=\log p(x\mid \theta)+\log p(\theta)-\log p(x),
]
and the last two terms have zero (\theta)-derivative in the interior. So you can turn the problem into **estimating the posterior score in (\theta)-space**, which is usually much lower-dimensional than (x). That is the key shortcut.

A very practical way to do this is **conditional denoising score matching on (\theta)**, not flow matching on (x). Ordinary score matching goes through Hyvärinen’s objective, while denoising score matching avoids the divergence/trace term that makes direct score matching expensive in high dimensions. 

Use the data pairs ((x_i,\theta_i)). Add Gaussian noise only to (\theta):
[
\tilde\theta=\theta+\sigma \varepsilon,\qquad \varepsilon\sim \mathcal N(0,I).
]
Train a network (s_\phi(\tilde\theta,x,\sigma)) with
[
\mathcal L(\phi)
================

\mathbb E\Bigg[
\left|
s_\phi(\tilde\theta,x,\sigma)
+
\frac{\tilde\theta-\theta}{\sigma^2}
\right|^2
\Bigg].
]
This learns (\nabla_{\tilde\theta}\log p_\sigma(\tilde\theta\mid x)), the score of the noise-smoothed posterior. As (\sigma\to 0), this approaches (\nabla_\theta \log p(\theta\mid x)=\nabla_\theta\log p(x\mid \theta)). The advantage over your flow-matching idea is that the network is trained to **output the score directly**; you do not need to differentiate a learned velocity field to get it. Conditional score learning from paired samples is exactly the kind of setting where this is natural. ([arXiv][1])

Then estimate Fisher information by local averaging over nearby (\theta)’s:
[
\widehat{\mathcal I}_\sigma(\theta_0)
=====================================

\frac{
\sum_{i=1}^n
K_h(\theta_i-\theta_0),
s_\phi(\theta_0,x_i,\sigma)s_\phi(\theta_0,x_i,\sigma)^\top
}{
\sum_{i=1}^n K_h(\theta_i-\theta_0)
}.
]
Here (K_h) is a kernel in (\theta)-space. If you have repeated samples at the same (\theta), this becomes just a sample average. The smoothness you already expect in (\theta) is exactly what makes this kernel step reasonable.

The main caveat is that denoising score methods get harder at very low noise: the low-(\sigma) regime is known to be unstable, so you should not trust a single tiny (\sigma). A good practice is to train on several small noise levels and extrapolate (\widehat{\mathcal I}_\sigma(\theta)) to (\sigma\to 0). 

There is also a second route that is even more robust if you only need Fisher information and not the score itself:

**estimate Fisher information from local classification.**

For scalar (\theta), fix (\theta_0) and a small (\varepsilon). Build a binary problem:

* class 1: samples from (p(x\mid \theta_0+\varepsilon/2)),
* class 0: samples from (p(x\mid \theta_0-\varepsilon/2)).

If a classifier outputs the logit
[
\ell(x)\approx \log \frac{p(x\mid \theta_0+\varepsilon/2)}{p(x\mid \theta_0-\varepsilon/2)},
]
then by Taylor expansion,
[
\ell(x)=\varepsilon,\partial_\theta \log p(x\mid \theta_0)+O(\varepsilon^3),
]
so
[
\frac{1}{\varepsilon^2},\mathbb E[\ell(x)^2]
;\to;
\mathcal I(\theta_0).
]
For vector (\theta), do this along a direction (v):
[
\frac{1}{\varepsilon^2},\mathbb E[\ell_v(x)^2]
;\to;
v^\top \mathcal I(\theta_0)v.
]
Then recover the matrix from several directions. This is essentially a **density-ratio / local distinguishability** estimator of Fisher information. A recent paper called ClassiFIM formalizes this classifier-to-Fisher reduction and proves consistency under regularity conditions. ([arXiv][2])

Between the two, my recommendation is:

* If (\dim(\theta)) is small and (x) is high-dimensional, use **conditional score estimation in (\theta)-space**.
* If you want the most stable, Jacobian-free estimator and can tolerate estimating Fisher information only through directional probes, use **local classification**.

A few honest limitations:

1. Truly “assumption-free” estimation is impossible in finite samples. You still need some structure: smoothness in (\theta), enough local data near each (\theta_0), and capacity control/regularization for the estimator.
2. Near the boundary of a bounded uniform prior, the identity
   (\nabla_\theta \log p(x\mid\theta)=\nabla_\theta\log p(\theta\mid x))
   needs boundary care. In practice, either work away from the edges, reflect the noise, or reparameterize (\theta) to an unconstrained variable.
3. If (\theta) is high-dimensional too, Fisher information estimation becomes much harder; then the classifier route with random directional probes is often better than trying to learn the full matrix directly.

So, compared with your two current ideas:

* **Wishart / full conditional density modeling**: overkill unless you already know a good family.
* **Flow matching likelihood + differentiate velocity**: not the cleanest path for Fisher information.
* **Best option**: estimate the **(\theta)-score directly**, or skip the score entirely and estimate Fisher information through **local classification**.

If you want, I can write down a concrete training recipe for the conditional-score method, including bandwidth choice in (\theta) and how to extrapolate (\sigma\to 0).

[1]: https://arxiv.org/pdf/2511.03953 "Conditional Score Learning for Quickest Change Detection in Markov Transition Kernels"
[2]: https://arxiv.org/html/2408.03323v1 "ClassiFIM: An Unsupervised Method To Detect Phase Transitions"