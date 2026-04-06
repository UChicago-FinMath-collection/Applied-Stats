# Study Guide: HW1 Problems 1 & 2 -- Bayesian Linear Regression Posterior

## The Big Picture (What You're Being Asked To Do)

In regular (frequentist) linear regression, you find one "best" set of coefficients
beta via OLS. In **Bayesian** linear regression, you start with prior beliefs about
beta and sigma^2, observe data, and update those beliefs into a **posterior
distribution**. The posterior tells you not just your best guess, but your
*uncertainty* about the parameters.

Problem 1 asks you to derive the exact posterior. Problem 2 asks what the
posterior mean looks like when the prior is "uninformative" (i.e., you let the
data speak for itself).

---

## Key Background Concepts

### 1. Bayes' Rule (the engine of everything)

$$
p(\theta \mid \text{data}) = \frac{p(\text{data} \mid \theta) \cdot p(\theta)}{p(\text{data})}
$$

- **Posterior** = (Likelihood x Prior) / Evidence
- Since the evidence (denominator) doesn't depend on theta, we often write:

$$
p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta) \cdot p(\theta)
$$

"Proportional to" means we can ignore constants that don't involve the
parameters. The strategy is to figure out what distribution the right-hand side
*looks like* (i.e., identify its "kernel").

### 2. What is a "Kernel"?

The **kernel** of a distribution is the part of the PDF that actually depends on
the random variable. For example, for a Gaussian:

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{1}{2\sigma^2}(x - \mu)^2\right)
$$

The kernel (in x) is just: $\exp(-\frac{1}{2\sigma^2}(x - \mu)^2)$

If you can show your posterior has this form, you know it's Gaussian without
computing the normalizing constant.

### 3. Conjugacy

A prior is **conjugate** to a likelihood if the posterior is in the same
distributional family as the prior. This is extremely convenient because you can
read off the posterior parameters by pattern-matching.

**Examples from class:**
- Beta prior + Bernoulli likelihood -> Beta posterior
- Gaussian prior + Gaussian likelihood -> Gaussian posterior
- Inv-chi-squared prior + Gaussian likelihood (known mean) -> Inv-chi-squared posterior
- **NIX prior + Gaussian linear regression likelihood -> NIX posterior** (this is your HW)

### 4. Completing the Square (the key algebra trick)

When you have a quadratic form in beta like:

$$
\beta^T A \beta - 2\beta^T b
$$

you can rewrite it as:

$$
(\beta - A^{-1}b)^T A (\beta - A^{-1}b) - b^T A^{-1} b
$$

This is the multivariate version of "completing the square." The first term is
the kernel of a Gaussian with mean $A^{-1}b$ and precision $A$. The leftover
term $-b^T A^{-1}b$ doesn't involve beta, so it gets absorbed elsewhere (into
the sigma^2 part).

**This hint is directly given in the problem statement.**

---

## The Model You're Working With

### Likelihood
$$
p(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^2, X)
= \prod_{i=1}^n \mathcal{N}(y_i \mid \mathbf{x}_i^T \boldsymbol{\beta}, \sigma^2)
$$

In vectorized form: $\mathbf{y} \sim \mathcal{N}(X\boldsymbol{\beta},\; \sigma^2 I_n)$

### Prior (Normal-Inverse-Chi-Squared = NIX)

The prior is a **joint** distribution on both beta and sigma^2:

$$
p(\boldsymbol{\beta}, \sigma^2) = \underbrace{\chi^{-2}(\sigma^2;\, \nu_0, \tau_0^2)}_{\text{marginal prior on } \sigma^2}
\cdot \underbrace{\mathcal{N}(\boldsymbol{\beta};\, \mathbf{m}_0, \sigma^2 L_0^{-1})}_{\text{conditional prior on } \beta \text{ given } \sigma^2}
$$

**Key feature:** the prior on beta *depends on* sigma^2. When sigma^2 is large,
beta is allowed to vary more (wider prior). This coupling is what makes NIX the
right conjugate family.

### The Distributions Involved

**Multivariate Gaussian** $\mathcal{N}(\mu, \Sigma)$:
$$
p(x) \propto \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)
$$
- $\Sigma^{-1}$ is the **precision matrix** (inverse of covariance)
- $L_0$ in this problem IS a precision matrix

**Scaled Inverse Chi-Squared** $\chi^{-2}(\nu_0, \tau_0^2)$:
$$
p(\sigma^2) \propto (\sigma^2)^{-\nu_0/2 - 1}
\exp\left(-\frac{\nu_0 \tau_0^2}{2\sigma^2}\right)
$$
- $\nu_0$ = degrees of freedom (controls how peaked the distribution is)
- $\tau_0^2$ = scale (roughly the prior mean of sigma^2; exactly $E[\sigma^2] = \frac{\nu_0}{\nu_0 - 2}\tau_0^2$ for $\nu_0 > 2$)
- Equivalent to InvGamma($\nu_0/2$, $\nu_0\tau_0^2/2$)

---

## Problem 1: Step-by-Step Strategy

### Step 1: Write the Joint

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)
\propto \underbrace{p(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^2, X)}_{\text{likelihood}}
\cdot \underbrace{p(\boldsymbol{\beta} \mid \sigma^2, \mathbf{m}_0, L_0)}_{\text{prior on } \beta}
\cdot \underbrace{p(\sigma^2 \mid \nu_0, \tau_0^2)}_{\text{prior on } \sigma^2}
$$

### Step 2: Collect Powers of sigma^2

Each factor contributes a power of sigma^2:
- Likelihood: $(\sigma^2)^{-n/2}$ (from the normalizing constant of n Gaussians)
- Prior on beta: $(\sigma^2)^{-p/2}$ (from $|\sigma^2 L_0^{-1}|^{-1/2}$)
- Prior on sigma^2: $(\sigma^2)^{-\nu_0/2 - 1}$

Total power: $(\sigma^2)^{-(n + p + \nu_0)/2 - 1}$

In the posterior NIX, the powers come from:
- Posterior chi-squared: $(\sigma^2)^{-\nu_n/2 - 1}$
- Posterior Gaussian on beta: $(\sigma^2)^{-p/2}$

Matching: $\nu_n + p = n + p + \nu_0$, giving:

$$\boxed{\nu_n = \nu_0 + n}$$

### Step 3: Collect Terms in the Exponent Involving beta

Everything in the exponent is divided by $-\frac{1}{2\sigma^2}$. Collect the
quadratic terms in beta:

**From likelihood** (expand $(y - X\beta)^T(y - X\beta)$):
$$
\mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T X^T \mathbf{y}
+ \boldsymbol{\beta}^T X^T X \boldsymbol{\beta}
$$

**From prior on beta** (expand $(\beta - m_0)^T L_0 (\beta - m_0)$):
$$
\mathbf{m}_0^T L_0 \mathbf{m}_0 - 2\boldsymbol{\beta}^T L_0 \mathbf{m}_0
+ \boldsymbol{\beta}^T L_0 \boldsymbol{\beta}
$$

**From prior on sigma^2**: $\nu_0 \tau_0^2$ (no beta terms)

Combining the beta-quadratic parts:
$$
\boldsymbol{\beta}^T \underbrace{(X^T X + L_0)}_{L_n} \boldsymbol{\beta}
- 2\boldsymbol{\beta}^T \underbrace{(X^T \mathbf{y} + L_0 \mathbf{m}_0)}_{L_n \mathbf{m}_n}
$$

This gives us:

$$\boxed{L_n = L_0 + X^T X}$$

$$\boxed{\mathbf{m}_n = L_n^{-1}(L_0 \mathbf{m}_0 + X^T \mathbf{y})}$$

### Step 4: Complete the Square to Find tau_n^2

Use the completing-the-square identity from the hint to separate the beta terms
from the leftovers:

$$
\boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n \mathbf{m}_n
= (\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)
- \mathbf{m}_n^T L_n \mathbf{m}_n
$$

The first part becomes the kernel of the posterior Gaussian on beta. The leftover
$-\mathbf{m}_n^T L_n \mathbf{m}_n$ combines with the other non-beta terms to form
the tau_n^2 parameter:

$$
\nu_n \tau_n^2 = \nu_0 \tau_0^2 + \mathbf{y}^T\mathbf{y}
+ \mathbf{m}_0^T L_0 \mathbf{m}_0 - \mathbf{m}_n^T L_n \mathbf{m}_n
$$

$$\boxed{\tau_n^2 = \frac{1}{\nu_n}\left(\nu_0 \tau_0^2 + \mathbf{y}^T\mathbf{y}
+ \mathbf{m}_0^T L_0 \mathbf{m}_0 - \mathbf{m}_n^T L_n \mathbf{m}_n\right)}$$

**Equivalent form** (which may be easier to interpret):
$$
\tau_n^2 = \frac{1}{\nu_n}\left[\nu_0 \tau_0^2
+ (\mathbf{y} - X\mathbf{m}_n)^T(\mathbf{y} - X\mathbf{m}_n)
+ (\mathbf{m}_n - \mathbf{m}_0)^T L_0 (\mathbf{m}_n - \mathbf{m}_0)\right]
$$

This second form is nice because:
- First term: prior contribution
- Second term: residual sum of squares at the posterior mean
- Third term: how far the posterior mean moved from the prior mean

---

## Problem 2: Uninformative Limit

### Part (a): E[beta] as L_0 -> 0, nu_0 -> 0

The posterior mean of beta is:
$$
\mathbf{m}_n = (L_0 + X^TX)^{-1}(L_0 \mathbf{m}_0 + X^T\mathbf{y})
$$

As $L_0 \to 0$:
$$
\mathbf{m}_n \to (X^TX)^{-1} X^T \mathbf{y} = \hat{\boldsymbol{\beta}}_{\text{OLS}}
$$

**This is exactly the ordinary least squares (OLS) estimator!** The Bayesian
posterior mean with an uninformative prior reduces to the frequentist answer.

### Part (b): E[sigma^2] as L_0 -> 0, nu_0 -> 0

The posterior marginal for sigma^2 is $\chi^{-2}(\nu_n, \tau_n^2)$, with mean:
$$
E[\sigma^2] = \frac{\nu_n}{\nu_n - 2}\tau_n^2 \quad (\text{requires } \nu_n > 2)
$$

In the limit $L_0 \to 0$, $\nu_0 \to 0$:
- $\nu_n = n$
- $\mathbf{m}_n = (X^TX)^{-1}X^T\mathbf{y} = \hat{\boldsymbol{\beta}}_{\text{OLS}}$
- $L_n = X^TX$
- $\nu_n \tau_n^2 = \mathbf{y}^T\mathbf{y} - \mathbf{m}_n^T X^TX \mathbf{m}_n$

Now use the hat matrix $H = X(X^TX)^{-1}X^T$:
$$
\nu_n \tau_n^2 = \mathbf{y}^T\mathbf{y} - \mathbf{y}^TX(X^TX)^{-1}X^T\mathbf{y}
= \mathbf{y}^T(I_n - H)\mathbf{y}
$$

Note: $\mathbf{y}^T(I - H)\mathbf{y} = \|\mathbf{y} - X\hat{\beta}_{OLS}\|^2$ = residual sum of squares (RSS).

So $\tau_n^2 = \mathbf{y}^T(I_n - H)\mathbf{y}/n$ and:
$$
E[\sigma^2] = \frac{n}{n-2} \cdot \frac{\mathbf{y}^T(I_n - H)\mathbf{y}}{n}
= \frac{\mathbf{y}^T(I_n - H)\mathbf{y}}{n - 2}
$$

This is closely related to the classical unbiased variance estimator
$s^2 = \text{RSS}/(n - p)$.

---

## Key Identities and Facts You'll Need

| Identity | What It Is |
|----------|-----------|
| $(y - X\beta)^T(y - X\beta) = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$ | Expanding the residual sum of squares |
| $a^TBa - 2a^TBc = (a-c)^TB(a-c) - c^TBc$ | Completing the square (given as Hint 2) |
| $H = X(X^TX)^{-1}X^T$ | The hat matrix (projects y onto column space of X) |
| $y^T(I-H)y = \|y - X\hat\beta_{OLS}\|^2$ | RSS in terms of hat matrix |
| $E[\sigma^2]$ for $\chi^{-2}(\nu, \tau^2)$ is $\frac{\nu}{\nu-2}\tau^2$ | Mean of scaled inverse chi-squared |
| OLS: $\hat\beta = (X^TX)^{-1}X^Ty$ | Ordinary least squares solution |

---

## Conceptual Analogies to Help Intuition

### The "Pseudocount" Interpretation
Just like in Beta-Bernoulli conjugacy (from Lecture 1), where the prior
parameters act as "pseudo-observations":
- **L_0** acts like a prior "precision from pseudo-data" -- it's like you've already
  seen some data that contributes precision L_0
- **nu_0** acts like the number of prior pseudo-observations for estimating sigma^2
- **tau_0^2** acts like the prior estimate of sigma^2

### Posterior Mean as Weighted Average
The posterior mean $\mathbf{m}_n = L_n^{-1}(L_0\mathbf{m}_0 + X^T\mathbf{y})$ is a
matrix-weighted average of the prior mean and the data, exactly analogous to
the scalar Gaussian-Gaussian case from Lecture 2:
$$
\mu_n = \rho \bar{y} + (1-\rho)\mu_0
$$

### Connection to Ridge Regression
When L_0 = alpha * I and m_0 = 0, the posterior mean becomes the ridge
regression solution $(X^TX + \alpha I)^{-1}X^Ty$. The prior precision L_0 plays
the role of the regularization parameter.
