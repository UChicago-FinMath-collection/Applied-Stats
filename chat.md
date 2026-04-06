# Chat

---

## Q: Work through Problem 2 in detail.

### A: Problem 2 -- Detailed Solution

We use the posterior parameters derived in Problem 1 and take the uninformative limit $L_0 \to 0$ and $\nu_0 \to 0$.

---

### Part (a): $\mathbb{E}[\boldsymbol{\beta} \mid \mathbf{y}, X, \boldsymbol{\eta}_0]$ in the uninformative limit

**Setup.** From Problem 1, the posterior on $\boldsymbol{\beta}$ (conditional on $\sigma^2$) is Gaussian with mean $\mathbf{m}_n$. Since the mean of a Gaussian is also its expectation, and the NIX structure means $\mathbb{E}[\boldsymbol{\beta} \mid \mathbf{y}, X] = \mathbf{m}_n$ regardless of $\sigma^2$ (the conditional mean doesn't depend on $\sigma^2$), we have:

$$
\mathbb{E}[\boldsymbol{\beta} \mid \mathbf{y}, X, \boldsymbol{\eta}_0] = \mathbf{m}_n = L_n^{-1}(L_0\mathbf{m}_0 + X^T\mathbf{y})
$$

**Why the mean doesn't depend on $\sigma^2$:** In the NIX posterior, $p(\boldsymbol{\beta} \mid \sigma^2, \mathbf{y}, X) = \mathcal{N}(\mathbf{m}_n, \sigma^2 L_n^{-1})$. The mean $\mathbf{m}_n$ is the same for every value of $\sigma^2$ -- only the variance $\sigma^2 L_n^{-1}$ changes. So when we marginalize over $\sigma^2$:

$$
\mathbb{E}[\boldsymbol{\beta} \mid \mathbf{y}, X]
= \mathbb{E}_{\sigma^2}\!\Big[\mathbb{E}[\boldsymbol{\beta} \mid \sigma^2, \mathbf{y}, X]\Big]
= \mathbb{E}_{\sigma^2}[\mathbf{m}_n]
= \mathbf{m}_n
$$

**Now take the limit.** As $L_0 \to 0$:

$$
L_n = L_0 + X^TX \;\to\; X^TX
$$

$$
L_n\mathbf{m}_n = L_0\mathbf{m}_0 + X^T\mathbf{y} \;\to\; X^T\mathbf{y}
$$

Therefore:

$$
\mathbf{m}_n \to (X^TX)^{-1}X^T\mathbf{y}
$$

We recognize this immediately:

$$
\boxed{\lim_{L_0 \to 0,\;\nu_0 \to 0}\;\mathbb{E}[\boldsymbol{\beta} \mid \mathbf{y}, X, \boldsymbol{\eta}_0] = (X^TX)^{-1}X^T\mathbf{y} = \hat{\boldsymbol{\beta}}_{\text{OLS}}}
$$

**Interpretation:** With an uninformative prior (no prior opinion about $\boldsymbol{\beta}$), the Bayesian posterior mean reduces to the ordinary least squares estimator. The prior contributes nothing, so the data fully determines the estimate.

Note: $\nu_0 \to 0$ does not affect $\mathbf{m}_n$ at all -- it only appears in $\nu_n$ and $\tau_n^2$.

---

### Part (b): $\mathbb{E}[\sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0]$ in the uninformative limit

**Setup.** From Problem 1, the posterior marginal on $\sigma^2$ is $\chi^{-2}(\nu_n, \tau_n^2)$. The mean of a scaled inverse chi-squared distribution is:

$$
\mathbb{E}[\sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0] = \frac{\nu_n}{\nu_n - 2}\,\tau_n^2
\qquad\text{(valid for } \nu_n > 2\text{)}
$$

**Step 1: Find $\nu_n$ in the limit.**

$$
\nu_n = \nu_0 + n \;\xrightarrow{\nu_0 \to 0}\; n
$$

**Step 2: Find $\tau_n^2$ in the limit.**

From Problem 1:

$$
\tau_n^2 = \frac{1}{\nu_n}\Big(\nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n\Big)
$$

As $L_0 \to 0$ and $\nu_0 \to 0$:
- $\nu_0\tau_0^2 \to 0$ (prior contribution vanishes)
- $\mathbf{m}_0^TL_0\mathbf{m}_0 \to 0$ (since $L_0 \to 0$, this quadratic form vanishes)
- $L_n \to X^TX$
- $\mathbf{m}_n \to (X^TX)^{-1}X^T\mathbf{y} = \hat{\boldsymbol{\beta}}_{\text{OLS}}$
- $\nu_n \to n$

So:

$$
\tau_n^2 \to \frac{1}{n}\Big(\mathbf{y}^T\mathbf{y} - \hat{\boldsymbol{\beta}}_{\text{OLS}}^T X^TX\, \hat{\boldsymbol{\beta}}_{\text{OLS}}\Big)
\tag{$*$}
$$

**Step 3: Simplify $\hat{\boldsymbol{\beta}}_{\text{OLS}}^T X^TX\, \hat{\boldsymbol{\beta}}_{\text{OLS}}$ using the hat matrix.**

Substituting $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (X^TX)^{-1}X^T\mathbf{y}$:

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}}^T X^TX\, \hat{\boldsymbol{\beta}}_{\text{OLS}}
= \Big[(X^TX)^{-1}X^T\mathbf{y}\Big]^T X^TX \Big[(X^TX)^{-1}X^T\mathbf{y}\Big]
$$

$$
= \mathbf{y}^TX\underbrace{(X^TX)^{-1} X^TX}_{=\;I}\,(X^TX)^{-1}X^T\mathbf{y}
$$

$$
= \mathbf{y}^T\underbrace{X(X^TX)^{-1}X^T}_{=\;H}\,\mathbf{y}
$$

$$
= \mathbf{y}^TH\mathbf{y}
$$

where $H = X(X^TX)^{-1}X^T$ is the hat matrix.

**Step 4: Substitute back into ($*$).**

$$
\tau_n^2 \to \frac{1}{n}\Big(\mathbf{y}^T\mathbf{y} - \mathbf{y}^TH\mathbf{y}\Big) = \frac{\mathbf{y}^T(I_n - H)\mathbf{y}}{n}
$$

Note: $\mathbf{y}^T(I-H)\mathbf{y} = \|\mathbf{y} - X\hat{\boldsymbol{\beta}}_{\text{OLS}}\|^2 = \text{RSS}$ (residual sum of squares), since $H\mathbf{y} = X\hat{\boldsymbol{\beta}}_{\text{OLS}}$ are the fitted values.

**Step 5: Compute the expectation.**

$$
\mathbb{E}[\sigma^2] = \frac{\nu_n}{\nu_n - 2}\,\tau_n^2
\;\to\;
\frac{n}{n-2}\cdot\frac{\mathbf{y}^T(I_n - H)\mathbf{y}}{n}
$$

$$
\boxed{\lim_{L_0 \to 0,\;\nu_0 \to 0}\;\mathbb{E}[\sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0] = \frac{\mathbf{y}^T(I_n - H)\mathbf{y}}{n - 2}}
$$

**Interpretation:** Compare this to the classical unbiased variance estimator $s^2 = \frac{\text{RSS}}{n-p}$ where $p$ is the number of parameters. Our answer has $n-2$ in the denominator because $p = 2$ in this problem (the covariates are $\mathbf{x}_i = (1, x_i) \in \mathbb{R}^2$). More generally, with a $p$-dimensional $\boldsymbol{\beta}$ and the uninformative limit, the posterior mean of $\sigma^2$ equals:

$$
\frac{\mathbf{y}^T(I_n - H)\mathbf{y}}{n - 2} = \frac{\text{RSS}}{n-2}
$$

This is very close to the classical estimator. The slight difference (using $n - 2$ from the $\chi^{-2}$ mean formula vs. the classical $n - p$) is because the $\chi^{-2}$ mean formula $\frac{\nu}{\nu-2}$ introduces the factor, and here $\nu_n = n$ while the "effective degrees of freedom" for the residuals is $n - p$. In general for $p$-dimensional regression with a fully uninformative prior, the posterior degrees of freedom in the $\chi^{-2}$ is $\nu_n = n$ but the RSS has $n-p$ effective degrees of freedom, so the result simplifies to $\text{RSS}/(n-2)$ specifically because $p=2$ in this problem.

---

---

## Q: Write a formal solution for Problem 1, as I would answer on the assignment.

### A: Problem 1 -- Formal Solution

**Goal:** Show that the posterior $p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)$ is $\text{NIX}(\mathbf{m}_n, L_n, \nu_n, \tau_n^2)$ and find the posterior parameters.

**Proof.**

By Bayes' rule, the joint posterior is proportional to the product of the likelihood and prior:

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)
\propto p(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^2, X)\; p(\boldsymbol{\beta} \mid \sigma^2, \mathbf{m}_0, L_0)\; p(\sigma^2 \mid \nu_0, \tau_0^2)
$$

Writing out each term and dropping constants that do not depend on $\boldsymbol{\beta}$ or $\sigma^2$:

$$
\propto (\sigma^2)^{-n/2} \exp\!\left(-\frac{1}{2\sigma^2}(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})\right)
\cdot (\sigma^2)^{-p/2} \exp\!\left(-\frac{1}{2\sigma^2}(\boldsymbol{\beta}-\mathbf{m}_0)^TL_0(\boldsymbol{\beta}-\mathbf{m}_0)\right)
\cdot (\sigma^2)^{-\nu_0/2-1} \exp\!\left(-\frac{\nu_0\tau_0^2}{2\sigma^2}\right)
$$

Combining:

$$
\propto (\sigma^2)^{-(n+p+\nu_0)/2-1}
\exp\!\left(-\frac{1}{2\sigma^2}\Big[
(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})
+ (\boldsymbol{\beta}-\mathbf{m}_0)^TL_0(\boldsymbol{\beta}-\mathbf{m}_0)
+ \nu_0\tau_0^2
\Big]\right)
\tag{1}
$$

We now expand the two quadratic forms inside the exponent. Expanding the likelihood term:

$$
(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})
= \boldsymbol{\beta}^TX^TX\boldsymbol{\beta} - 2\boldsymbol{\beta}^TX^T\mathbf{y} + \mathbf{y}^T\mathbf{y}
$$

Expanding the prior term:

$$
(\boldsymbol{\beta}-\mathbf{m}_0)^TL_0(\boldsymbol{\beta}-\mathbf{m}_0)
= \boldsymbol{\beta}^TL_0\boldsymbol{\beta} - 2\boldsymbol{\beta}^TL_0\mathbf{m}_0 + \mathbf{m}_0^TL_0\mathbf{m}_0
$$

Collecting terms by dependence on $\boldsymbol{\beta}$, the bracket in the exponent of (1) becomes:

$$
\boldsymbol{\beta}^T\underbrace{(X^TX + L_0)}_{L_n}\boldsymbol{\beta}
\;-\; 2\boldsymbol{\beta}^T\underbrace{(X^T\mathbf{y} + L_0\mathbf{m}_0)}_{L_n\mathbf{m}_n}
\;+\; \underbrace{\mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2}_{c}
\tag{2}
$$

Define the posterior precision and posterior mean:

$$
L_n \triangleq X^TX + L_0, \qquad
\mathbf{m}_n \triangleq L_n^{-1}(X^T\mathbf{y} + L_0\mathbf{m}_0)
$$

We now apply the completing-the-square identity $\mathbf{a}^TB\mathbf{a} - 2\mathbf{a}^TB\mathbf{c} = (\mathbf{a}-\mathbf{c})^TB(\mathbf{a}-\mathbf{c}) - \mathbf{c}^TB\mathbf{c}$ with $\mathbf{a}=\boldsymbol{\beta}$, $B=L_n$, $\mathbf{c}=\mathbf{m}_n$ to rewrite (2) as:

$$
(\boldsymbol{\beta}-\mathbf{m}_n)^TL_n(\boldsymbol{\beta}-\mathbf{m}_n)
\;-\; \mathbf{m}_n^TL_n\mathbf{m}_n
\;+\; \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2
$$

$$
= (\boldsymbol{\beta}-\mathbf{m}_n)^TL_n(\boldsymbol{\beta}-\mathbf{m}_n)
\;+\; \underbrace{\nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n}_{\triangleq\; \nu_n\tau_n^2}
\tag{3}
$$

Substituting (3) back into (1) and separating the $\sigma^2$ power as $(\sigma^2)^{-\nu_n/2-1}\cdot(\sigma^2)^{-p/2}$ where $\nu_n = \nu_0 + n$:

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)
\propto
\underbrace{(\sigma^2)^{-\nu_n/2-1}\exp\!\left(-\frac{\nu_n\tau_n^2}{2\sigma^2}\right)}_{\chi^{-2}(\sigma^2;\;\nu_n,\,\tau_n^2)}
\;\cdot\;
\underbrace{(\sigma^2)^{-p/2}\exp\!\left(-\frac{(\boldsymbol{\beta}-\mathbf{m}_n)^TL_n(\boldsymbol{\beta}-\mathbf{m}_n)}{2\sigma^2}\right)}_{\mathcal{N}(\boldsymbol{\beta};\;\mathbf{m}_n,\,\sigma^2L_n^{-1})}
$$

This is precisely the kernel of $\text{NIX}(\mathbf{m}_n, L_n, \nu_n, \tau_n^2)$. Since the posterior must normalize to 1, we conclude:

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0) = \text{NIX}(\boldsymbol{\beta}, \sigma^2;\;\mathbf{m}_n, L_n, \nu_n, \tau_n^2)
$$

with posterior parameters:

$$
\boxed{
\begin{aligned}
\nu_n &= \nu_0 + n \\[4pt]
L_n &= L_0 + X^TX \\[4pt]
\mathbf{m}_n &= L_n^{-1}(L_0\mathbf{m}_0 + X^T\mathbf{y}) \\[4pt]
\tau_n^2 &= \frac{1}{\nu_n}\left(\nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n\right)
\end{aligned}
}
$$

$\blacksquare$

---

## Q: Still confused on Step 4 for finding $\tau_n^2$. Could you re-explain?

### A:

The confusion is natural -- Step 4 is really just "cleaning up the leftovers from Step 3." Here's the idea from scratch.

**Where we are after Step 3.** The entire exponent (the stuff inside $\exp(-\frac{1}{2\sigma^2}[\ldots])$) contains these terms:

$$
\underbrace{\boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n\mathbf{m}_n}_{\text{depends on } \boldsymbol{\beta}}
\quad+\quad
\underbrace{\mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2}_{\text{does NOT depend on } \boldsymbol{\beta}}
$$

In Step 3, we said the left part "is the Gaussian kernel." But that's not quite right yet! The Gaussian kernel is $(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)$, not $\boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n\mathbf{m}_n$. These two things **differ by a constant**.

**The problem.** Watch what happens when you expand the "nice" form:

$$
(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)
= \boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n\mathbf{m}_n + \mathbf{m}_n^TL_n\mathbf{m}_n
$$

See that extra $+\mathbf{m}_n^TL_n\mathbf{m}_n$ at the end? Our expression doesn't have it. So:

$$
\boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n\mathbf{m}_n
= (\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n) \;{\color{red} - \;\mathbf{m}_n^TL_n\mathbf{m}_n}
$$

That red term is a **leftover constant** -- it doesn't depend on $\boldsymbol{\beta}$, so it can't go into the Gaussian. It has to go somewhere else.

**Where does it go?** Into the $\chi^{-2}$ part! Substituting back into the full exponent:

$$
-\frac{1}{2\sigma^2}\bigg[
\underbrace{(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)}_{\text{this IS the Gaussian kernel now}}
+
\underbrace{\mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2 - \mathbf{m}_n^TL_n\mathbf{m}_n}_{\text{all the constants (including the leftover)}}
\bigg]
$$

**Now recall what the $\chi^{-2}$ kernel looks like.** The density of $\chi^{-2}(\sigma^2; \nu_n, \tau_n^2)$ has:

$$
\exp\!\left(-\frac{\nu_n\tau_n^2}{2\sigma^2}\right)
$$

So whatever ends up as the "constant group" in our exponent must equal $\nu_n\tau_n^2$. This is the same "read off the coefficients" logic as Step 3, but for the $\chi^{-2}$ instead of the Gaussian:

$$
\nu_n\tau_n^2 = \nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n
$$

Divide by $\nu_n = \nu_0 + n$:

$$
\boxed{\tau_n^2 = \frac{1}{\nu_0 + n}\Big(\nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n\Big)}
$$

**Summary of the logic in one sentence:** Completing the square converts the $\boldsymbol{\beta}$ terms into a clean Gaussian kernel, but produces a leftover constant $(-\mathbf{m}_n^TL_n\mathbf{m}_n)$ which joins the other constants to form $\nu_n\tau_n^2$.

---

## Q: How do we know how the kernel looks in Step 3 when we match it with our prior?

### A:

The key idea: we **already know** what a multivariate Gaussian looks like when you expand it. So we just compare.

**What we're trying to match.** The posterior Gaussian on $\boldsymbol{\beta}$ should be $\mathcal{N}(\boldsymbol{\beta}; \mathbf{m}_n, \sigma^2 L_n^{-1})$. The kernel (the part inside $\exp$) of *any* Gaussian $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \Sigma)$ always has the same shape:

$$
\exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

In our case, $\mathbf{x} = \boldsymbol{\beta}$, $\boldsymbol{\mu} = \mathbf{m}_n$, and $\Sigma^{-1} = \frac{1}{\sigma^2}L_n$ (since the covariance is $\sigma^2 L_n^{-1}$, the precision is $\frac{1}{\sigma^2}L_n$). So the kernel must be:

$$
\exp\!\left(-\frac{1}{2\sigma^2}(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)\right)
$$

**Now expand this** using $(a-b)^TWa - 2a^TWb + b^TWb$:

$$
(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)
= \boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n \mathbf{m}_n + \mathbf{m}_n^T L_n \mathbf{m}_n
$$

**Meanwhile, from the derivation** we got:

$$
\boldsymbol{\beta}^T \underbrace{(X^TX + L_0)}_{?} \boldsymbol{\beta}
- 2\boldsymbol{\beta}^T \underbrace{(X^T\mathbf{y} + L_0\mathbf{m}_0)}_{?}
+ \text{constants}
$$

**The matching is just "read off the coefficients":**

| Term type | From derivation | Must equal (from Gaussian kernel) |
|-----------|----------------|-----------------------------------|
| Coefficient of $\boldsymbol{\beta}^T[\cdot]\boldsymbol{\beta}$ | $X^TX + L_0$ | $L_n$ |
| Coefficient of $-2\boldsymbol{\beta}^T[\cdot]$ | $X^T\mathbf{y} + L_0\mathbf{m}_0$ | $L_n\mathbf{m}_n$ |

From the first row: $L_n = X^TX + L_0$.

From the second row: $L_n\mathbf{m}_n = X^T\mathbf{y} + L_0\mathbf{m}_0$, so $\mathbf{m}_n = L_n^{-1}(X^T\mathbf{y} + L_0\mathbf{m}_0)$.

**In short:** we didn't "guess" the kernel shape. Every Gaussian has the same quadratic form $\mathbf{x}^T A \mathbf{x} - 2\mathbf{x}^T \mathbf{b} + \text{const}$ inside the exponent. If your expression has that shape, it *is* a Gaussian with precision $A$ and mean $A^{-1}\mathbf{b}$. You just read off $A$ and $\mathbf{b}$ from whatever you derived.

---

## Q: Give me the solution to Q1 in the steps mentioned in the guide. Show step 2 in detail.

### A:

We want the joint posterior $p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)$ and need to show it is NIX with parameters $\nu_n, \tau_n^2, \mathbf{m}_n, L_n$.

---

### Step 1: Write the Joint

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)
\propto
\underbrace{p(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^2, X)}_{\text{likelihood}}
\cdot
\underbrace{p(\boldsymbol{\beta} \mid \sigma^2, \mathbf{m}_0, L_0)}_{\text{prior on } \beta}
\cdot
\underbrace{p(\sigma^2 \mid \nu_0, \tau_0^2)}_{\text{prior on } \sigma^2}
$$

Writing out each factor explicitly:

**Likelihood:**
$$
p(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^2, X)
= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(-\frac{1}{2\sigma^2}(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2\right)
= (2\pi\sigma^2)^{-n/2}
\exp\!\left(-\frac{1}{2\sigma^2}(\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})\right)
$$

**Prior on $\boldsymbol{\beta}$:**
$$
p(\boldsymbol{\beta} \mid \sigma^2, \mathbf{m}_0, L_0)
= (2\pi)^{-p/2} |\sigma^2 L_0^{-1}|^{-1/2}
\exp\!\left(-\frac{1}{2\sigma^2}(\boldsymbol{\beta} - \mathbf{m}_0)^T L_0 (\boldsymbol{\beta} - \mathbf{m}_0)\right)
$$

Note: $|\sigma^2 L_0^{-1}| = (\sigma^2)^p |L_0|^{-1}$, so $|\sigma^2 L_0^{-1}|^{-1/2} \propto (\sigma^2)^{-p/2}$.

**Prior on $\sigma^2$:**
$$
p(\sigma^2 \mid \nu_0, \tau_0^2)
= \frac{(\frac{\nu_0 \tau_0^2}{2})^{\nu_0/2}}{\Gamma(\nu_0/2)}
(\sigma^2)^{-\nu_0/2 - 1}
\exp\!\left(-\frac{\nu_0 \tau_0^2}{2\sigma^2}\right)
$$

Dropping all constants that don't involve $\boldsymbol{\beta}$ or $\sigma^2$:

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)
\propto
(\sigma^2)^{-n/2 - p/2 - \nu_0/2 - 1}
\exp\!\left(
-\frac{1}{2\sigma^2}\Big[
(\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})
+ (\boldsymbol{\beta} - \mathbf{m}_0)^T L_0 (\boldsymbol{\beta} - \mathbf{m}_0)
+ \nu_0 \tau_0^2
\Big]
\right)
$$

---

### Step 2: Collect Powers of $\sigma^2$ (IN DETAIL)

The posterior should factor as:

$$
\text{NIX} = \underbrace{\chi^{-2}(\sigma^2 \mid \nu_n, \tau_n^2)}_{\propto\; (\sigma^2)^{-\nu_n/2-1} \exp(\ldots)}
\cdot
\underbrace{\mathcal{N}(\boldsymbol{\beta} \mid \mathbf{m}_n, \sigma^2 L_n^{-1})}_{\propto\; (\sigma^2)^{-p/2} \exp(\ldots)}
$$

So the total power of $\sigma^2$ in the posterior NIX is:

$$
(\sigma^2)^{-\nu_n/2 - 1 - p/2}
$$

From Step 1, the total power of $\sigma^2$ we computed from the joint is:

$$
(\sigma^2)^{-n/2 - p/2 - \nu_0/2 - 1}
$$

Let's see where each piece came from:

| Source | Power of $(\sigma^2)$ | Why |
|--------|----------------------|-----|
| Likelihood | $-n/2$ | Product of $n$ Gaussians, each contributes $(\sigma^2)^{-1/2}$ |
| Prior on $\boldsymbol{\beta}$ | $-p/2$ | One $p$-dimensional Gaussian contributes $(\sigma^2)^{-p/2}$ from the determinant $\|\sigma^2 L_0^{-1}\|^{-1/2}$ |
| Prior on $\sigma^2$ | $-\nu_0/2 - 1$ | Directly from the $\chi^{-2}$ density formula |
| **Total** | $-n/2 - p/2 - \nu_0/2 - 1$ | Sum of the above |

Now match the two expressions:

$$
-\frac{\nu_n}{2} - 1 - \frac{p}{2} = -\frac{n}{2} - \frac{p}{2} - \frac{\nu_0}{2} - 1
$$

The $-p/2$ and $-1$ cancel from both sides:

$$
-\frac{\nu_n}{2} = -\frac{n}{2} - \frac{\nu_0}{2}
$$

$$
\boxed{\nu_n = \nu_0 + n}
$$

**Interpretation:** The posterior degrees of freedom equals the prior degrees of freedom plus the number of data points. Each observation contributes one "degree of freedom" to our knowledge of $\sigma^2$, just like how $n$ data points give $n$ degrees of freedom in classical statistics.

---

### Step 3: Collect Terms in $\boldsymbol{\beta}$, Find $L_n$ and $\mathbf{m}_n$ (DETAILED)

Recall from Step 1, the entire exponent (everything inside $\exp(\ldots)$) is:

$$
-\frac{1}{2\sigma^2}\Big[
(\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})
+ (\boldsymbol{\beta} - \mathbf{m}_0)^T L_0 (\boldsymbol{\beta} - \mathbf{m}_0)
+ \nu_0 \tau_0^2
\Big]
$$

We need to expand the two quadratic forms and sort every term into three buckets: quadratic in $\boldsymbol{\beta}$, linear in $\boldsymbol{\beta}$, and constant (no $\boldsymbol{\beta}$).

---

**Expanding the likelihood term** $(\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})$:

This is just $(a - b)^T(a-b) = a^Ta - 2a^Tb + b^Tb$ with $a = \mathbf{y}$ and $b = X\boldsymbol{\beta}$:

$$
= \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^TX\boldsymbol{\beta} + \boldsymbol{\beta}^TX^TX\boldsymbol{\beta}
$$

Note: $\mathbf{y}^TX\boldsymbol{\beta}$ is a scalar, so it equals its own transpose: $\mathbf{y}^TX\boldsymbol{\beta} = \boldsymbol{\beta}^TX^T\mathbf{y}$. So we can write:

$$
= \underbrace{\boldsymbol{\beta}^T X^TX \boldsymbol{\beta}}_{\text{quadratic}} - \underbrace{2\boldsymbol{\beta}^T X^T\mathbf{y}}_{\text{linear}} + \underbrace{\mathbf{y}^T\mathbf{y}}_{\text{constant}}
$$

---

**Expanding the prior term** $(\boldsymbol{\beta} - \mathbf{m}_0)^T L_0 (\boldsymbol{\beta} - \mathbf{m}_0)$:

This is the weighted version: $(a-b)^T W (a-b) = a^TWa - 2a^TWb + b^TWb$, with $a = \boldsymbol{\beta}$, $b = \mathbf{m}_0$, $W = L_0$:

$$
= \boldsymbol{\beta}^T L_0 \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_0 \mathbf{m}_0 + \mathbf{m}_0^T L_0 \mathbf{m}_0
$$

(Here we used that $L_0$ is symmetric, so $\boldsymbol{\beta}^T L_0 \mathbf{m}_0 = \mathbf{m}_0^T L_0 \boldsymbol{\beta}$.)

$$
= \underbrace{\boldsymbol{\beta}^T L_0 \boldsymbol{\beta}}_{\text{quadratic}} - \underbrace{2\boldsymbol{\beta}^T L_0\mathbf{m}_0}_{\text{linear}} + \underbrace{\mathbf{m}_0^TL_0\mathbf{m}_0}_{\text{constant}}
$$

---

**The prior on $\sigma^2$ term** contributes $\nu_0\tau_0^2$ (constant -- no $\boldsymbol{\beta}$).

---

**Now group everything into the three buckets:**

| Bucket | From likelihood | From prior on $\boldsymbol{\beta}$ | From prior on $\sigma^2$ | Combined |
|--------|----------------|-------------------------------------|--------------------------|----------|
| Quadratic in $\boldsymbol{\beta}$ | $\boldsymbol{\beta}^T X^TX \boldsymbol{\beta}$ | $\boldsymbol{\beta}^T L_0 \boldsymbol{\beta}$ | -- | $\boldsymbol{\beta}^T(X^TX + L_0)\boldsymbol{\beta}$ |
| Linear in $\boldsymbol{\beta}$ | $-2\boldsymbol{\beta}^T X^T\mathbf{y}$ | $-2\boldsymbol{\beta}^T L_0\mathbf{m}_0$ | -- | $-2\boldsymbol{\beta}^T(X^T\mathbf{y} + L_0\mathbf{m}_0)$ |
| Constant (no $\boldsymbol{\beta}$) | $\mathbf{y}^T\mathbf{y}$ | $\mathbf{m}_0^TL_0\mathbf{m}_0$ | $\nu_0\tau_0^2$ | $\mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2$ |

So the full exponent is:

$$
-\frac{1}{2\sigma^2}\Big[
\boldsymbol{\beta}^T\underbrace{(X^TX + L_0)}_{A}\boldsymbol{\beta}
- 2\boldsymbol{\beta}^T\underbrace{(X^T\mathbf{y} + L_0\mathbf{m}_0)}_{\mathbf{b}}
+ \underbrace{\mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2}_{c}
\Big]
$$

---

**Identifying $L_n$ and $\mathbf{m}_n$:**

We need the $\boldsymbol{\beta}$-dependent part to look like the kernel of $\mathcal{N}(\boldsymbol{\beta}; \mathbf{m}_n, \sigma^2 L_n^{-1})$, which is:

$$
-\frac{1}{2\sigma^2}(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)
= -\frac{1}{2\sigma^2}\Big[\boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n \mathbf{m}_n + \mathbf{m}_n^T L_n \mathbf{m}_n\Big]
$$

Matching the quadratic coefficient of $\boldsymbol{\beta}$:

$$
A = L_n \quad\Longrightarrow\quad \boxed{L_n = X^TX + L_0}
$$

Matching the linear coefficient of $\boldsymbol{\beta}$:

$$
\mathbf{b} = L_n \mathbf{m}_n \quad\Longrightarrow\quad L_n \mathbf{m}_n = X^T\mathbf{y} + L_0\mathbf{m}_0
$$

Multiply both sides on the left by $L_n^{-1}$:

$$
\boxed{\mathbf{m}_n = L_n^{-1}(X^T\mathbf{y} + L_0\mathbf{m}_0) = (X^TX + L_0)^{-1}(X^T\mathbf{y} + L_0\mathbf{m}_0)}
$$

**Interpretation of $L_n$:** The posterior precision = prior precision + data precision. More data (larger $X^TX$) means more precise estimates of $\boldsymbol{\beta}$.

**Interpretation of $\mathbf{m}_n$:** The posterior mean is a precision-weighted average of what the data says ($X^T\mathbf{y}$, pointing toward the OLS solution) and what the prior says ($L_0\mathbf{m}_0$, pointing toward $\mathbf{m}_0$). Whichever has higher precision "wins."

---

### Step 4: Complete the Square to Find $\tau_n^2$ (DETAILED)

After Step 3, we identified the $\boldsymbol{\beta}$-part. But when we wrote $\boldsymbol{\beta}^T A \boldsymbol{\beta} - 2\boldsymbol{\beta}^T\mathbf{b}$ as $(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)$, we introduced an extra constant term. Let's track this carefully.

**The completing-the-square identity** (given as Hint 2 in the problem):

$$
\mathbf{a}^T B \mathbf{a} - 2\mathbf{a}^T B\mathbf{c} = (\mathbf{a} - \mathbf{c})^T B (\mathbf{a} - \mathbf{c}) - \mathbf{c}^T B \mathbf{c}
$$

**Verify this is correct** (expand the RHS):

$$
(\mathbf{a}-\mathbf{c})^T B (\mathbf{a}-\mathbf{c}) - \mathbf{c}^TB\mathbf{c}
= \mathbf{a}^TB\mathbf{a} - 2\mathbf{a}^TB\mathbf{c} + \mathbf{c}^TB\mathbf{c} - \mathbf{c}^TB\mathbf{c}
= \mathbf{a}^TB\mathbf{a} - 2\mathbf{a}^TB\mathbf{c} \;\checkmark
$$

**Apply with** $\mathbf{a} = \boldsymbol{\beta}$, $B = L_n$, $\mathbf{c} = \mathbf{m}_n$:

$$
\boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n \mathbf{m}_n
= \underbrace{(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)}_{\text{depends on } \boldsymbol{\beta}}
- \underbrace{\mathbf{m}_n^T L_n \mathbf{m}_n}_{\text{leftover constant}}
$$

---

**Now reassemble the full exponent.** From Step 3, the exponent was:

$$
-\frac{1}{2\sigma^2}\Big[
\underbrace{\boldsymbol{\beta}^T L_n \boldsymbol{\beta} - 2\boldsymbol{\beta}^T L_n\mathbf{m}_n}_{\text{the } \boldsymbol{\beta} \text{ terms}}
+ \underbrace{\mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2}_{\text{the constants from Step 3}}
\Big]
$$

Substituting the completed square:

$$
= -\frac{1}{2\sigma^2}\Big[
(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)
- \mathbf{m}_n^T L_n \mathbf{m}_n
+ \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 + \nu_0\tau_0^2
\Big]
$$

Rearrange into two groups:

$$
= -\frac{1}{2\sigma^2}\underbrace{(\boldsymbol{\beta} - \mathbf{m}_n)^T L_n (\boldsymbol{\beta} - \mathbf{m}_n)}_{\text{Part A: goes into the Gaussian on } \boldsymbol{\beta}}
\;-\; \frac{1}{2\sigma^2}\underbrace{\Big(\nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n\Big)}_{\text{Part B: goes into the inv-chi-squared on } \sigma^2}
$$

---

**Now combine Part B with the $(\sigma^2)$ power from Step 2.**

The full posterior (putting power and exponent together) is:

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \cdots) \propto
(\sigma^2)^{-\nu_n/2 - 1 - p/2}
\exp\!\left(
-\frac{1}{2\sigma^2}\Big[\text{Part A} + \text{Part B}\Big]
\right)
$$

We want this to factor as $\chi^{-2}(\sigma^2) \times \mathcal{N}(\boldsymbol{\beta})$:

$$
= \underbrace{(\sigma^2)^{-\nu_n/2 - 1}\exp\!\left(-\frac{\text{Part B}}{2\sigma^2}\right)}_{\chi^{-2}(\sigma^2 \mid \nu_n,\, \tau_n^2)}
\times
\underbrace{(\sigma^2)^{-p/2}\exp\!\left(-\frac{\text{Part A}}{2\sigma^2}\right)}_{\mathcal{N}(\boldsymbol{\beta} \mid \mathbf{m}_n,\, \sigma^2 L_n^{-1})}
$$

---

**Identifying $\tau_n^2$:**

The $\chi^{-2}(\sigma^2 \mid \nu_n, \tau_n^2)$ density has the exponent $\exp\!\left(-\frac{\nu_n\tau_n^2}{2\sigma^2}\right)$.

Matching with Part B:

$$
\nu_n \tau_n^2 = \nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n
$$

Divide both sides by $\nu_n = \nu_0 + n$:

$$
\boxed{\tau_n^2 = \frac{1}{\nu_0 + n}\Big(\nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n\Big)}
$$

---

**Interpretation of $\tau_n^2$:** It's a weighted average of:
- $\tau_0^2$: your prior guess for $\sigma^2$ (weighted by $\nu_0$)
- A data-driven quantity (weighted by $n$)

The $\mathbf{y}^T\mathbf{y} + \mathbf{m}_0^TL_0\mathbf{m}_0 - \mathbf{m}_n^TL_n\mathbf{m}_n$ term can be rewritten (with some algebra) as:

$$
(\mathbf{y} - X\mathbf{m}_n)^T(\mathbf{y} - X\mathbf{m}_n) + (\mathbf{m}_n - \mathbf{m}_0)^TL_0(\mathbf{m}_n - \mathbf{m}_0)
$$

This alternative form has a nice interpretation:
- First part: residual sum of squares evaluated at the posterior mean $\mathbf{m}_n$
- Second part: a penalty for how far the posterior mean moved from the prior mean

---

**Sanity check -- does the factorization work?**

Putting everything together:

$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0)
\propto
\underbrace{(\sigma^2)^{-\nu_n/2-1}\exp\!\left(-\frac{\nu_n\tau_n^2}{2\sigma^2}\right)}_{\chi^{-2}(\sigma^2;\,\nu_n,\tau_n^2)}
\times
\underbrace{(\sigma^2)^{-p/2}\exp\!\left(-\frac{(\boldsymbol{\beta}-\mathbf{m}_n)^TL_n(\boldsymbol{\beta}-\mathbf{m}_n)}{2\sigma^2}\right)}_{\mathcal{N}(\boldsymbol{\beta};\,\mathbf{m}_n,\,\sigma^2 L_n^{-1})}
$$

This is exactly the NIX form. The posterior is $\text{NIX}(\mathbf{m}_n, L_n, \nu_n, \tau_n^2)$.

---

### Final Answer (Summary)

The posterior is:
$$
p(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}, X, \boldsymbol{\eta}_0) = \text{NIX}(\boldsymbol{\beta}, \sigma^2;\; \mathbf{m}_n, L_n, \nu_n, \tau_n^2)
$$

where:

$$
\nu_n = \nu_0 + n
$$

$$
L_n = L_0 + X^TX
$$

$$
\mathbf{m}_n = L_n^{-1}(L_0 \mathbf{m}_0 + X^T\mathbf{y})
$$

$$
\tau_n^2 = \frac{1}{\nu_n}\left(\nu_0\tau_0^2 + \mathbf{y}^T\mathbf{y} + \mathbf{m}_0^T L_0 \mathbf{m}_0 - \mathbf{m}_n^T L_n \mathbf{m}_n\right)
$$
