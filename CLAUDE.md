# STAT 348 -- Modern Applied Statistics (UChicago, Spring 2026)

## Course Overview
Bayesian probabilistic modeling and inference. Taught by Prof. Schein.
Homework is in Jupyter notebooks (Python + PyTorch) with math derivations.
HW1 is due Sunday April 5, 11:59pm on GradeScope.

## Lecture Notes Summary

### Lecture 1: Foundations
- **Bayes' rule**: posterior = likelihood * prior / evidence
- **Posterior odds** = likelihood ratio * prior odds (base rates matter!)
- **Decision theory**: optimal decision minimizes expected loss under the posterior
- **Discriminative vs generative models**: discriminative models P(y|x) directly; generative models P(y) and P(x|y)
- **Naive Bayes**: assumes conditional independence of features given label
- **Conjugacy**: when prior and posterior are in the same family
  - Beta-Bernoulli: posterior Beta(alpha + sum(x_i), beta + sum(1-x_i))
  - Posterior mean is weighted combo of prior mean and MLE
- **Laplace smoothing** = adding pseudocounts = using a Beta prior
- **Dutch book argument**: coherent beliefs must obey probability axioms (de Finetti)

### Lecture 2: Bayesian Linear Regression (sigma^2 fixed)
- Model: y = X*beta + epsilon, epsilon ~ N(0, sigma^2 I)
- Prior: beta ~ N(m_0, L_0^{-1}), with sigma^2 fixed/known
- **Posterior** (Gaussian-Gaussian conjugacy):
  - Posterior precision: L_n = L_0 + (1/sigma^2) X^T X
  - Posterior mean: m_n = L_n^{-1}(L_0 m_0 + (1/sigma^2) X^T y)
- MAP = posterior mode = posterior mean (for Gaussian)
- Uninformative prior (large variance) -> MAP = OLS = (X^T X)^{-1} X^T y
- Prior with L_0^{-1} = (sigma^2/alpha)*I, m_0=0 -> MAP = ridge regression
- **Priors act as regularizers** (Gaussian prior = L2, Laplace prior = L1)
- **Posterior predictive**: y_{n+1} | x_{n+1}, data ~ N(x_{n+1}^T m_n, x_{n+1}^T L_n^{-1} x_{n+1} + sigma^2)
- **Marginal likelihood** for model evaluation/selection; Bayes factor rewards simplicity

### Lecture 3: Hierarchical Models & Conjugate Priors for Variance
- **8 schools example**: hierarchical model with school means drawn from shared prior
- **Sufficient statistics**: posterior depends on data only through sufficient stats (e.g., sample mean, sample size)
- **Shrinkage estimators**: posterior mean shrinks local estimates toward global mean
  - tau^2 -> infinity: no pooling (each school independent)
  - tau^2 -> 0: complete pooling (all schools share one mean)
- **Empirical Bayes** = type-II MLE: set hyperparameters by maximizing marginal likelihood
- **Conjugate prior for variance** (known mean):
  - Scaled inverse chi-squared: sigma^2 ~ chi^{-2}(nu_0, tau_0^2)
  - Equivalent to InvGamma(nu_0/2, nu_0*tau_0^2/2)
  - Mean: E[sigma^2] = nu_0/(nu_0-2) * tau_0^2 (for nu_0 > 2)
  - Posterior: chi^{-2}(nu_0 + M, tau_M^2) where tau_M^2 weighted avg of prior scale and data variance
- **NIX distribution** = Normal-Inverse-Chi-Squared = conjugate prior for Gaussian with unknown mean AND variance
  - NIX(mu, sigma^2; m_0, kappa_0, nu_0, tau_0^2) = chi^{-2}(sigma^2; nu_0, tau_0^2) * N(mu; m_0, sigma^2/kappa_0)
- **PGMs**: graphical models encode conditional independence via directed graphs
  - Joint factorizes according to parent structure
  - Markov blanket: parents + children + co-parents
- **Gibbs sampling preview**: sample each latent variable from its complete conditional

### Lecture 4: Gibbs Sampling & MCMC
- **Gibbs sampler**: Markov chain that updates one variable at a time from complete conditionals
- **Markov chains**: stationary distribution, ergodicity, detailed balance
- **Convergence**: burn-in period, then collect samples; thin to reduce autocorrelation
- **Monte Carlo estimates**: unbiased, RMSE = O(1/sqrt(M)) regardless of dimension
- **Missing data**: treat as latent variables in the Gibbs sampler
- **Geweke testing**: forward sampler vs backward sampler to verify MCMC correctness

### Linderman Lecture 01 Slides (Stanford STATS 305C reference)
- Thorough derivation of normal model with unknown mean, unknown precision, and both
- **Chi-squared family**:
  - chi^2(nu_0) = Gamma(nu_0/2, 1/2) -- for precision
  - Scaled chi^2(nu_0, lambda_0) -- adding scale parameter
  - Scaled inverse chi^{-2}(nu_0, sigma_0^2) = InvGamma(nu_0/2, nu_0*sigma_0^2/2) -- for variance
- **NIX posterior** (scalar case): NIX(mu_N, kappa_N, nu_N, sigma_N^2) where
  - kappa_N = kappa_0 + N
  - nu_N = nu_0 + N
  - mu_N = (kappa_0*mu_0 + sum(x_n)) / kappa_N
  - sigma_N^2 = (1/nu_N)(nu_0*sigma_0^2 + kappa_0*mu_0^2 + sum(x_n^2) - kappa_N*mu_N^2)
- **Posterior marginals**:
  - sigma^2 marginal is chi^{-2}(nu_N, sigma_N^2)
  - mu marginal is Student's t: St(nu_N, mu_N, sigma_N^2/kappa_N)
- **Student's t arises** from integrating out sigma^2 from a Gaussian-inverse-chi-squared

## Key Formulas Quick Reference

| Concept | Formula |
|---------|---------|
| OLS | beta_hat = (X^T X)^{-1} X^T y |
| Hat matrix | H = X(X^T X)^{-1} X^T |
| RSS | y^T(I - H)y |
| Gaussian kernel | exp(-1/2 * (x-mu)^T Sigma^{-1} (x-mu)) |
| Complete the square | a^T B a - 2a^T Bc = (a-c)^T B (a-c) - c^T B c |
| Inv chi-sq mean | E[sigma^2] = nu/(nu-2) * tau^2, for nu > 2 |

## Tech Stack
- Python, PyTorch (tensors, distributions)
- Jupyter notebooks
- Key PyTorch: torch.linalg.solve, @ for matmul, .T for transpose, torch.distributions
