# Latent-Regime Block Conformal Prediction for Synthetic Control

**Andrew Wang** · University of Washington, Department of Political Science

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuanzhouwang/ConformalHMM/blob/main/latent_regime_conformal_main.ipynb)

---

## Overview

This project proposes a method for uncertainty quantification in synthetic control estimators when the prediction residuals are heteroscedastic — that is, when their variance shifts over time due to latent economic or political regimes.

Standard conformal inference for synthetic control (Chernozhukov et al., 2021) achieves exact finite-sample coverage by treating the residuals from a first-stage predictor as exchangeable across time. This assumption breaks down whenever residuals are only *conditionally* stable — for example, if a time series passes through distinct volatility regimes (calm, turbulent, crisis) driven by an unobserved Markov process. In that case, the standard permutation test may reject the true counterfactual path because it compares post-treatment residuals to a calm pre-treatment period, not because the candidate path is wrong.

**The key insight** is that the residual sequence, together with the latent regime at each time period, can be modeled as a Hidden Markov Model (HMM). The method of Nettasinghe et al. (2023) constructs exchangeable *blocks* from HMM sequences, enabling valid conformal inference even under temporal dependence. This project adapts that construction to counterfactual inference: instead of predicting latent states, the HMM structure is used to test whether a candidate untreated path produces residuals consistent with the pre-treatment regime-conditional distribution.

A key technical contribution is the **discretize-for-blocks, score-on-continuity** separation: residuals are discretized only to identify the correct exchangeable block partition (making the continuous residual sequence compatible with the discrete-emission HMM framework), while conformity scores are evaluated on the original continuous residuals to avoid information loss.

---

## Method Summary

Let $\hat{u}_t(y)$ be the prediction residual at time $t$ under candidate untreated path $y$, and let $Z_t \in \{1, \ldots, K\}$ be the latent regime. The procedure:

1. **First-stage prediction.** Fit a synthetic control predictor (e.g., Generalized Synthetic Control) to obtain residuals $\hat{u}_{1:T}(y)$.
2. **Learn a residual partition** $b(\cdot)$ from pre-treatment residuals only (e.g., quantile bins).
3. **Discretize** residuals to $\tilde{u}_t(y) = b(\hat{u}_t(y))$ and form paired sequences $\{(Z_t, \tilde{u}_t(y))\}$.
4. **Partition into exchangeable blocks** using the $(Z_t, \tilde{u}_t)$ template structure.
5. **Permute blocks** and score on the continuous residuals to obtain the conformal $p$-value.
6. **Accept** candidate path $y$ if its conformity score does not exceed the permutation quantile.

Exact finite-sample coverage follows from the exchangeability of blocks under the latent Markov structure, conditional on the known regime path.

---

## Repository Structure

```
ConformalHMM/
├── latent_regime_conformal_main.ipynb   # Main simulation: Algorithm 1 vs. Chernozhukov baseline
├── hmm_conformal_batched.ipynb          # Batched GPU implementation and additional experiments
└── ConformalHMM_Manuscript/
    └── main.tex                         # Paper
```

---

## Simulation Design

The Monte Carlo study simulates a panel with 20 donor units and a single treated unit over 36 time periods (24 pre-treatment, 12 post-treatment). The treated unit's residuals are drawn from a 3-state Markov regime process with state-dependent noise standard deviations of 2.0, 5.5, and 8.5 — a substantial range of heteroscedasticity. The first-stage predictor is Generalized Synthetic Control (Xu, 2017) fitted via `gsynth` in R.

Both methods are evaluated on 500 candidate untreated paths per dataset, including the true path. The target miscoverage level is $\alpha = 0.10$.

**Key results:** The latent-regime block method maintains coverage close to the nominal $1 - \alpha = 0.90$ target across simulations, while the Chernozhukov-style global permutation baseline is miscalibrated when regime-switching induces heteroscedasticity.

---

## Technical Stack

| Component | Tools |
|---|---|
| First-stage estimator | R (`gsynth`), called from Python via `rpy2` |
| Numerical computation | NumPy, PyTorch (GPU-batched conformity scoring) |
| Discretization | scikit-learn `KBinsDiscretizer` |
| Environment | Google Colab (CUDA GPU) |

---

## References

- Abadie, Diamond, Hainmueller (2010). Synthetic Control Methods for Comparative Case Studies.
- Chernozhukov, Wüthrich, Zhu (2021). An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls.
- Nettasinghe, Krishnamurthy, Swami (2023). Conformal Prediction for Hidden Markov Models.
- Xu (2017). Generalized Synthetic Control Method.
