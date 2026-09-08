# PROJECT_HANDOFF.md

## 1. Project Identity

**Project title**

> **Derivatives Pricing and Model Calibration: A Comparative Study of Analytical, Simulation, PDE and Stochastic Volatility Methods**

**GitHub repository**

`https://github.com/liufangyu2025-ui/derivatives-pricing-calibration`

**Local path**

`/Users/amber/derivatives-pricing-calibration`

**Primary purpose**

This is a research-oriented quantitative finance application project. It should **not** be simplified into a basic “option pricing calculator”.

The project is intended to support applications to Quantitative Finance / Computational Finance programs and demonstrate:

- mathematical understanding;
- numerical methods;
- model validation;
- error diagnosis;
- stochastic-volatility modelling;
- calibration;
- model comparison;
- reproducible research practice.

---

## 2. Core Research Logic

The central project progression is:

```text
Pricing
→ Validation
→ Diagnosis
→ Extension
→ Calibration
→ Comparison
```

More specifically:

```text
Analytical benchmark
→ Numerical pricing
→ Numerical validation / error
→ Implied-volatility diagnostics
→ Black-Scholes model limitation
→ Heston stochastic volatility
→ Heston pricing
→ Heston calibration
→ Black-Scholes vs Heston comparison
```

The most important conceptual distinction in the project is:

> **Numerical Error vs Model Error**

### Numerical error

Examples already studied:

- Monte Carlo sampling error;
- PDE discretization error;
- Heston Monte Carlo time-discretization error;
- Fourier integration / truncation error.

### Model error

Main direction:

- Black-Scholes assumes constant volatility;
- a single constant volatility cannot reproduce a non-flat implied-volatility surface;
- Heston introduces stochastic variance and stock-variance correlation;
- later comparison should evaluate whether Heston reduces structured pricing / IV residuals.

A potential paper title is:

> **From Numerical Error to Model Error: A Comparative Study of Option Pricing and Stochastic Volatility Calibration**

---

## 3. User / Working Style

The user is a finance undergraduate preparing for Quantitative Finance / Computational Finance applications.

When continuing this project:

- explain in **Chinese**;
- keep the project academically rigorous but beginner-friendly;
- explain **why** a method is used before asking the user to copy code;
- give exact Cursor / VS Code steps when needed;
- explicitly label notebook content as **Markdown Cell** and **Code Cell**;
- in Jupyter notebooks, use `$...$` and `$$...$$` for formulas;
- when explaining formulas in chat, **give LaTeX source first, then rendered formula**;
- distinguish stochastic sampling error from systematic numerical bias;
- do not overclaim from one numerical experiment;
- do not claim parameter uniqueness merely because calibration RMSE is low;
- prefer phrases such as “under the parameter configurations examined here” when appropriate;
- before ending each technical day:
  - run tests;
  - `Restart Kernel → Run All`;
  - inspect outputs / figures;
  - commit and push to GitHub.

Do **not** redesign Days 1–7 unless a bug must be fixed.

---

## 4. Environment

**OS / editor**

- macOS
- Cursor / VS Code
- Jupyter

**Virtual environment**

`/Users/amber/derivatives-pricing-calibration/.venv/bin/python`

**Python**

`3.13.9`

**Jupyter kernel**

`.venv (Python 3.13.9)`

**pytest config**

`pytest.ini`

```ini
[pytest]
pythonpath = .
```

**Latest validated test count**

> **33 passed**

This was confirmed after Day 7 and the additional calibration tests.

---

## 5. Repository Structure

```text
derivatives-pricing-calibration/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── data/
├── figures/
├── notebooks/
│   ├── 01_black_scholes_greeks.ipynb
│   ├── 02_monte_carlo.ipynb
│   ├── 03_finite_difference.ipynb
│   ├── 04_implied_volatility.ipynb
│   ├── 05_heston_model.ipynb
│   ├── 06_heston_pricing.ipynb
│   └── 07_heston_calibration.ipynb
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── black_scholes.py
│   │   └── heston.py
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── monte_carlo.py
│   │   ├── finite_difference.py
│   │   └── heston_pricing.py
│   ├── volatility/
│   │   └── implied_volatility.py
│   └── calibration/
│       ├── __init__.py
│       └── heston_calibration.py
└── tests/
    ├── test_black_scholes.py
    ├── test_monte_carlo.py
    ├── test_finite_difference.py
    ├── test_implied_volatility.py
    ├── test_heston.py
    ├── test_heston_pricing.py
    ├── test_heston_calibration.py
    └── test_heston_calibration_extra.py
```

---

# 6. Completed Work

## Day 1 — Black-Scholes & Greeks

Completed:

- European Black-Scholes call / put pricing;
- analytical benchmark;
- Greeks;
- finite-difference validation of Greeks;
- unit tests.

Representative benchmark:

```text
Black-Scholes call ≈ 10.450583572185565
```

Research purpose:

> establish an analytical benchmark before introducing numerical methods.

---

## Day 2 — Monte Carlo Pricing

Completed:

- risk-neutral GBM simulation;
- European call and put Monte Carlo pricing;
- standard error;
- confidence interval;
- antithetic variates;
- repeated simulation logic;
- sampling-error analysis.

Core idea:

```text
Monte Carlo error
≈ sampling uncertainty
```

---

## Day 3 — Finite Difference / Crank-Nicolson

Completed:

- Black-Scholes PDE;
- Crank-Nicolson scheme;
- call and put pricing;
- comparison with analytical Black-Scholes;
- grid refinement;
- convergence analysis;
- comparison with Monte Carlo;
- repeated Monte Carlo experiment;
- robust error comparison.

### Important recent fix

The Day 3 notebook previously contained a duplicate comparison cell that used `mc_rmse` **before `mc_rmse` was defined**, producing:

```text
NameError: name 'mc_rmse' is not defined
```

The incorrect duplicate cell was removed.

Correct logical order:

```text
Repeated Monte Carlo
→ define mc_rmse
→ robust cross-method comparison
→ use mc_rmse
```

The corrected notebook should be stored as:

`notebooks/03_finite_difference.ipynb`

**If the corrected notebook has not yet been committed after the fix, commit and push it before final project delivery.**

Suggested commit message:

```bash
git commit -m "Fix finite difference notebook execution order"
```

---

## Day 4 — Implied Volatility Diagnostics

File:

`src/volatility/implied_volatility.py`

Solver:

```python
implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    sigma_lower=1e-8,
    sigma_upper=5.0,
    tol=1e-10,
)
```

Method:

- Brent root finding (`scipy.optimize.brentq`);
- no-arbitrage bounds;
- call / put implied volatility;
- synthetic volatility skew / surface;
- constant-volatility Black-Scholes fit diagnostics.

Research transition:

```text
Black-Scholes numerical pricing can be accurate
but
constant volatility can still be a poor model of a non-flat IV surface
```

This is the project’s transition from **numerical error** to **model specification error**.

---

## Day 5 — Heston Dynamics

File:

`src/models/heston.py`

Key functions:

```python
feller_condition(...)
expected_variance(...)
simulate_heston_paths(...)
```

Simulation uses:

- positivity-truncated Euler approximation for the variance process;
- log-Euler update for the stock process;
- correlated Brownian shocks.

Important wording:

> Do not automatically call the implementation “full truncation Euler”; the code is described as a positivity-truncated Euler approximation.

Risk-neutral dynamics:

```latex
dS_t=rS_tdt+\sqrt{v_t}S_t\,dW_t^S
```

```latex
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW_t^v
```

```latex
dW_t^S dW_t^v=\rho\,dt
```

Parameters:

```text
v0
kappa
theta
xi
rho
```

Feller sufficient condition:

```latex
2\kappa\theta>\xi^2
```

Important interpretation:

- it is sufficient for strict positivity in the continuous-time process;
- failure of the Feller condition does not automatically invalidate the Heston model;
- discrete Euler simulation can still reach zero and therefore truncation is used.

---

## Day 6 — Heston Semi-Analytical Pricing & Numerical Validation

File:

`src/pricing/heston_pricing.py`

Main functions:

```python
heston_characteristic_function(...)
heston_call_price(...)
heston_put_price(...)
heston_monte_carlo_price(...)
```

### Characteristic function checks

```latex
\phi(0)=1
```

```latex
\phi(-i)=\mathbb E[S_T]=S_0e^{rT}
```

These passed.

### Semi-analytical pricing

Call representation:

```latex
C_0=S_0P_1-Ke^{-rT}P_2
```

Put obtained by parity:

```latex
P=C-S_0+Ke^{-rT}
```

### Baseline Heston parameters

```text
S0     = 100
K      = 100
T      = 1.0
r      = 0.05
v0     = 0.04
kappa  = 2.0
theta  = 0.04
xi     = 0.30
rho    = -0.70
```

Baseline Heston call:

```text
10.3942185523
```

Baseline Heston put:

```text
5.5171610024
```

### Fourier integration stability

```text
L=25   → 10.3919173684
L=50   → 10.3942132039
L=75   → 10.3942185540
L=100  → 10.3942185523
L=150  → 10.3942185523
L=200  → 10.3942185523
```

Conclusion:

> For the baseline configuration, the semi-analytical price is effectively stable by approximately `L=75–100`; default `integration_limit=150` is conservative.

### Fourier vs Heston Monte Carlo

```text
Fourier Price:          10.3942185523
Monte Carlo Price:      10.3785384398
MC Standard Error:       0.0868564224
95% CI:                 [10.20829985, 10.54877703]
MC - Fourier:           -0.0156801125
Difference in SE units: -0.180529
```

Interpretation:

> The independent Monte Carlo estimate is statistically consistent with the Fourier benchmark.

Important caveat:

The MC interval quantifies sampling uncertainty but not Euler time-discretization bias.

### Monte Carlo time-step experiment

Tested:

```text
n_steps = 63, 126, 252, 504, 1008
```

with repeated seeds.

Observed mean biases were small relative to sampling variability. No clear monotonic discretization pattern could be detected at the current simulation precision.

Key conclusion:

> Once the grid is sufficiently fine, remaining discretization effects can be smaller than Monte Carlo sampling noise.

### Heston-generated implied volatility

Day 6 also completed:

- Heston prices across strikes;
- inversion into Black-Scholes implied volatility;
- downward IV skew under negative `rho`;
- `rho` sensitivity;
- `xi` sensitivity;
- multiple maturities.

This closes the Day 4 → Day 5 → Day 6 logic:

```text
Black-Scholes IV skew diagnostic
→ stochastic variance model
→ Heston option prices
→ Heston-generated non-flat Black-Scholes IV
```

---

## Day 7 — Heston Calibration

File:

`src/calibration/heston_calibration.py`

Main calibration structure:

```text
Observed option prices
→ candidate Heston parameters
→ Heston Fourier prices
→ residuals
→ nonlinear least squares
→ calibrated parameters
```

Parameter vector:

```latex
\boldsymbol{\psi}
=
(v_0,\kappa,\theta,\xi,\rho)
```

Objective:

```latex
\hat{\boldsymbol{\psi}}
=
\arg\min_{\boldsymbol{\psi}\in\mathcal B}
\sum_{i=1}^{N}
w_i
\left[
C_{\mathrm{Heston}}
(K_i,T_i;\boldsymbol{\psi})
-
C_i^{\mathrm{obs}}
\right]^2
```

### Controlled synthetic calibration

True parameters:

```text
v0     = 0.04
kappa  = 2.00
theta  = 0.04
xi     = 0.30
rho    = -0.70
```

Synthetic surface:

```text
5 strikes × 4 maturities = 20 options
```

Strikes:

```text
80, 90, 100, 110, 120
```

Maturities:

```text
0.25, 0.50, 1.00, 2.00
```

First calibration result:

```text
Success: True
Function evaluations: 12
RMSE ≈ 4.62e-11
MAE  ≈ 3.60e-11
```

The true parameter vector was recovered to approximately machine precision under the ideal noise-free synthetic experiment.

### Multiple-start calibration

Completed:

- several materially different initial guesses;
- optimization-stability comparison;
- RMSE comparison;
- parameter comparison.

Purpose:

```text
Different initial guesses
→ same optimum?
→ local-minimum / optimization-stability diagnostic
```

### Noisy synthetic market

Noise was introduced in **implied-volatility space**, then converted back to Black-Scholes prices.

Completed:

- noisy IV surface;
- noisy price calibration;
- multiple-start calibration;
- price residuals;
- IV residuals;
- price-fit vs parameter-recovery comparison.

### Identifiability / Jacobian diagnostics

Completed:

- normalized Jacobian;
- singular values;
- local identifiability diagnostics;
- comparison of noise-free vs noisy calibration stability.

Central Day 7 conclusion:

> **Good price fit ≠ perfect parameter recovery ≠ unique parameter identification**

This distinction must be preserved in later writing.

---

## 7. Current Test Status

Latest verified command:

```bash
pytest -v
```

Result:

```text
33 passed
```

Current tests cover:

- Black-Scholes;
- Monte Carlo;
- finite difference;
- implied volatility;
- Heston simulation;
- Heston semi-analytical pricing;
- Heston Monte Carlo pricing;
- Heston calibration;
- weighted residual logic;
- validation that non-positive calibration weights raise errors.

---

# 8. Remaining Work

There are **three remaining days**.

## Day 8 — Model Comparison & Realistic Calibration

Day 8 is the **final major technical day**.

### Main research question

> When numerical pricing error has been controlled, how much of the remaining cross-sectional option-pricing error is due to model specification?

Day 8 should explicitly compare:

```text
Black-Scholes
vs
Heston
```

on the **same realistic / quasi-market option surface**.

### Main goals

- create or use a realistic option-surface dataset;
- fit a constant-volatility Black-Scholes benchmark;
- calibrate Heston to the same observations;
- compare:
  - price RMSE;
  - MAE;
  - implied-volatility RMSE;
  - residuals by strike;
  - residuals by maturity;
  - systematic residual structure;
- explain which errors are:
  - numerical;
  - sampling-related;
  - model-specification-related;
  - calibration-related.

### Desired Day 8 conclusion

```text
Analytical / numerical validation
→ numerical errors quantified
→ Black-Scholes IV mismatch
→ Heston stochastic-volatility extension
→ calibration
→ model comparison
```

Day 8 should **not** introduce another major model unless absolutely necessary.

---

## Day 9 — Project Integration

No major new quantitative model.

Goals:

- inspect all notebooks;
- `Restart Kernel → Run All`;
- remove duplicate / broken / unnecessary cells;
- verify every figure;
- verify tests;
- clean data outputs;
- check repository structure;
- rewrite README;
- explain motivation, methods, research questions, architecture, key results, model comparison, limitations, and reproducibility;
- make GitHub suitable for direct review by admissions readers.

README should communicate:

```text
Why this problem?
→ What methods?
→ What was validated?
→ What failed under Black-Scholes?
→ Why Heston?
→ How was calibration validated?
→ What does model comparison show?
```

---

## Day 10 — Application & Research Packaging

Prepare:

### CV bullets

Emphasize:

- Black-Scholes / Monte Carlo / PDE;
- Heston stochastic volatility;
- Fourier pricing;
- nonlinear calibration;
- numerical validation;
- residual / identifiability analysis.

### SOP paragraph

Focus on:

- transition from finance to computational quantitative methods;
- interest in numerical pricing and model calibration;
- numerical vs model error;
- motivation for graduate study.

### Technical interview version

Prepare:

- 30-second summary;
- 2-minute explanation;
- detailed technical walkthrough;
- likely interview questions.

### Research summary / paper outline

Possible paper:

> **From Numerical Error to Model Error: A Comparative Study of Option Pricing and Stochastic Volatility Calibration**

Possible structure:

```text
1. Introduction
2. Black-Scholes Benchmark
3. Monte Carlo and PDE Numerical Error
4. Implied-Volatility Diagnostics
5. Heston Stochastic Volatility
6. Semi-Analytical Pricing
7. Calibration and Identifiability
8. Black-Scholes vs Heston Comparison
9. Limitations
10. Conclusion
```

---

# 9. Important Methodological Rules

## Numerical validation before interpretation

Do not interpret model results before validating the pricing implementation.

Examples already used:

```text
Black-Scholes analytical benchmark
MC confidence intervals
PDE grid convergence
phi(0)=1
phi(-i)=S0 exp(rT)
put-call parity
Fourier truncation stability
MC cross-validation
```

## Sampling error is not the same as discretization error

For Heston Monte Carlo:

```latex
\hat C_{\mathrm{MC}}-C_{\mathrm{Heston}}
=
\text{sampling error}
+
\text{discretization error}
```

Do not attribute every MC-vs-Fourier difference to Euler bias.

## Calibration fit is not parameter identification

Do not write:

> “The parameters are correct because RMSE is low.”

Instead distinguish:

```text
price-fit quality
optimization stability
parameter recovery
parameter identifiability
```

## Feller condition

The Feller condition is sufficient, not necessary.

Do not say:

> “If the Feller condition fails, the Heston model is invalid.”

## Heston Fourier implementation

The current implementation is an educational / research baseline.

Potential future numerical improvements include:

- more robust characteristic-function formulations for extreme parameters;
- optimized quadrature;
- cached terms;
- vectorized / fixed-node integration;
- calibration speed improvements.

Do not claim universal numerical robustness.

---

# 10. Recommended New-Conversation Prompt

When starting a new ChatGPT conversation, upload this file and say:

> I am continuing my quantitative finance project from a previous conversation. Please read `PROJECT_HANDOFF.md` first and treat it as the authoritative project state. Do not restart or redesign Days 1–7. Continue from Day 8 and preserve the research logic, implementation style, mathematical notation, validation standards, and workflow described in the handoff. Start by explaining the Day 8 research question and expected outputs, then guide me step by step.

If relevant, also upload:

```text
README.md
notebooks/07_heston_calibration.ipynb
src/calibration/heston_calibration.py
src/pricing/heston_pricing.py
src/volatility/implied_volatility.py
```

The new conversation should request additional files only when needed.

---

# 11. Immediate Next Step

The next substantive task is:

> **Day 8 — Black-Scholes vs Heston Model Comparison**

Before writing new code, first define:

1. what data / surface will be treated as the comparison target;
2. how Black-Scholes will be fitted;
3. how Heston will be calibrated to exactly the same observations;
4. which error metrics will be compared;
5. how residual structure will distinguish model error from numerical error.

Do **not** jump directly into code before defining the research question.

---

# 12. Git / Reproducibility Checklist

Before final project delivery:

```bash
pytest -v
```

Expected baseline:

```text
33 passed
```

Then for every notebook:

```text
Restart Kernel
→ Run All
```

Check:

- no traceback;
- no stale outputs;
- figures reproduce;
- data tables reproduce;
- paths are project-relative rather than machine-fragile where possible.

Finally:

```bash
git status
```

Desired:

```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

### Special current reminder

Verify that the corrected Day 3 notebook has been committed and pushed after removing the premature `mc_rmse` comparison cell.

---

# 13. One-Sentence Project Summary

> This project progresses from analytical and numerical option pricing to stochastic-volatility calibration, using systematic numerical validation to separate computational error from model-specification and calibration error.
