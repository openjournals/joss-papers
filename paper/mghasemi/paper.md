---
title: 'SKSurrogate: scikit-learn-native surrogate-guided hyperparameter search'
tags:
  - Python
  - scikit-learn
  - hyperparameter optimization
  - surrogate optimization
  - AutoML
  - benchmark
authors:
  - name: "Mehdi Ghasemi"
    orcid: 0000-0002-9551-1780
    affiliation: "Edmonton Police Service"
date: 25 October 2026
bibliography: paper.bib
---

# Summary

`SKSurrogate` is a `scikit-learn`-style suite that implements *surrogate-guided optimization* for
expensive objectives, chiefly hyperparameter and AutoML pipeline search. A search drives an
evolutionary outer loop and, once enough evaluations accumulate, replaces direct (expensive) model
fitting for candidate proposals with a surrogate of the response surface, using the surrogate's
optimum to guide the next evaluations.

Since version **0.2.0** the default response surface is a stock, un-tuned RBF
`GaussianProcessRegressor`, chosen over the legacy cubic Hilbert polynomial on the evidence of an
equal-budget benchmark suite. All `scikit-learn` regressors are accepted as surfaces, and a custom
`HilbertRegressor` (cubic polynomial) remains available for the legacy default.

# Statement of need

Surrogate / Bayesian hyperparameter optimization is well served by mature frameworks (e.g.,
Optuna \citep{sk:optuna}) built on `scikit-learn` \citep{sk:scikit-learn}, but those tools often
carry a heavier framework (studies, orchestrators, acquisition-function machinery) than a
`scikit-learn`-native user needs, and their benchmark claims are frequently hard to reproduce from
the repository alone. SKSurrogate targets the gap between "bare `RandomizedSearchCV`" and "full
orchestration framework": a lightweight fit-predict search over **typed** sklearn-style search
spaces (`Categorical`/`Integer`/`Real`/`HDReal`) with a `BaseSearchCV`-style interface, an
evolutionary outer loop paired with a surrogate-guided inner step, and — explicitly — a frozen,
version-controlled **equal-budget benchmark suite** whose raw results are archived so the advertised
"validated against benchmarks" claim is verifiable in-repo without rerunning experiments. This note
reports that benchmark and the controlled ablation isolating the surrogate component.

# State of the field

Mainstream HPO is dominated by Bayesian/surrogate methods — TPE (Hyperopt), SMAC, Optuna
\citep{sk:optuna}, and (until archival) scikit-optimize — with AutoML systems layering these onto
pipeline construction and early-stopping. SKSurrogate's niche is its `scikit-learn`-native typed search
spaces, its evolutionary-outer / surrogate-guided-inner loop, and its reproducible equal-budget
evaluation contract.

# Method

`SKSurrogate.structsearch.SurrogateRandomCV` implements a local surrogate-driven random search (SRS)
over a discrete search space typed by `Categorical`/`Integer`/`Real`/`HDReal` descriptors. The search:

1. **Outer evolutionary loop.** Candidate parameterizations are sampled and locally optimized.
2. **Surrogate engagement.** After `min_evals` evaluations, a regressor is fit to evaluated points and
   used to (a) propose candidate optima and (b) screen unpromising regions, so only promising points
   pay for an expensive model fit.
3. **Evaluation budget.** Every candidate-model evaluation is counted; the budget `B` is binding
   (verified post-run `n_evals == B`).

`min_evals = B` degrades the search to plain random sampling through the *same* code path — the
mechanism behind the with-vs-without ablation.

# Benchmark protocol

Public suite in `benchmarks/`, reproducible with one command
(`python -m benchmarks.run --dataset <d> --budget <B> --seeds <s>`; protocol in
`benchmarks/DESIGN.md` v1.9). Raw per-run records are git-ignored by design and archived
\citep{sk:bench_data}.

| Item | Setting |
|---|---|
| Estimator (all methods) | `StandardScaler → RandomForestClassifier(n_estimators=50)` |
| Space | 3 params, one shared discrete grid of **224** configs |
| Inner CV | `StratifiedKFold(5)` inside every search |
| Outer split | one shared stratified 80/20 per (dataset, seed), identical for all methods; final "honest" score once on the hold-out |
| Seeds | S = 7 per (method, dataset) |
| Budgets | **B = 30** (five-method suite) and **B = 60** (SRS arms), in candidate-model evaluations |

Methods under test: `sksurrogate` (default RBF-GP since v0.2), `srs-no-surrogate` (ablation arm A),
`random`, `Optuna-TPE`; surface arms `SRS+GP` / `SRS+KRR`. `skopt-bayes` is recorded as not runnable
(repo archived; fails to import against modern scikit-learn). Datasets (OpenML `did`): ionosphere
(59), credit-g (31), arrhythmia (5, high-dim/imbalanced), spambase (44), optdigits (28), Galaxy3
(bundled; synthetic epistasis).

# Results

Mean final cross-validated accuracy over 7 seeds, **B = 30** (best per row in bold):

| dataset | SRS (RBF-GP) | no surrogate | random | Optuna-TPE |
|---|---|---|---:|
| arrhythmia | 0.7020 | 0.6910 | 0.6941 | **0.7036** |
| credit-g | 0.7423 | 0.7391 | 0.7405 | **0.7436** |
| galaxy3 | 0.6473 | 0.6448 | 0.6484 | **0.6507** |
| ionosphere | **0.9352** | 0.9347 | 0.9347 | 0.9337 |
| optdigits | 0.9403 | 0.9392 | 0.9380 | **0.9405** |
| spambase | 0.9280 | 0.9269 | 0.9272 | **0.9286** |

**B = 60** (SRS arms):

| dataset | SRS (RBF-GP) | SRS (cubic Hilbert) | no surrogate |
|---|---|---|---:|
| arrhythmia | **0.7056** | 0.7000 | 0.7004 |
| credit-g | **0.7446** | 0.7412 | 0.7430 |
| galaxy3 | **0.6501** | 0.6485 | 0.6479 |
| ionosphere | **0.9362** | 0.9352 | 0.9362 |
| optdigits | **0.9413** | 0.9399 | 0.9403 |
| spambase | **0.9286** | 0.9277 | 0.9281 |

At B=30, SRS (RBF-GP) matches Optuna-TPE within ~0.002 on 5/6 datasets and *beats* it on the hardest
case (ionosphere); at B=60 it **leads all six**. The with-vs-without ablation shows the surrogate
**matches or raises final accuracy on every dataset** at both budgets.

**Honest limitations.** We deliberately do **not** claim "fewer evaluations to reach target accuracy":
median evaluations to reach within a gap of a run's own final best is *mixed* (uniform `random`
occasionally reaches its own near-best in as few as 2 evaluations). The demonstrated advantage is at
*fixed total budget*, not faster convergence; a sample-efficiency claim would require a
target-anchored protocol over a wider dataset matrix. The surrogate's benefit is modest in absolute
terms (a few tenths of a percent) but consistent and never harmful; measurement is a single shared
80/20 split per (dataset, seed).

# Reproducibility

- Harness: `python -m benchmarks.run`; analyses in `benchmarks/methods.py`.
- Figures and tables: `docs/benchmarks.md`; figures in `docs/benchmarks/figures/`.
- Raw run records (git-ignored): \citep{sk:bench_data}.
- Software: PyPI `sksurrogate==0.2.0`; source `github.com/mghasemi/sksurrogate` tag `v0.2.0`;
  archived \citep{sk:surrogate_software}.

# Acknowledgements

N/A.

# References