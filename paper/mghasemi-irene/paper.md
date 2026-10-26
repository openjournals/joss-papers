---
title: 'Irene: A Python Toolkit Unifying SOS, SONC, and Geometric Programming Relaxations for Polynomial Optimization'
tags:
  - Python
  - polynomial optimization
  - sum-of-squares
  - SONC
  - geometric programming
  - semidefinite programming
authors:
  - name: "Mehdi Ghasemi"
    orcid: 0000-0002-9551-1780
    affiliation: "Edmonton Police Service"
date: 25 October 2026
bibliography: paper.bib
---

# Summary

Computing the global minimum of a polynomial on a semialgebraic set is a nonconvex problem for which the most widely used certificates come from convex relaxations. We present *Irene*, an open-source Python package that unifies three such relaxation families behind one API: moment/sum-of-squares (SOS) semidefinite hierarchies, geometric-programming (GP) lower bounds, and sum-of-nonnegative-circuit (SONC) certificates — together with the algebraic structure that makes sparse reductions possible, namely commutative semigroups, their semigroup algebras, Newton polytopes, and border bases. All relaxations return certified lower bounds on the global minimum; optional structural reductions compose multiplicatively to shrink moment-matrix dimension. The package ships a benchmark problem gallery with structured output for regression tracking, runs under continuous integration across two Python versions and two solver backends, and is installable from PyPI. This paper documents the implemented methods, the API design, and the reproducibility infrastructure; it reports no new mathematical results.

# Statement of need

Practitioners who certify global optima — in control verification, robust optimization, and applied algebraic geometry — currently face a fragmented landscape. The mature implementations of moment/SOS hierarchies live in MATLAB (SOSTOOLS, YALMIP's SOS module, GloptiPoly) or the Julia stack (SumOfSquares.jl with PolyJuMP and MultivariateMoments); Python users who need these certificates either leave their ecosystem or assemble semidefinite programs by hand. A further gap is methodological: SOS/moment bounds, GP bounds built on circuit structure, and SONC certificates are each strongest on different instance classes (dense versus sparse objectives, constrained versus separating problems), yet no existing toolkit lets a user compute all three from one problem model and compare them side by side. Irene addresses both needs with a single `pip`-installable package in which the three relaxation families share one representation and one reduction pipeline.

# State of the field

Sum-of-squares and moment hierarchies are mature in MATLAB: SOSTOOLS [@parrilo2000thesis], YALMIP's SOS module, and GloptiPoly [@henrion2003globtopoly] — all ecosystem-locked. The Julia stack is the most developed non-MATLAB alternative and covers signomial relaxations via relative entropy [@murray2021signomial]. Python's offerings to date: CVXPY provides disciplined geometric programming as a modeling language but no first-class moment/SOS hierarchy [@diamond2016cvxpy]; SumOfSquares.py offers SOS certificates and pseudoexpectation duality without moment relaxations; ncpol3sdpa (2025) ports Lasserre's SDP relaxations from R. To our knowledge, Irene is the only published Python package that unifies SOS/moment-SDP, SONC/SOSONC, and GP lower-bound hierarchies in a single toolkit over a common semigroup-algebraic backend; its development began in late 2016, before any other open-source Python effort in this space (verified against PyPI/GitHub coverage of installable packages).

Irene deliberately builds on existing ecosystems rather than reinventing them: conic formulations are assembled through CVXPY's DCP layer [@diamond2016cvxpy] with Clarabel, SCS, or CVXOPT as backends. What it contributes is the semigroup-algebraic backbone — a representation for which Newton-polytope pruning and border-basis quotienting are first-class, composable operations — plus the unification of three certificate families that no single existing package spans.

# Software design

**Semigroup-algebraic representation.** Irene does not model problems as plain polynomials $f(x_1,\dots,x_n)$ directly: the user declares a commutative semigroup (monomial exponents) and works inside its semigroup algebra, so expressions carry explicit support/exponent data that downstream routines consume as geometry rather than re-parsing. The algebra layer is built on a swappable symbolic engine — SymEngine primary, SymPy fallback, selected via `IRENE_SYMBOLIC_BACKEND`. A hot-path micro-benchmark shows the abstraction's per-call tax is small at shipped problem sizes (≈+6 µs per polynomial construction, ≈+27 µs per Gröbner basis against solve times of $\ge 0.1$ s), so backend choice is a second-order factor in end-to-end runtimes.

**Unified engine.** One entry point covers all relaxation families:

```python
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig

config = RelaxationConfig(reduction_method="newton_polytope",
                          monomial_pruning=True, sparsity_detection=True)
engine = RelaxationEngine(prog, order=2, solver="clarabel", config=config)
result = engine.solve("sos")   # or "sonc", "sosonc_sos_first", "sosonc_sonc_first"
```

The documented return contract is a `RelaxResult` carrying the certified lower bound (`value`, −inf when infeasible), solver status, method used, runtimes, and — on the SDP path — the certificate plus metadata including the relaxation order actually used. A convenience function `compare_all()` runs all four methods side by side for the same problem and order: the package's built-in answer to "which relaxation wins here?"

**Sparse reductions.** Three structural reductions shrink the moment-matrix basis $B$ (all opt-in via `RelaxationConfig`, disabled by default so legacy call sites produce unchanged relaxations): border-basis quotienting, which replaces symbolic Gröbner reduction with numerically computed multiplication tables of the quotient algebra [@moller1987borderbasis]; Newton-polytope pruning, restricting $B_{2t}$ to monomials inside the dilated Newton polytope of the data [@lasserre2003sparsity; @iliman2016simplex] with a certified-safe retention set (a 2026 audit found and fixed an over-pruning defect that had silently discarded positive-semidefinite coupling entries); and correlative-sparsity detection, replacing one dense PSD block by coupled clique-sized blocks via chordal completion of the variable-interaction graph [@lasserre2003sparsity]. The documented two-stage composition (inner algebraic reduction + outer chordal decomposition) is designed so dimension factors compose multiplicatively.

**Solver routing.** SDP paths run through CVXPY's DCP layer with Clarabel or CVXOPT as primary backends (SCS also supported), a native non-DCP CVXOPT path, and external solvers (DSDP, SDPA, CSDP) via CLI for large instances and the mean-polynomial path; GP/SONC relaxations use a gpkit-based backend. The documentation is explicit about known behavioral differences between backends — for example, infeasibility detection can differ on structurally feasible instances — which we treat as a feature of honest API design rather than something to hide.

**Benchmarking and reproducibility.** `benchmarks/gallery.yaml` catalogs problems (id, variables, degree, category, objective, constraints, known optimum); `run_gallery.py` executes any subset with structured JSON output for regression tracking, building every problem through the same public semigroup-algebra API a user would write. A minimal gallery smoke subset of the documented examples runs in CI on every push; the ill-conditioned Motzkin cell is checked against a documented early-stop band rather than a point value because its order-3 SDP has moment-matrix condition number $\approx 4.9\times10^{13}$, so solver outputs are early stops whose spread (tens of percent) exceeds any double-precision tolerance budget — exactly the instance class Irene's stability warning exists to flag. Measured speedups from the shipped reduction pipeline on dense bivariate instances are $0.69$–$1.32\times$: correctness is verified, and dimension reductions take effect once certified sparsity sets exist for sparse instances.

# Limitations

Structural reductions are opt-in; users must configure `RelaxationConfig` to benefit from them. The package's own README states a known gap: the Phase-3 reduction modules are implemented and tested, but full end-to-end integration through all legacy code paths is still in progress. Border-basis reduction is numerical (floating-point multiplication tables); precision caveats are documented. On structurally feasible instances of relaxation order $\ge 3$ some backends report infeasibility where Clarabel returns an optimal value; we recommend Clarabel for those cases and advise cross-checking status strings when switching solvers. The GP/SONC path requires a working gpkit backend and degrades noisily on ill-conditioned posynomials (documented). Finally, the legacy pre-2026 API remains importable and tested with a documented migration mapping, but new work should use the unified engine.

# Research impact statement

Irene is in active research use. The co-authored paper *The Cone Generated by Positive Semidefinite Mean Polynomials* [@ghasemi2026meanpolynomialcone] (arXiv:2608.22739) cites Irene four times in its body for the computations supporting it — adoption by a collaborating research group at another institution, and an independent installation of the public package. Earlier versions of the codebase (developed under the same name since late 2016) were used to produce the computational results in the author's published geometric-programming lower-bound line: *Lower bounds for polynomials using geometric programming* [@ghasemi2012lowerbounds], *Lower bounds on the global minimum of a polynomial* [@ghasemi2013globalmin], *Lower bounds for a polynomial on a basic closed semialgebraic set using geometric programming* [@ghasemi2013semialgebraic], and *Lower bound for a polynomial on a product of hyperellipsoids using geometric programming* [@ghasemi2025hyperellipsoids]. These methods were extended to constrained problems by the SONC Positivstellensatz chain [@dressler2017positivstellensatz; @dressler2019constrained]. The package is designed for maintainable extension: a public issue tracker documents bug reports and feature requests, and the legacy-compatibility policy exists precisely so external users can migrate deliberately rather than all at once.

# AI usage disclosure

In preparing this manuscript and developing parts of the software, the author used Hermes Agent assistance (an agent framework running local LLM models — Qwen-3 family 27B-parameter models served locally via LM Studio/Ollama) for drafting prose, structuring sections, refactoring code, test scaffolding, generating documentation examples, and verifying bibliographic metadata against Crossref/OpenAlex/arXiv records. All mathematical statements were checked by the author against the primary literature; all software claims were verified by direct inspection of the source code and documentation at the cited version; all benchmark numbers are produced by running the shipped scripts listed in this paper. The author made the core design decisions, reviewed, edited, and validated all AI-assisted outputs, and takes full responsibility for the content.

# Acknowledgements

This work was partially supported by Defence Research and Development Canada (DRDC) under grant CMN1-008. The funding agency had no role in the design or implementation of the software or in the preparation of this manuscript.
