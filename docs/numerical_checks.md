# Numerical validation

The calculations implement the probability model described in [Statistical model and inference](method.md). Validation combines analytic results, independent numerical integration and distributional tests.

## Event counting

The dataset contains 190 intervals between 191 events. Event times are constructed as `[0, cumsum(intervals)]`, spanning 40,549 days. Tests check that the first event, final event and coincident-event pair are all retained. An event at a change point belongs to the segment on its right.

## Normalized priors

The even-order-statistic position density has normalization

$$
C_k=\frac{(2k+1)!}{L^{2k+1}},
\qquad
\frac{C_{k+1}}{C_k}=\frac{2(k+1)(2k+3)}{L^2}.
$$

Numerical integration checks the position density for one and two change points. Dimension-changing acceptance ratios are checked against direct evaluation of the complete normalized prior, including the truncated Poisson model probability and Gamma rate densities.

## Model evidence

The constant-rate evidence is analytic. For one change point, integrating out both Gamma-distributed rates leaves a one-dimensional integral over position. Quadrature is performed separately between event times, where the segment counts are constant.

For the bundled dataset, with $\alpha=1$ and $\beta=200$ days:

| Quantity | Reference value |
| --- | ---: |
| $\log Z_0$ | -1217.087769400 |
| $\log Z_1$ | -1187.343876579 |
| $\log B_{10}$ | 29.743892821 |

Tests compare these calculations with independent integrals and compare the M1 sampler's empirical position distribution with the integrated reference CDF. The evidences are conditional on $M_0$ and $M_1$; model-index prior odds are applied separately.

## Reversible-jump sampling

The tests check:

- Split/merge inversion and the transformation Jacobian using finite differences.
- Reciprocal birth/death acceptance ratios evaluated from normalized densities.
- Positive rates, ordered interior change points and bounded model dimension.
- Recovery of the model prior when the likelihood is nearly uninformative.
- Reproducibility with explicit random generators and isolation from global random state.
- Consistent burn-in handling, variable-length state storage and posterior summaries.

## Diagnostics and uncertainty

Classical split R-hat and autocorrelation-based effective sample sizes are reported for the number of change points and integrated intensity. Constant traces produce undefined diagnostics, represented as JSON null values.

Rate-function intervals are pointwise posterior intervals. Their calculation is checked against direct evaluation on small chains. Numerical correctness of transitions and satisfactory scalar diagnostics do not establish convergence of every change-point configuration; inspect the traces and compare independent runs.
