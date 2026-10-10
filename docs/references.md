# References

- Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation
  and Bayesian model determination. *Biometrika*, **82**(4), 711–732.
  [doi:10.1093/biomet/82.4.711](https://doi.org/10.1093/biomet/82.4.711).
  This is the methodological reference for sampling across models with
  different numbers of change points.
- Jarrett, R. G. (1979). A note on the intervals between coal-mining disasters.
  *Biometrika*, **66**(1), 191–193.
  [doi:10.1093/biomet/66.1.191](https://doi.org/10.1093/biomet/66.1.191).
  This paper corrects and extends the historical disaster series.
- Hand, D. J., Daly, F., Lunn, A. D., McConway, K. J., and Ostrowski, E.
  (1994). *A Handbook of Small Data Sets*. Chapman & Hall.
  This is the source credited by the
  [`boot::coal` documentation](https://stat.ethz.ch/R-manual/R-devel/library/boot/html/coal.html).
- Canty, A., Ripley, B., and Brazzale, A. R. `boot`: Bootstrap Functions,
  version 1.3-32 (2025).
  [CRAN package record](https://CRAN.R-project.org/package=boot).
  The bundled interval data were exported from this package under its stated
  `Unlimited` terms; see [data provenance](data.md).

## Coursework framework

Christopher J. Moore, *S2 Statistical Methods coursework*, Cambridge MPhil in Data Intensive Science, Lent Term 2026. The coursework supplies the model specification, priors and a partial RJMCMC framework following Green (1995). The Gibbs rate update and birth/death log-prior ratios are the exercise components implemented here.
