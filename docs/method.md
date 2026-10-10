# Statistical model and inference

The analysis models accident times as an inhomogeneous Poisson process with a piecewise-constant rate. It compares a constant-rate model, a model with one change point, and a family with an unknown number of change points. The transdimensional construction follows [Green (1995)](references.md).

## Events and observation interval

The historical data contain 190 inter-accident intervals. Their cumulative sum, together with an event at time zero, gives **191 events over 40,549 days**. A zero interval represents two distinct accidents on the same day; both events contribute to the likelihood. All calculations use this same event list and duration. See [data provenance](data.md) for the source and ordering of the intervals.

Time is measured in elapsed days and rates in accidents per day. For a change point at time $s$, events with $t<s$ belong to the left segment and events with $t\geq s$ belong to the right segment. The final observed event at $L$ is assigned to the last segment. The historical workflow takes $L$ to be the span between the first and last recorded events; it does not add an unobserved interval after the final event.

## Likelihood and normalized priors

For $k$ ordered change points, write $0=s_0<s_1<\cdots<s_k<s_{k+1}=L$, segment lengths $d_j=s_{j+1}-s_j$, event counts $n_j$, and rates $h_j>0$. The log likelihood is

$$
\log\mathcal L(\mathbf s,\mathbf h)
=\sum_{j=0}^{k}\left[n_j\log h_j-h_jd_j\right].
$$

This is the event-time Poisson-process likelihood. No segment-specific factorial is included: the observations are event times, rather than only binned counts. Parameter-independent constants cancel in the model comparisons below.

The number of change points has a truncated Poisson prior,

$$
\pi(k)=\frac{\lambda^k/k!}{\sum_{r=0}^{k_{\max}}\lambda^r/r!},
\qquad k=0,\ldots,k_{\max},
$$

with defaults $\lambda=3$ and $k_{\max}=30$. Rates independently follow a Gamma distribution in the **shape-rate** convention,

$$
\pi(h_j)=\frac{\beta^\alpha}{\Gamma(\alpha)}
h_j^{\alpha-1}e^{-\beta h_j},
\qquad \alpha=1,\quad\beta=200\ \mathrm{days}.
$$

Conditional on $k$, change points are the even-numbered order statistics of $2k+1$ independent uniform draws on $[0,L]$. Their normalized density on the ordered region is

$$
\pi(\mathbf s\mid k)
=\frac{(2k+1)!}{L^{2k+1}}\prod_{j=0}^{k}d_j.
$$

Equivalently, the normalized segment lengths follow $\mathrm{Dirichlet}(2,\ldots,2)$. For $k=0$ this density is one; for $k=1$ it reduces to $6s(L-s)/L^3$. The prior suppresses nearly coincident change points while allowing uncertainty in their positions.

## Fixed-dimension inference and evidence

Gamma-Poisson conjugacy gives the conditional rate posterior

$$
h_j\mid\mathbf s,\mathbf t
\sim\mathrm{Gamma}(\alpha+n_j,\beta+d_j).
$$

The one-change-point sampler alternates these Gibbs updates with a Gaussian random-walk Metropolis update for the position. Proposals outside $(0,L)$ are rejected. The Gaussian proposal is symmetric; it is not resampled until it falls inside the interval.

For model comparison, integrate each rate analytically. Define the segment contribution

$$
A(n,d)=\frac{\beta^\alpha\Gamma(n+\alpha)}
{\Gamma(\alpha)(\beta+d)^{n+\alpha}}.
$$

Then the constant-rate evidence is $Z_0=A(N,L)$, and the one-change-point evidence is

$$
Z_1=\int_0^L\frac{6s(L-s)}{L^3}
A(n_0(s),s)A(N-n_0(s),L-s)\,\mathrm ds.
$$

Numerical quadrature is split at the unique event times, where the segment counts change. The computation uses logarithms and scaling to avoid underflow. The same integrand, normalized by $Z_1$, gives an independent reference for the marginal posterior of the change-point position.

The Bayes factor is $B_{10}=Z_1/Z_0$. Posterior odds additionally include the model-prior ratio: $O_{10}=\lambda B_{10}$. Log evidences, log Bayes factors and ordinary ratios are reported as distinct quantities. Nested sampling and a Savage-Dickey density estimate are not used to calculate the reported evidence; see [numerical checks](numerical_checks.md).

## Reversible-jump MCMC

The sampler explores the posterior over $k$, positions and rates through four move types:

1. A Gibbs update for one segment rate.
2. A position move between the neighbouring change points, with a Metropolis-Hastings correction.
3. A birth move that inserts a uniformly proposed change point and splits a segment rate.
4. A death move that removes a selected change point and merges its neighbouring rates.

Birth and death acceptance probabilities include the likelihood ratio, the complete normalized prior ratio, the forward/reverse proposal probabilities and the Jacobian. For a split into lengths $d_\ell,d_r$, let $w=d_\ell/(d_\ell+d_r)$ and draw $u\in(0,1)$. The rate transformation is

$$
h_\ell=h\left(\frac{u}{1-u}\right)^{1-w},
\qquad
h_r=h\left(\frac{1-u}{u}\right)^w.
$$

The inverse merge is $h=h_\ell^w h_r^{1-w}$, and the absolute split Jacobian is $(h_\ell+h_r)^2/h$. The implementation evaluates the full prior density when forming acceptance ratios, retaining the normalization that depends on $k$.

After discarding burn-in, model probabilities are estimated from the frequencies of sampled values of $k$. Each retained draw also defines a rate function on a common time grid. Pointwise means and percentiles across these draws marginalize over the number of change points. The 25th–75th and 5th–95th percentiles give **pointwise** 50% and 90% credible intervals, not simultaneous bands for the entire function.

Trace plots, move acceptance rates and autocorrelation diagnostics assess sampling behaviour. Effective sample size is a Monte Carlo diagnostic; neither a large value nor a stationary-looking single trace establishes convergence. Grid-based rate summaries likewise do not increase the number of independent posterior draws.
