'''Define piecewise Poisson models with Gamma rates and ordered change points.'''

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp


@dataclass
class ChangePointModel:
    '''Specify event times, an observation window and normalized model priors.

    Args
        events (ArrayLike): Nondecreasing event times in days, including observed
            endpoint events. Repeated times are allowed.
        duration (float): Positive observation duration in days.
        alpha (float): Positive Gamma shape parameter for segment rates.
        beta (float): Positive Gamma rate parameter in days.
        lam (float): Positive Poisson mean for the number of change points.
        k_max (int): Maximum allowed number of change points.
    '''

    events: ArrayLike
    duration: float
    alpha: float = 1.0
    beta: float = 200.0
    lam: float = 3.0
    k_max: int = 30

    def __post_init__(self) -> None:
        self.events = np.asarray(self.events, dtype=float).copy()
        if self.events.ndim != 1 or not np.all(np.isfinite(self.events)):
            raise ValueError('Events must be a finite one-dimensional array.')
        if any(not np.isfinite(x) or x <= 0 for x in
               (self.duration, self.alpha, self.beta, self.lam)):
            raise ValueError('Duration and prior parameters must be positive and finite.')
        if isinstance(self.k_max, bool) or not isinstance(self.k_max, (int, np.integer)) or self.k_max < 0:
            raise ValueError('k_max must be a nonnegative integer.')
        if np.any(np.diff(self.events) < 0) or np.any(self.events < 0) or np.any(self.events > self.duration):
            raise ValueError('Events must be ordered within [0, duration].')
        self.events.setflags(write=False)

    def valid_state(self, state: ArrayLike) -> bool:
        '''Check the dimensions, ordered interior positions and positive rates.

        Args
            state (ArrayLike): Vector [s1, ..., sk, h0, ..., hk].

        Returns
            bool: Whether the state belongs to the model support.
        '''
        x = np.asarray(state)
        if x.ndim != 1 or len(x) == 0 or len(x) % 2 != 1 or not np.all(np.isfinite(x)):
            return False
        k = (len(x) - 1) // 2
        return bool(k <= self.k_max and np.all(x[k:] > 0)
                    and np.all(x[:k] > 0) and np.all(x[:k] < self.duration)
                    and np.all(np.diff(x[:k]) > 0))

    def segment_counts(self, positions: ArrayLike) -> NDArray[np.int64]:
        '''Count events in right-continuous segments, including the final endpoint.

        Args
            positions (ArrayLike): Ordered interior change-point times in days.

        Returns
            ndarray: Event counts of shape (k+1,).
        '''
        positions = np.asarray(positions, dtype=float)
        indices = np.searchsorted(positions, self.events, side='right')
        return np.bincount(indices, minlength=len(positions) + 1)

    def log_likelihood(self, state: ArrayLike) -> float:
        '''Evaluate the Poisson-process log likelihood up to a common data constant.

        Args
            state (ArrayLike): Vector [s1, ..., sk, h0, ..., hk], in days and day^-1.

        Returns
            float: Log likelihood, or negative infinity outside the support.
        '''
        x = np.asarray(state, dtype=float)
        if not self.valid_state(x):
            return -np.inf
        k = (len(x) - 1) // 2
        gaps = np.diff(np.r_[0.0, x[:k], self.duration])
        counts = self.segment_counts(x[:k])
        return float(counts @ np.log(x[k:]) - gaps @ x[k:])

    def log_position_prior(self, positions: ArrayLike) -> float:
        '''Evaluate the normalized even-numbered order-statistic position density.

        Args
            positions (ArrayLike): Ordered interior change-point times in days.

        Returns
            float: Log density conditional on k, or negative infinity off support.
        '''
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 1 or not np.all(np.isfinite(positions)):
            return -np.inf
        k = len(positions)
        gaps = np.diff(np.r_[0.0, positions, self.duration])
        if k > self.k_max or np.any(gaps <= 0):
            return -np.inf
        return float(gammaln(2 * k + 2) - (2 * k + 1) * np.log(self.duration)
                     + np.log(gaps).sum())

    def log_prior(self, state: ArrayLike) -> float:
        '''Evaluate normalized model, position and independent Gamma rate priors.

        Args
            state (ArrayLike): Vector [s1, ..., sk, h0, ..., hk].

        Returns
            float: Joint log prior density, or negative infinity off support.
        '''
        x = np.asarray(state, dtype=float)
        if not self.valid_state(x):
            return -np.inf
        k = (len(x) - 1) // 2
        ks = np.arange(self.k_max + 1)
        log_weights = ks * np.log(self.lam) - gammaln(ks + 1)
        log_model = log_weights[k] - logsumexp(log_weights)
        rates = x[k:]
        log_rates = ((k + 1) * (self.alpha * np.log(self.beta) - gammaln(self.alpha))
                     + (self.alpha - 1) * np.log(rates).sum() - self.beta * rates.sum())
        return float(log_model + self.log_position_prior(x[:k]) + log_rates)

    def log_joint(self, state: ArrayLike) -> float:
        '''Evaluate the joint log density of events and model parameters.

        Args
            state (ArrayLike): Vector [s1, ..., sk, h0, ..., hk].

        Returns
            float: Log likelihood plus the normalized joint log prior.
        '''
        return self.log_likelihood(state) + self.log_prior(state)

    def constant_posterior(self) -> tuple[float, float]:
        '''Return the Gamma shape and rate for the constant-rate posterior.

        Returns
            tuple: Posterior shape and rate in days for model M0.
        '''
        return self.alpha + len(self.events), self.beta + self.duration

    def log_evidence_m0(self) -> float:
        '''Integrate the constant rate analytically, conditional on model M0.

        Returns
            float: Log evidence without the model-index prior probability.
        '''
        shape, rate = self.constant_posterior()
        return float(self.alpha * np.log(self.beta) - gammaln(self.alpha)
                     + gammaln(shape) - shape * np.log(rate))

    def log_marginal_m1(self, position: float) -> float:
        '''Integrate both rates at a fixed change point and include its prior.

        Args
            position (float): Interior change-point time in days.

        Returns
            float: Log of the evidence integrand with respect to change-point time.
        '''
        if not np.isfinite(position) or not 0 < position < self.duration:
            return -np.inf
        n0 = np.searchsorted(self.events, position, side='left')
        counts = np.array([n0, len(self.events) - n0])
        shapes = counts + self.alpha
        rates = np.array([position, self.duration - position]) + self.beta
        return float(2 * (self.alpha * np.log(self.beta) - gammaln(self.alpha))
                     + np.sum(gammaln(shapes) - shapes * np.log(rates))
                     + np.log(6 * position * (self.duration - position) / self.duration**3))

    def log_evidence_m1(self, *, upper: float | None = None) -> float:
        '''Integrate the collapsed one-change-point model between event boundaries.

        Args
            upper (float | None): Upper integration limit in days. Defaults to the
                observation duration; partial integrals give the position CDF.

        Returns
            float: Log evidence, or log partial evidence when upper is specified.
        '''
        upper = self.duration if upper is None else float(upper)
        if not np.isfinite(upper) or not 0 <= upper <= self.duration:
            raise ValueError('The integration limit must lie in [0, duration].')
        if upper == 0:
            return -np.inf
        edges = np.unique(np.r_[0.0, self.events[(self.events > 0) & (self.events < upper)], upper])
        terms = []
        for left, right in zip(edges[:-1], edges[1:]):
            reference = self.log_marginal_m1((left + right) / 2)
            integral, _ = quad(lambda s: np.exp(self.log_marginal_m1(s) - reference),
                               left, right, epsabs=1e-10, epsrel=1e-10)
            terms.append(np.log(integral) + reference)
        return float(logsumexp(terms))
