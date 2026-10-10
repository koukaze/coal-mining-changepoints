'''Summarise scalar MCMC traces without asserting convergence.'''

import numpy as np


def effective_sample_size(trace: np.ndarray) -> float:
    '''Estimate scalar effective sample size using an initial positive pair sum.
    Args
        trace (ndarray): Finite one-dimensional post-burn-in samples.
    Returns
        float: Autocorrelation-based effective sample size, or NaN for a constant trace.
    '''
    values = np.asarray(trace, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError('A trace must contain at least four finite values.')
    centred = values - values.mean()
    size = 1 << (2 * len(values) - 1).bit_length()
    transformed = np.fft.rfft(centred, n=size)
    autocov = np.fft.irfft(transformed * transformed.conjugate(), n=size)[:len(values)]
    if autocov[0] <= 0:
        return float('nan')
    acf = autocov / autocov[0]
    n_pairs = len(acf) // 2
    pairs = acf[:2 * n_pairs].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    positive = pairs[:stop[0]] if len(stop) else pairs
    tau = max(1.0, -1.0 + 2.0 * positive.sum())
    return float(len(values) / tau)


def split_rhat(chains: np.ndarray) -> float:
    '''Compute the classical split R-hat for a scalar across independent chains.
    Args
        chains (ndarray): Finite post-burn-in samples, shape (n_chains, n_draws).
    Returns
        float: Split variance-ratio statistic, or NaN if within-chain variance vanishes.
    '''
    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4 or not np.isfinite(values).all():
        raise ValueError('Provide at least two finite chains with at least four draws each.')
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = split.var(axis=1, ddof=1).mean()
    if within <= 0:
        return float('nan')
    between = half * split.mean(axis=1).var(ddof=1)
    variance = (half - 1) / half * within + between / half
    return float(np.sqrt(variance / within))


def k_trace(chain: list[np.ndarray]) -> np.ndarray:
    '''Extract the number of change points from each variable-length state.
    Args
        chain (list): Arrays with k positions followed by k+1 rates.
    Returns
        ndarray: Integer model indices in chain order.
    '''
    return np.array([(len(state) - 1) // 2 for state in chain], dtype=int)


def integrated_rate(chain: list[np.ndarray], duration: float) -> np.ndarray:
    '''Calculate the expected event count implied by each sampled rate function.
    Args
        chain (list): Variable-length arrays containing positions followed by rates.
        duration (float): Observation period in days.
    Returns
        ndarray: Integrated intensities, in expected events, for the sampled states.
    '''
    result = np.empty(len(chain))
    for i, state in enumerate(chain):
        k = (len(state) - 1) // 2
        result[i] = np.dot(np.diff(np.r_[0.0, state[:k], duration]), state[k:])
    return result


def rate_summary(chain: list[np.ndarray], duration: float, n_grid: int = 400) -> dict[str, np.ndarray]:
    '''Evaluate pointwise posterior rate means and central credible intervals.
    Args
        chain (list): Selected post-burn-in states from one or more chains.
        duration (float): Observation period in days.
        n_grid (int): Number of equally spaced evaluation times, including both endpoints.
    Returns
        dict: Time grid in days and mean, 5th, 25th, 75th and 95th rate percentiles per day.
    '''
    if not chain or n_grid < 2 or duration <= 0:
        raise ValueError('Provide posterior states, positive duration and at least two grid points.')
    time = np.linspace(0, duration, n_grid)
    rates = np.empty((len(chain), n_grid))
    for i, state in enumerate(chain):
        k = (len(state) - 1) // 2
        rates[i] = state[k:][np.searchsorted(state[:k], time, side='right')]
    quantiles = np.percentile(rates, [5, 25, 75, 95], axis=0)
    return dict(time_days=time, mean=rates.mean(axis=0), p5=quantiles[0], p25=quantiles[1],
                p75=quantiles[2], p95=quantiles[3])
