'''Sample fixed and varying numbers of Poisson-rate change points.'''

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import ChangePointModel


MOVE_NAMES = ('height', 'position', 'birth', 'death')


def split_state(state: ArrayLike, position: float, u: float,
                duration: float) -> tuple[NDArray[np.float64], float]:
    '''Split one segment using a uniform auxiliary variable and Green's rate map.

    Args
        state (ArrayLike): Valid vector [s1, ..., sk, h0, ..., hk].
        position (float): New interior change-point time, distinct from existing times.
        u (float): Auxiliary variable strictly between zero and one.
        duration (float): Observation duration in days.

    Returns
        tuple: Proposed state and log absolute Jacobian of the birth transformation.
    '''
    x = np.asarray(state, dtype=float)
    k = (len(x) - 1) // 2
    positions, rates = x[:k], x[k:]
    if not 0 < position < duration or np.any(positions == position) or not 0 < u < 1:
        raise ValueError('The new position and auxiliary variable must be interior.')
    j = int(np.searchsorted(positions, position))
    left = positions[j - 1] if j else 0.0
    right = positions[j] if j < k else duration
    fraction = (position - left) / (right - left)
    log_ratio = np.log1p(-u) - np.log(u)
    log_old = np.log(rates[j])
    log_left = log_old - (1 - fraction) * log_ratio
    log_right = log_old + fraction * log_ratio
    new_rates = np.r_[rates[:j], np.exp(log_left), np.exp(log_right), rates[j + 1:]]
    proposed = np.r_[np.insert(positions, j, position), new_rates]
    log_jacobian = 2 * np.logaddexp(log_left, log_right) - log_old
    return proposed, float(log_jacobian)


def merge_state(state: ArrayLike, index: int,
                duration: float) -> tuple[NDArray[np.float64], float, float]:
    '''Merge adjacent segments using the inverse birth transformation.

    Args
        state (ArrayLike): Valid state with at least one change point.
        index (int): Zero-based index of the change point to remove.
        duration (float): Observation duration in days.

    Returns
        tuple: Proposed state, recovered auxiliary variable and log death Jacobian.
    '''
    x = np.asarray(state, dtype=float)
    k = (len(x) - 1) // 2
    if not isinstance(index, (int, np.integer)) or not 0 <= index < k:
        raise ValueError('The change-point index is out of range.')
    positions, rates = x[:k], x[k:]
    left = positions[index - 1] if index else 0.0
    right = positions[index + 1] if index + 1 < k else duration
    fraction = (positions[index] - left) / (right - left)
    log_left, log_right = np.log(rates[index:index + 2])
    log_merged = fraction * log_left + (1 - fraction) * log_right
    log_sum = np.logaddexp(log_left, log_right)
    new_rates = np.r_[rates[:index], np.exp(log_merged), rates[index + 2:]]
    proposed = np.r_[np.delete(positions, index), new_rates]
    return proposed, float(np.exp(log_left - log_sum)), float(log_merged - 2 * log_sum)


def move_probabilities(model: ChangePointModel) -> NDArray[np.float64]:
    '''Allocate height, position, birth and death move probabilities for each k.

    Args
        model (ChangePointModel): Model with a truncated Poisson prior on k.

    Returns
        ndarray: Array of shape (k_max+1, 4), with rows summing to one.
    '''
    ks = np.arange(model.k_max + 1)
    births = np.minimum(1.0, model.lam / (ks + 1))
    deaths = np.minimum(1.0, ks / model.lam)
    births[-1] = 0.0
    maximum = np.max(births + deaths)
    scale = 0.9 / maximum if maximum > 0 else 0.0
    births *= scale
    deaths *= scale
    within = 1 - births - deaths
    probabilities = np.column_stack([within / 2, within / 2, births, deaths])
    probabilities[0, 0] = within[0]
    probabilities[0, 1] = 0.0
    return probabilities


def _check_iterations(value: int, name: str, *, allow_zero: bool = False) -> None:
    '''Validate a count used to allocate or iterate over a Markov chain.'''
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer.')


def sample_m1(model: ChangePointModel, rng: np.random.Generator, *,
              burn: int = 10000, n_samples: int = 50000,
              sigma_s: float = 500.0) -> tuple[NDArray[np.float64], float]:
    '''Alternate conjugate rate draws and a Gaussian change-point Metropolis step.

    Args
        model (ChangePointModel): Event data and prior parameters, with k_max >= 1.
        rng (numpy.random.Generator): Explicit random number generator.
        burn (int): Number of initial states to discard.
        n_samples (int): Number of retained states.
        sigma_s (float): Positive Gaussian proposal width in days. Proposals outside
            the observation window are rejected.

    Returns
        tuple: Chain of shape (n_samples, 3), with columns [s1, h0, h1], and the
            position acceptance fraction over all attempted updates.
    '''
    _check_iterations(burn, 'burn', allow_zero=True)
    _check_iterations(n_samples, 'n_samples')
    if model.k_max < 1 or not np.isfinite(sigma_s) or sigma_s <= 0:
        raise ValueError('M1 requires k_max >= 1 and a positive finite proposal width.')
    chain = np.empty((n_samples, 3))
    position = model.duration / 2
    accepted = 0
    for iteration in range(burn + n_samples):
        counts = model.segment_counts([position])
        gaps = np.array([position, model.duration - position])
        rates = rng.gamma(model.alpha + counts, 1 / (model.beta + gaps))
        candidate = position + rng.normal(0, sigma_s)
        old_state = np.r_[position, rates]
        new_state = np.r_[candidate, rates]
        log_ratio = model.log_joint(new_state) - model.log_joint(old_state)
        if np.log(rng.random()) < log_ratio:
            position = candidate
            accepted += 1
        if iteration >= burn:
            chain[iteration - burn] = np.r_[position, rates]
    return chain, accepted / (burn + n_samples)


def sample_rjmcmc(model: ChangePointModel, rng: np.random.Generator, *,
                  n_steps: int = 100000,
                  initial_state: ArrayLike | None = None
                  ) -> tuple[list[NDArray[np.float64]], dict[str, dict[str, int | float]]]:
    '''Sample the joint posterior with rate, position, birth and death moves.

    Args
        model (ChangePointModel): Event data and normalized prior parameters.
        rng (numpy.random.Generator): Explicit random number generator.
        n_steps (int): Number of transitions after initialization.
        initial_state (ArrayLike | None): Optional valid starting state. By default,
            start in M0 with a rate drawn from its conjugate posterior.

    Returns
        tuple: A list of n_steps+1 independent state arrays including initialization,
            and per-move dictionaries containing attempted, accepted and acceptance.
    '''
    _check_iterations(n_steps, 'n_steps')
    if initial_state is None:
        shape, rate = model.constant_posterior()
        state = np.array([rng.gamma(shape, 1 / rate)])
    else:
        state = np.asarray(initial_state, dtype=float).copy()
    if not model.valid_state(state):
        raise ValueError('The initial state is outside the model support.')
    probabilities = move_probabilities(model)
    chain = [state.copy()]
    attempts = np.zeros(4, dtype=int)
    accepts = np.zeros(4, dtype=int)
    for _ in range(n_steps):
        k = (len(state) - 1) // 2
        move = int(rng.choice(4, p=probabilities[k]))
        attempts[move] += 1
        candidate = state.copy()
        adjustment = 0.0
        if move == 0:
            j = int(rng.integers(k + 1))
            gaps = np.diff(np.r_[0.0, state[:k], model.duration])
            counts = model.segment_counts(state[:k])
            candidate[k + j] = rng.gamma(model.alpha + counts[j], 1 / (model.beta + gaps[j]))
            accepted = True
        else:
            if move == 1:
                j = int(rng.integers(k))
                left = state[j - 1] if j else 0.0
                right = state[j + 1] if j + 1 < k else model.duration
                candidate[j] = rng.uniform(left, right)
            elif move == 2:
                position, u = rng.uniform(0, model.duration), rng.random()
                if position == 0 or u == 0 or np.any(state[:k] == position):
                    chain.append(state.copy())
                    continue
                candidate, log_jacobian = split_state(state, position, u, model.duration)
                adjustment = (np.log(probabilities[k + 1, 3]) - np.log(k + 1)
                              - np.log(probabilities[k, 2]) + np.log(model.duration)
                              + log_jacobian)
            else:
                j = int(rng.integers(k))
                candidate, _, log_jacobian = merge_state(state, j, model.duration)
                adjustment = (np.log(probabilities[k - 1, 2]) - np.log(model.duration)
                              - np.log(probabilities[k, 3]) + np.log(k)
                              + log_jacobian)
            log_ratio = model.log_joint(candidate) - model.log_joint(state) + adjustment
            accepted = bool(np.log(rng.random()) < log_ratio)
        if accepted:
            state = candidate
            accepts[move] += 1
        chain.append(state.copy())
    summary = {name: {'attempted': int(attempts[i]), 'accepted': int(accepts[i]),
                      'acceptance': float(accepts[i] / attempts[i]) if attempts[i] else 0.0}
               for i, name in enumerate(MOVE_NAMES)}
    return chain, summary
