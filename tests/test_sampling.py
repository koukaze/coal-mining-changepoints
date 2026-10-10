import numpy as np
import pytest
from scipy.stats import poisson

from coal_changepoints.model import ChangePointModel
from coal_changepoints.sampling import (
    merge_state, move_probabilities, sample_m1, sample_rjmcmc, split_state,
)


def test_split_merge_round_trip_and_jacobian():
    state = np.array([2.0, 0.4, 0.8])
    proposed, log_birth = split_state(state, 6, 0.37, 10)
    recovered, u, log_death = merge_state(proposed, 1, 10)
    np.testing.assert_allclose(recovered, state, rtol=1e-14)
    assert u == pytest.approx(0.37)
    assert log_birth + log_death == pytest.approx(0, abs=1e-14)

    def rate_map(h, auxiliary):
        return split_state([h], 3, auxiliary, 10)[0][1:]

    h, auxiliary, eps = 0.8, 0.37, 1e-6
    jacobian = np.column_stack([
        (rate_map(h + eps, auxiliary) - rate_map(h - eps, auxiliary)) / (2 * eps),
        (rate_map(h, auxiliary + eps) - rate_map(h, auxiliary - eps)) / (2 * eps),
    ])
    _, expected_log = split_state([h], 3, auxiliary, 10)
    assert abs(np.linalg.det(jacobian)) == pytest.approx(np.exp(expected_log), rel=1e-8)


def test_birth_death_log_acceptance_are_reciprocal():
    model = ChangePointModel([0, 1, 4, 10], 10, beta=2)
    original = np.array([2, 0.4, 0.8])
    new, birth_jacobian = split_state(original, 6, 0.37, 10)
    recovered, _, death_jacobian = merge_state(new, 1, 10)
    probabilities = move_probabilities(model)
    birth_proposal = np.log(probabilities[2, 3] / 2) - np.log(probabilities[1, 2] / 10)
    death_proposal = np.log(probabilities[1, 2] / 10) - np.log(probabilities[2, 3] / 2)
    birth = model.log_joint(new) - model.log_joint(original) + birth_proposal + birth_jacobian
    death = model.log_joint(recovered) - model.log_joint(new) + death_proposal + death_jacobian
    assert birth + death == pytest.approx(0, abs=1e-13)


def test_move_probabilities_respect_dimension_boundaries():
    for k_max in (0, 1, 30):
        probabilities = move_probabilities(ChangePointModel([], 10, k_max=k_max))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        assert np.all(probabilities >= 0)
        assert probabilities[0, 1] == probabilities[0, 3] == probabilities[-1, 2] == 0


def test_rjmcmc_seed_support_and_state_independence():
    model = ChangePointModel([0, 0.5, 0.5, 2, 3, 10], 10, beta=2, k_max=4)
    first, summary = sample_rjmcmc(model, np.random.default_rng(12), n_steps=600)
    second, second_summary = sample_rjmcmc(model, np.random.default_rng(12), n_steps=600)
    assert len(first) == 601
    assert all(model.valid_state(state) for state in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert summary == second_summary
    assert sum(item['attempted'] for item in summary.values()) == 600
    assert summary['height']['acceptance'] == 1
    assert len({id(state) for state in first}) == len(first)


def test_constant_model_sampler_matches_conjugate_moments():
    model = ChangePointModel([0, 1, 2], 3, alpha=2, beta=4, k_max=0)
    chain, _ = sample_rjmcmc(model, np.random.default_rng(24), n_steps=6000)
    draws = np.array(chain[1:])[:, 0]
    shape, rate = model.constant_posterior()
    standard_error = np.sqrt(shape) / rate / np.sqrt(len(draws))
    assert abs(draws.mean() - shape / rate) < 5 * standard_error
    assert draws.var() == pytest.approx(shape / rate**2, rel=0.08)


def test_m1_samples_match_collapsed_position_probability():
    model = ChangePointModel([0, 0.4, 0.8, 1, 1.2, 1.5, 2, 9, 10], 10, beta=1)
    chain, acceptance = sample_m1(model, np.random.default_rng(42), burn=1000,
                                 n_samples=10000, sigma_s=1)
    probability = np.exp(model.log_evidence_m1(upper=3) - model.log_evidence_m1())
    assert np.mean(chain[:, 0] <= 3) == pytest.approx(probability, abs=0.05)
    assert 0 < acceptance < 1
    assert all(model.valid_state(state) for state in chain)


def test_rjmcmc_recovers_nearly_uninformative_model_prior():
    model = ChangePointModel([], 0.01, beta=200, lam=2, k_max=3)
    chain, _ = sample_rjmcmc(model, np.random.default_rng(94), n_steps=14000)
    k = np.array([(len(state) - 1) // 2 for state in chain[1000:]])
    observed = np.bincount(k, minlength=4) / len(k)
    expected = poisson.pmf(np.arange(4), 2) / poisson.cdf(3, 2)
    np.testing.assert_allclose(observed, expected, atol=0.04)


def test_samplers_do_not_change_legacy_global_rng():
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    model = ChangePointModel([0, 1], 1, beta=1)
    sample_rjmcmc(model, np.random.default_rng(5), n_steps=20)
    sample_m1(model, np.random.default_rng(5), burn=0, n_samples=20, sigma_s=0.1)
    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize('n_steps', [0, -1, 2.5, True])
def test_invalid_iteration_counts_are_rejected(n_steps):
    with pytest.raises(ValueError):
        sample_rjmcmc(ChangePointModel([], 1), np.random.default_rng(1), n_steps=n_steps)
