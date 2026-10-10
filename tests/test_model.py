import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.stats import gamma, poisson

from coal_changepoints.model import ChangePointModel


def test_endpoint_and_coincident_events_use_right_segments():
    model = ChangePointModel([0, 1, 2, 2, 4, 10], 10)
    np.testing.assert_array_equal(model.segment_counts([2, 3]), [2, 2, 2])
    state = [2, 3, 0.4, 0.7, 1.2]
    expected = 2 * np.log(0.4 * 0.7 * 1.2) - (2 * 0.4 + 0.7 + 7 * 1.2)
    assert model.log_likelihood(state) == pytest.approx(expected)
    assert model.log_likelihood([0.3]) == pytest.approx(6 * np.log(0.3) - 3)


@pytest.mark.parametrize('events,duration', [([1, 0], 2), ([0, 3], 2), ([-1], 2), ([np.nan], 2), ([0], 0)])
def test_invalid_event_data_are_rejected(events, duration):
    with pytest.raises(ValueError):
        ChangePointModel(events, duration)


def test_model_copies_event_data():
    events = np.array([0.0, 2.0])
    model = ChangePointModel(events, 2)
    events[1] = 1
    assert model.events[1] == 2
    assert not model.events.flags.writeable


def test_position_priors_have_unit_integrals():
    model = ChangePointModel([], 2)
    integral_1, _ = quad(lambda s: np.exp(model.log_position_prior([s])), 0, 2)
    integral_2, _ = dblquad(lambda s2, s1: np.exp(model.log_position_prior([s1, s2])),
                           0, 2, lambda s1: s1, lambda s1: 2)
    assert np.exp(model.log_position_prior([])) == pytest.approx(1)
    assert integral_1 == pytest.approx(1)
    assert integral_2 == pytest.approx(1)


def test_joint_prior_includes_normalized_truncated_model_probability():
    model = ChangePointModel([], 10, alpha=2, beta=4, lam=2, k_max=3)
    state = np.array([3, 0.4, 0.8])
    expected = (poisson.logpmf(1, 2) - poisson.logcdf(3, 2)
                + np.log(6 * 3 * 7 / 10**3)
                + gamma.logpdf(state[1:], 2, scale=0.25).sum())
    assert model.log_prior(state) == pytest.approx(expected)


def test_birth_position_ratio_matches_normalized_density():
    model = ChangePointModel([], 10)
    difference = model.log_position_prior([2, 6]) - model.log_position_prior([2])
    expected_ratio = (5 * 4 / 10**2) * (4 * 4 / 8)
    assert np.exp(difference) == pytest.approx(expected_ratio)


def test_constant_rate_evidence_matches_direct_integration():
    model = ChangePointModel([0, 1, 2], 3, alpha=2, beta=4)
    reference, _ = quad(lambda h: np.exp(3 * np.log(h) - 3 * h)
                        * gamma.pdf(h, 2, scale=0.25), 0, np.inf)
    assert np.exp(model.log_evidence_m0()) == pytest.approx(reference, rel=1e-10)
    assert model.constant_posterior() == (5, 7)


def test_collapsed_m1_evidence_and_partial_integrals():
    model = ChangePointModel([], 10, alpha=2, beta=4)
    reference, _ = quad(lambda s: 6 * s * (10 - s) / 10**3
                        * (4 / (4 + s))**2 * (4 / (14 - s))**2, 0, 10)
    total = model.log_evidence_m1()
    assert np.exp(total) == pytest.approx(reference, rel=1e-10)
    assert np.exp(model.log_evidence_m1(upper=5) - total) == pytest.approx(0.5)
    assert model.log_evidence_m1(upper=0) == -np.inf


@pytest.mark.parametrize('state', [[], [1, 2], [0], [10, 1, 1], [2, 2, 1, 1, 1], [2, -1, 1]])
def test_invalid_states_have_zero_density(state):
    model = ChangePointModel([0, 10], 10)
    assert not model.valid_state(state)
    assert model.log_joint(state) == -np.inf
