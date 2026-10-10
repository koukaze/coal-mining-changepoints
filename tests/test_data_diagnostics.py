'''Check event bookkeeping and scalar-chain diagnostics.'''

from pathlib import Path

import numpy as np
import pytest

from coal_changepoints.data import events_from_intervals, load_intervals
from coal_changepoints.diagnostics import effective_sample_size, rate_summary, split_rhat


def test_public_catalogue_includes_both_endpoint_events():
    '''All 191 events, including the coincident pair, must enter the likelihood.'''
    path = Path(__file__).resolve().parents[1] / 'data' / 'coal_intervals_days.csv'
    intervals = load_intervals(path)
    events, duration = events_from_intervals(intervals)
    assert len(intervals) == 190
    assert len(events) == 191
    assert events[0] == 0 and events[-1] == duration == 40549
    assert (np.diff(events) == 0).sum() == 1


def test_invalid_intervals_rejected():
    '''Bad intervals must not be silently sorted, rounded or removed.'''
    for values in ([], [0, 0], [2, -1], [2, np.nan], [[1, 2]]):
        with pytest.raises(ValueError):
            events_from_intervals(np.asarray(values))


def test_diagnostics_distinguish_independent_and_sticky_traces():
    '''Correlated draws have lower ESS, and separated chains have elevated R-hat.'''
    rng = np.random.default_rng(19)
    independent = rng.normal(size=(4, 4000))
    assert 0.99 < split_rhat(independent) < 1.02
    independent_ess = effective_sample_size(independent[0])
    assert independent_ess > 2500
    sticky = np.empty(4000)
    sticky[0] = 0
    for i in range(1, len(sticky)):
        sticky[i] = 0.95 * sticky[i-1] + independent[0, i]
    assert effective_sample_size(sticky) < independent_ess / 5
    assert split_rhat(independent + np.arange(4)[:, None] * 3) > 2
    assert np.isnan(effective_sample_size(np.ones(20)))
    assert np.isnan(split_rhat(np.ones((2, 20))))


def test_pointwise_rate_summary_uses_segment_boundaries():
    '''At a change point the rate belongs to the segment on its right.'''
    summary = rate_summary([np.array([5.0, 2.0, 6.0])], duration=10, n_grid=3)
    np.testing.assert_array_equal(summary['mean'], [2, 6, 6])
    np.testing.assert_array_equal(summary['p5'], summary['p95'])
