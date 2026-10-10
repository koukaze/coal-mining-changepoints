'''Verify exported chain bookkeeping and reproducible analysis settings.'''

import json
from pathlib import Path

import numpy as np
import pytest

from coal_changepoints.pipeline import run_analysis


def test_pipeline_outputs_preserve_chain_draws(tmp_path):
    '''Saved ragged states and model probabilities must use the same retained draws.'''
    data = Path(__file__).resolve().parents[1] / 'data' / 'coal_intervals_days.csv'
    summary = run_analysis(data, tmp_path, steps=300, burn=50, n_chains=2,
                           m1_samples=500, rate_stride=5, seed=29)
    assert summary['retained_per_chain'] == 250
    assert summary['rate_summary_draws'] == 100
    assert summary['n_events'] == 191
    assert sum(move['attempted'] for move in summary['move_statistics'][0].values()) == 300
    assert json.loads((tmp_path / 'summary.json').read_text()) == summary
    with np.load(tmp_path / 'chains_and_posteriors.npz', allow_pickle=False) as archive:
        assert archive['k_chains'].shape == (2, 301)
        assert archive['m1_chain'].shape == (500, 3)
        for i in range(2):
            values, offsets = archive[f'chain_{i}_values'], archive[f'chain_{i}_offsets']
            assert offsets[0] == 0 and offsets[-1] == len(values)
            k = (np.diff(offsets) - 1) // 2
            np.testing.assert_array_equal(k, archive['k_chains'][i])
        retained = archive['k_chains'][:, 51:]
        expected = np.bincount(retained.ravel(), minlength=31) / retained.size
        np.testing.assert_array_equal(expected, archive['k_probabilities'])
        assert np.all(archive['rate_p5'] <= archive['rate_p25'])
        assert np.all(archive['rate_p75'] <= archive['rate_p95'])
    for stem in ('fixed_models', 'reversible_jump', 'trace_diagnostics'):
        assert (tmp_path / f'{stem}.png').stat().st_size > 1000
    assert len((tmp_path / 'posterior_k.csv').read_text().splitlines()) == 32
    assert len((tmp_path / 'rate_function.csv').read_text().splitlines()) == 401


def test_pipeline_rejects_invalid_retention_before_opening_inputs(tmp_path):
    '''Insufficient retained draws must fail before expensive sampling starts.'''
    with pytest.raises(ValueError, match='retained draws'):
        run_analysis('missing.csv', tmp_path, steps=100, burn=99)
