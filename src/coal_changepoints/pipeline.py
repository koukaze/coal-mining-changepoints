'''Run fixed-dimension and reversible-jump inference with reproducible outputs.'''

from collections.abc import Callable
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import events_from_intervals, load_intervals
from .diagnostics import effective_sample_size, integrated_rate, k_trace, rate_summary, split_rhat
from .model import ChangePointModel
from .plotting import fixed_model_figure, reversible_jump_figure, trace_figure
from .sampling import sample_m1, sample_rjmcmc


def _json_number(value: float) -> float | None:
    '''Represent undefined diagnostics as JSON null.'''
    return float(value) if np.isfinite(value) else None


def run_analysis(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 1123,
    steps: int = 100000,
    burn: int = 10000,
    n_chains: int = 4,
    m1_samples: int = 50000,
    rate_stride: int = 20,
    progress: Callable[[str], None] | None = None,
) -> dict:
    '''Fit all model families and save chains, summaries and diagnostic figures.
    Args
        data_path (str or Path): Public one-column interval CSV.
        output_dir (str or Path): Destination for generated files.
        seed (int): SeedSequence entropy used to spawn independent random streams.
        steps (int): Number of RJMCMC transitions per chain.
        burn (int): Discarded transitions per chain; also the M1 burn-in count.
        n_chains (int): Number of independent RJMCMC chains, at least two.
        m1_samples (int): Retained draws from the one-change-point sampler.
        rate_stride (int): Subsampling interval for rate figures only, after burn-in.
        progress (Callable or None): Optional callback for chain-level progress messages.
    Returns
        dict: Settings, model evidence, scalar diagnostics and posterior model probabilities.
    '''
    for name, value in (('steps', steps), ('burn', burn), ('n_chains', n_chains),
                        ('m1_samples', m1_samples), ('rate_stride', rate_stride)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name} must be an integer.')
    if burn < 0 or steps - burn < 4 or n_chains < 2 or m1_samples < 4 or rate_stride < 1:
        raise ValueError('Use at least two chains, four retained draws, non-negative burn and positive rate stride.')
    intervals = load_intervals(data_path)
    events, duration = events_from_intervals(intervals)
    model = ChangePointModel(events, duration)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(seed).spawn(n_chains + 1)
    if progress:
        progress('Sampling the one-change-point model.')
    m1, m1_acceptance = sample_m1(model, np.random.default_rng(streams[0]), burn=burn, n_samples=m1_samples)
    chains, moves = [], []
    for i in range(n_chains):
        if progress:
            progress(f'RJMCMC chain {i+1}/{n_chains}: {steps:,} transitions.')
        chain, move_summary = sample_rjmcmc(model, np.random.default_rng(streams[i+1]), n_steps=steps)
        chains.append(chain)
        moves.append(move_summary)
    k_chains = np.array([k_trace(chain) for chain in chains])
    count_chains = np.array([integrated_rate(chain, duration) for chain in chains])
    # Index zero is the initial state, not a completed transition.
    retained_k = k_chains[:, burn+1:]
    retained_counts = count_chains[:, burn+1:]
    k_values = np.arange(model.k_max + 1)
    counts = np.bincount(retained_k.ravel(), minlength=model.k_max + 1)
    probabilities = counts / counts.sum()
    rate_states = [state for chain in chains for state in chain[burn+1::rate_stride]]
    rate = rate_summary(rate_states, duration)
    log_z0, log_z1 = model.log_evidence_m0(), model.log_evidence_m1()
    summary = {
        'n_events': len(events), 'n_intervals': len(intervals), 'duration_days': duration,
        'alpha': model.alpha, 'beta_days': model.beta, 'poisson_mean': model.lam, 'k_max': model.k_max,
        'seed': seed, 'random_generator': 'NumPy PCG64 with SeedSequence.spawn',
        'steps_per_chain': steps, 'burn_transitions': burn, 'n_chains': n_chains,
        'retained_per_chain': steps - burn, 'rate_stride': rate_stride,
        'rate_summary_draws': len(rate_states), 'm1_retained_draws': m1_samples,
        'm1_position_acceptance': m1_acceptance,
        'm1_posterior_median': {'position_days': float(np.median(m1[:, 0])),
                                'rate_before_per_day': float(np.median(m1[:, 1])),
                                'rate_after_per_day': float(np.median(m1[:, 2]))},
        'log_evidence_m0': log_z0, 'log_evidence_m1': log_z1,
        'log_bayes_factor_m1_over_m0': log_z1 - log_z0,
        'log_posterior_odds_m1_over_m0': log_z1 - log_z0 + float(np.log(model.lam)),
        'posterior_k_map': int(k_values[np.argmax(probabilities)]),
        'posterior_k_probabilities': probabilities.tolist(),
        'diagnostics': {
            'k_split_rhat': _json_number(split_rhat(retained_k)),
            'integrated_rate_split_rhat': _json_number(split_rhat(retained_counts)),
            'k_ess_per_chain': [_json_number(effective_sample_size(row)) for row in retained_k],
            'integrated_rate_ess_per_chain': [_json_number(effective_sample_size(row)) for row in retained_counts],
        },
        'move_statistics': moves,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    arrays = dict(m1_chain=m1, k_chains=k_chains, integrated_rate_chains=count_chains,
                  k_values=k_values, k_probabilities=probabilities, **{f'rate_{k}': v for k, v in rate.items()})
    for i, chain in enumerate(chains):
        lengths = np.array([len(state) for state in chain])
        arrays[f'chain_{i}_offsets'] = np.r_[0, lengths.cumsum()]
        arrays[f'chain_{i}_values'] = np.concatenate(chain)
    np.savez_compressed(output_dir / 'chains_and_posteriors.npz', **arrays)
    with (output_dir / 'posterior_k.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['k', 'probability'])
        writer.writerows(zip(k_values, probabilities))
    with (output_dir / 'rate_function.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_days', 'mean_per_day', 'p5_per_day', 'p25_per_day', 'p75_per_day', 'p95_per_day'])
        writer.writerows(zip(*(rate[key] for key in ('time_days', 'mean', 'p5', 'p25', 'p75', 'p95'))))
    for name, figure in (
        ('fixed_models', fixed_model_figure(model, m1)),
        ('reversible_jump', reversible_jump_figure(k_values, probabilities, rate)),
        ('trace_diagnostics', trace_figure(k_chains, count_chains, burn)),
    ):
        figure.savefig(output_dir / f'{name}.png', dpi=180, bbox_inches='tight')
        figure.savefig(output_dir / f'{name}.svg', bbox_inches='tight')
        plt.close(figure)
    return summary
