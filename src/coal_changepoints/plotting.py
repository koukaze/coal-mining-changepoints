'''Plot fixed-dimension inference and reversible-jump posterior summaries.'''

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .model import ChangePointModel

DAYS_PER_YEAR = 365.25


def fixed_model_figure(model: ChangePointModel, chain: np.ndarray) -> Figure:
    '''Compare the event record, change-point posterior and segment rates.
    Args
        model (ChangePointModel): Event catalogue and prior settings.
        chain (ndarray): Post-burn-in M1 samples, columns position, first rate, second rate.
    Returns
        Figure: Three panels with elapsed years and rates per year.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    years = model.events / DAYS_PER_YEAR
    axes[0].step(years, np.arange(1, len(years) + 1), where='post', color='#176b87', label='Recorded events')
    shape, rate = model.constant_posterior()
    time = np.linspace(0, model.duration, 400)
    axes[0].plot(time / DAYS_PER_YEAR, time * shape / rate, '--', color='#e07a5f', label='Constant-rate posterior mean')
    axes[0].set(xlabel='Elapsed time [years]', ylabel='Cumulative events', title='Coal-mining disaster record')
    axes[0].legend(fontsize=8)
    axes[1].hist(chain[:, 0] / DAYS_PER_YEAR, bins=45, density=True, alpha=0.5, color='#176b87', label='MCMC')
    limits = np.percentile(chain[:, 0], [0.05, 99.95])
    grid = np.linspace(*limits, 1000)
    density = np.exp(np.array([model.log_marginal_m1(s) for s in grid]) - model.log_evidence_m1())
    axes[1].plot(grid / DAYS_PER_YEAR, density * DAYS_PER_YEAR, color='#e07a5f', linewidth=1.3, label='Integrated-rate reference')
    axes[1].set(xlabel='Change-point time [years]', ylabel='Density [year$^{-1}$]', title='One-change-point position')
    axes[1].legend(fontsize=8)
    for column, label, colour in ((1, 'Before change', '#176b87'), (2, 'After change', '#e07a5f')):
        axes[2].hist(chain[:, column] * DAYS_PER_YEAR, bins=45, density=True, histtype='step', linewidth=1.5, color=colour, label=label)
    axes[2].set(xlabel='Rate [events / year]', ylabel='Posterior density', title='One-change-point rates')
    axes[2].legend(fontsize=8)
    return fig


def reversible_jump_figure(k_values: np.ndarray, probabilities: np.ndarray, rate: dict[str, np.ndarray]) -> Figure:
    '''Show posterior model probabilities and pointwise rate uncertainty.
    Args
        k_values (ndarray): Integer numbers of change points.
        probabilities (ndarray): Empirical posterior probabilities aligned with k_values.
        rate (dict): Time grid and pointwise rate summaries in days and events per day.
    Returns
        Figure: Model-probability and rate-function panels.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(k_values, probabilities, color='#176b87', width=0.75)
    axes[0].set(xlabel='Number of change points $k$', ylabel='Posterior probability', title='RJMCMC model frequencies')
    last_visited = max(6, int(k_values[probabilities > 0].max(initial=0)))
    axes[0].set_xticks(np.arange(last_visited + 1))
    axes[0].set_xlim(-0.75, last_visited + 0.75)
    years = rate['time_days'] / DAYS_PER_YEAR
    axes[1].fill_between(years, rate['p5'] * DAYS_PER_YEAR, rate['p95'] * DAYS_PER_YEAR,
                         color='#176b87', alpha=0.15, label='90% pointwise interval')
    axes[1].fill_between(years, rate['p25'] * DAYS_PER_YEAR, rate['p75'] * DAYS_PER_YEAR,
                         color='#176b87', alpha=0.3, label='50% pointwise interval')
    axes[1].plot(years, rate['mean'] * DAYS_PER_YEAR, color='#176b87', label='Posterior mean')
    axes[1].set(xlabel='Elapsed time [years]', ylabel='Rate [events / year]', title='Model-averaged event rate')
    axes[1].legend(fontsize=8)
    return fig


def trace_figure(k_chains: np.ndarray, count_chains: np.ndarray, burn: int) -> Figure:
    '''Inspect model-index and integrated-rate mixing across independent chains.
    Args
        k_chains (ndarray): Model-index traces, shape (n_chains, n_steps+1).
        count_chains (ndarray): Integrated-rate traces with the same shape.
        burn (int): Number of transition draws discarded after the initial state.
    Returns
        Figure: Two trace panels with the burn-in region shaded.
    '''
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, constrained_layout=True)
    stride = max(1, k_chains.shape[1] // 4000)
    iteration = np.arange(k_chains.shape[1])[::stride]
    for i, (k_values, counts) in enumerate(zip(k_chains, count_chains)):
        axes[0].plot(iteration, k_values[::stride], alpha=0.6, linewidth=0.5, label=f'Chain {i+1}')
        axes[1].plot(iteration, counts[::stride], alpha=0.6, linewidth=0.5)
    for axis in axes:
        axis.axvspan(0, burn, color='grey', alpha=0.12)
    axes[0].set(ylabel='Change points $k$', title=f'RJMCMC scalar traces (display stride: {stride})')
    axes[0].legend(ncol=min(4, len(k_chains)), fontsize=8)
    axes[1].set(xlabel='Transition number', ylabel='Integrated rate [events]')
    return fig
