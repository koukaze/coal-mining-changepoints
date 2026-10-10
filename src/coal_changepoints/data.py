'''Load the public coal-mining intervals and construct an event catalogue.'''

from pathlib import Path

import numpy as np


def load_intervals(filepath: str | Path) -> np.ndarray:
    '''Read a one-column CSV of elapsed days between successive events.
    Args
        filepath (str or Path): CSV with an interval_days header.
    Returns
        ndarray: Non-negative intervals in days, in chronological order.
    '''
    values = np.genfromtxt(filepath, delimiter=',', names=True, dtype=float)
    if values.dtype.names != ('interval_days',):
        raise ValueError('The CSV must have exactly one column named interval_days.')
    intervals = np.atleast_1d(values['interval_days'])
    if intervals.size == 0 or not np.isfinite(intervals).all() or (intervals < 0).any():
        raise ValueError('Intervals must be finite and non-negative.')
    return intervals


def events_from_intervals(intervals: np.ndarray) -> tuple[np.ndarray, float]:
    '''Include the first event at day zero and accumulate the remaining intervals.
    Args
        intervals (ndarray): One-dimensional non-negative inter-event intervals in days.
    Returns
        tuple: Event times including zero, and the elapsed observation duration in days.
    '''
    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim != 1 or intervals.size == 0 or not np.isfinite(intervals).all() or (intervals < 0).any():
        raise ValueError('Provide a non-empty one-dimensional array of non-negative intervals.')
    events = np.concatenate(([0.0], np.cumsum(intervals)))
    duration = float(events[-1])
    if duration <= 0:
        raise ValueError('The elapsed observation duration must be positive.')
    return events, duration
