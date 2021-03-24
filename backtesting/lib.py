"""
Collection of common building blocks, helper auxiliary functions and
composable strategy classes for reuse.

Intended for simple missing-link procedures, not reinventing
of better-suited, state-of-the-art, fast libraries,
such as TA-Lib, Tulipy, PyAlgoTrade, NumPy, SciPy ...

Please raise ideas for additions to this collection on the [issue tracker].

[issue tracker]: https://github.com/kernc/backtesting.py
"""

from collections import OrderedDict
from itertools import compress
from numbers import Number
from typing import Sequence, Union

import numpy as np
import pandas as pd

from . import Strategy

__pdoc__ = {}

"""Dictionary of rules for aggregating resampled OHLCV data frames,
e.g.

    df.resample('4H', label='right').agg(OHLCV_AGG)
"""

TRADES_AGG = OrderedDict(
    (
        ('Size', 'sum'),
        ('EntryBar', 'first'),
        ('ExitBar', 'last'),
        ('EntryPrice', 'mean'),
        ('ExitPrice', 'mean'),
        ('PnL', 'sum'),
        ('ReturnPct', 'mean'),
        ('EntryTime', 'first'),
        ('ExitTime', 'last'),
        ('Duration', 'sum'),
    )
)
"""Dictionary of rules for aggregating resampled trades data,
e.g.

    stats['_trades'].resample('1D', on='ExitTime',
                              label='right').agg(TRADES_AGG)
"""

_EQUITY_AGG = {
    'Equity': 'last',
    'DrawdownPct': 'max',
    'DrawdownDuration': 'max',
}


def barssince(condition: Sequence[bool], default=np.inf) -> int:
    """
    Return the number of bars since `condition` sequence was last `True`,
    or if never, return `default`.

        >>> barssince(self.data.Close > self.data.Open)
        3
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


def cross(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` and `series2` just crossed (either
    direction).

        >>> cross(self.data.Close, self.sma)
        True

    """
    return crossover(series1, series2) or crossover(series2, series1)


def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` just crossed over
    `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    series1 = series1.values if isinstance(series1, pd.Series) else (series1, series1) if isinstance(series1, Number) else series1
    series2 = series2.values if isinstance(series2, pd.Series) else (series2, series2) if isinstance(series2, Number) else series2
    try:
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]
    except IndexError:
        return False


def quantile(series: Sequence, quantile: Union[None, float] = None):
    """
    If `quantile` is `None`, return the quantile _rank_ of the last
    value of `series` wrt former series values.

    If `quantile` is a value between 0 and 1, return the _value_ of
    `series` at this quantile. If used to working with percentiles, just
    divide your percentile amount with 100 to obtain quantiles.

        >>> quantile(self.data.Close[-20:], .1)
        162.130
        >>> quantile(self.data.Close)
        0.13
    """
    if quantile is None:
        try:
            last, series = series[-1], series[:-1]
            return np.mean(series < last)
        except IndexError:
            return np.nan
    assert 0 <= quantile <= 1, 'quantile must be within [0, 1]'
    return np.nanpercentile(series, quantile * 100)


# Prevent pdoc3 documenting __init__ signature of Strategy subclasses
for cls in list(globals().values()):
    if isinstance(cls, type) and issubclass(cls, Strategy):
        __pdoc__[f'{cls.__name__}.__init__'] = False


# NOTE: Don't put anything below this __all__ list

__all__ = [
    getattr(v, '__name__', k)
    for k, v in globals().items()  # export
    if (
        (callable(v) and v.__module__ == __name__ or k.isupper())  # callables from this module
        and not getattr(v, '__name__', k).startswith('_')  # or CONSTANTS
    )
]  # neither marked internal

# NOTE: Don't put anything below here. See above.
