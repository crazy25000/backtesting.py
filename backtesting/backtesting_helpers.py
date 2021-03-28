import os
import warnings
from numbers import Number
from typing import List, Sequence, Dict

import numpy as np
import pandas as pd
from skopt.space import Integer, Real, Categorical

from backtesting.backtesting import Strategy
from backtesting._util import try_
from backtesting.broker import _OutOfMoneyError
from backtesting.metrics import compute_stats


def get_grid_frac(max_tries, _grid_size):
    if max_tries is None:
        return 1
    elif 0 < max_tries <= 1:
        return max_tries

    return max_tries / _grid_size()


def get_max_tries(max_tries, _grid_size):
    if max_tries is None:
        return 200
    elif 0 < max_tries <= 1:
        return max(1, int(max_tries * _grid_size()))

    return max_tries


class AttrDict(dict):
    def __getattr__(self, item):
        return self[item]


def _tuple(x: List[str]) -> List[str]:
    return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)


def validate_and_get_data(cash, data):
    data = data.copy(deep=False)
    largest_numeric_index = data.index.is_numeric() and (data.index > pd.Timestamp('1975').timestamp()).mean() > 0.8
    if not isinstance(data.index, pd.DatetimeIndex) and not isinstance(data.index, pd.RangeIndex) and largest_numeric_index:
        try:
            data.index = pd.to_datetime(data.index, infer_datetime_format=True)
        except ValueError:
            pass
    if 'Volume' not in data:
        data['Volume'] = np.nan
    if len(data) == 0:
        raise ValueError('OHLC `data` is empty')
    if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
        raise ValueError(
            '`data` must be a pandas.DataFrame with columns ' "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'")
    if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
        raise ValueError(
            'Some OHLC values are missing (NaN). '
            'Please strip those lines with `df.dropna()` or '
            'fill them in with `df.interpolate()` or whatever.'
        )
    if np.any(data['Close'] > cash):
        warnings.warn(
            'Some prices are larger than initial cash value. Note that fractional '
            'trading is not supported. If you want to trade Bitcoin, '
            'increase initial cash, or trade μBTC or satoshis instead (GH-134).',
            stacklevel=2,
        )
    if not data.index.is_monotonic_increasing:
        warnings.warn('Data index is not sorted in ascending order. Sorting.', stacklevel=2)
        data = data.sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        warnings.warn(
            'Data index is not datetime. Assuming simple periods, but `pd.DateTimeIndex` is advised.',
            stacklevel=2,
        )
    return data


def validate_instance_types(commission, data, strategy):
    if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
        raise TypeError('`strategy` must be a Strategy sub-type')
    if not isinstance(data, pd.DataFrame):
        raise TypeError('`data` must be a pandas.DataFrame with columns')
    if not isinstance(commission, Number):
        raise TypeError('`commission` must be a float value, percent of entry order price')


def get_perf_metrics(broker, data, strategy, self_data):
    data._set_length(len(self_data))
    equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
    results = compute_stats(trades=broker.closed_trades, equity=equity, ohlc_data=self_data, risk_free_rate=0.0)
    results.loc['_strategy'] = strategy
    return results


def loop_through_data(broker, data, indicator_attrs, start, strategy, self_data):
    with np.errstate(invalid='ignore'):
        for i in range(start, len(self_data)):
            data._set_length(i + 1)
            for attr, indicator in indicator_attrs:
                setattr(strategy, attr, indicator[..., : i + 1])
            try:
                broker.next()
            except _OutOfMoneyError:
                break

            strategy.next()
        else:
            for trade in broker.trades:
                trade.close()
            if start < len(self_data):
                try_(broker.next, exception=_OutOfMoneyError)

        return get_perf_metrics(broker, data, strategy, self_data)


def construct_dimensions(kwargs) -> list:
    dimensions = []
    for key, values in kwargs.items():
        values = np.asarray(values)
        if values.dtype.kind in 'mM':  # timedelta, datetime64
            values = values.astype(int)

        if values.dtype.kind in 'iumM':
            dimensions.append(Integer(low=values.min(), high=values.max(), name=key))
        elif values.dtype.kind == 'f':
            dimensions.append(Real(low=values.min(), high=values.max(), name=key))
        else:
            dimensions.append(Categorical(values.tolist(), name=key, transform='onehot'))
    return dimensions


def _batch(seq: List[Dict[str, str]]):
    n = np.clip(len(seq) // (os.cpu_count() or 1), 5, 300)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def def_constraint(_) -> bool:
    return True


def get_constraint_func(constraint):
    if constraint is None:
        constraint = def_constraint(None)
    elif not callable(constraint):
        raise TypeError(
            '`constraint` must be a function that accepts a dict '
            'of strategy parameters and returns a bool whether '
            'the combination of parameters is admissible or not'
        )

    return constraint
