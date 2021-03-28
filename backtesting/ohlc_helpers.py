from collections import OrderedDict
from inspect import currentframe
from typing import Optional, Callable, Sequence, Union

import numpy as np
import pandas as pd

from backtesting import Strategy
from backtesting._util import _Array, _as_str

OHLCV_AGG = OrderedDict(
    (
        ('Open', 'first'),
        ('High', 'max'),
        ('Low', 'min'),
        ('Close', 'last'),
        ('Volume', 'sum'),
    )
)


def resample_apply(
    rule: str,
    func: Optional[Callable[..., Sequence]],
    series: Union[pd.Series, pd.DataFrame, _Array],
    *args,
    agg: Union[str, dict] = None,
    **kwargs,
):
    if func is None:

        def func(x, *_, **__):
            return x

    if not isinstance(series, (pd.Series, pd.DataFrame)):
        assert isinstance(series, _Array), 'resample_apply() takes either a `pd.Series`, `pd.DataFrame`, ' 'or a `Strategy.data.*` array'
        series = series.s

    if agg is None:
        agg = OHLCV_AGG.get(getattr(series, 'name', None), 'last')
        if isinstance(series, pd.DataFrame):
            agg = {column: OHLCV_AGG.get(column, 'last') for column in series.columns}

    resampled = series.resample(rule, label='right').agg(agg).dropna()
    resampled.name = _as_str(series) + '[' + rule + ']'

    # Check first few stack frames if we are being called from
    # inside Strategy.init, and if so, extract Strategy.Indicator wrapper.
    frame, level = currentframe(), 0
    while frame and level <= 3:
        frame = frame.f_back
        level += 1
        if isinstance(frame.f_locals.get('self'), Strategy):  # type: ignore
            strategy_I = frame.f_locals['self'].Indicator  # type: ignore
            break
    else:

        def strategy_I(func, *args, **kwargs):
            return func(*args, **kwargs)

    def wrap_func(resampled, *args, **kwargs):
        result = func(resampled, *args, **kwargs)
        if not isinstance(result, pd.DataFrame) and not isinstance(result, pd.Series):
            result = np.asarray(result)
            if result.ndim == 1:
                result = pd.Series(result, name=resampled.name)
            elif result.ndim == 2:
                result = pd.DataFrame(result.T)
        # Resample back to data index
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = resampled.index
        result = result.reindex(index=series.index.union(resampled.index), method='ffill').reindex(series.index)
        return result

    wrap_func.__name__ = func.__name__  # type: ignore

    array = strategy_I(wrap_func, resampled, *args, **kwargs)
    return array


def random_ohlc_data(example_data: pd.DataFrame, *, frac=1.0, random_state: int = None) -> pd.DataFrame:
    """
    OHLC data generator. The generated OHLC data has basic
    [descriptive statistics](https://en.wikipedia.org/wiki/Descriptive_statistics)
    similar to the provided `example_data`.

    `frac` is a fraction of data to sample (with replacement). Values greater
    than 1 result in oversampling.

    Such random data can be effectively used for stress testing trading
    strategy robustness, Monte Carlo simulations, significance testing, etc.

    >>> from backtesting.test import EURUSD
    >>> ohlc_generator = random_ohlc_data(EURUSD)
    >>> next(ohlc_generator)  # returns new random data
    ...
    >>> next(ohlc_generator)  # returns new random data
    ...
    """

    def shuffle(x):
        return x.sample(frac=frac, replace=frac > 1, random_state=random_state)

    if len(example_data.columns.intersection({'Open', 'High', 'Low', 'Close'})) != 4:
        raise ValueError('`data` must be a pandas.DataFrame with columns ' "'Open', 'High', 'Low', 'Close'")
    while True:
        df = shuffle(example_data)
        df.index = example_data.index
        padding = df.Close - df.Open.shift(-1)
        gaps = shuffle(example_data.Open.shift(-1) - example_data.Close)
        deltas = (padding + gaps).shift(1).fillna(0).cumsum()
        for key in ('Open', 'High', 'Low', 'Close'):
            df[key] += deltas
        yield df
