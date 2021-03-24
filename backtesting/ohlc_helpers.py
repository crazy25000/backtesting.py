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
    """
    Apply `func` (such as an indicator) to `series`, resampled to
    a time frame specified by `rule`. When called from inside
    `backtesting.backtesting.Strategy.init`,
    the result (returned) series will be automatically wrapped in
    `backtesting.backtesting.Strategy.Indicator`
    wrapper method.

    `rule` is a valid [Pandas offset string] indicating
    a time frame to resample `series` to.

    [Pandas offset string]: \
http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    `func` is the indicator function to apply on the resampled series.

    `series` is a data series (or array), such as any of the
    `backtesting.backtesting.Strategy.data` series. Due to pandas
    resampling limitations, this only works when input series
    has a datetime index.

    `agg` is the aggregation function to use on resampled groups of data.
    Valid values are anything accepted by `pandas/resample/.agg()`.
    Default value for dataframe input is `OHLCV_AGG` dictionary.
    Default value for series input is the appropriate entry from `OHLCV_AGG`
    if series has a matching name, or otherwise the value `"last"`,
    which is suitable for closing prices,
    but you might prefer another (e.g. `"max"` for peaks, or similar).

    Finally, any `*args` and `**kwargs` that are not already eaten by
    implicit `backtesting.backtesting.Strategy.Indicator` call
    are passed to `func`.

    For example, if we have a typical moving average function
    `SMA(values, lookback_period)`, _hourly_ data source, and need to
    apply the moving average MA(10) on a _daily_ time frame,
    but don't want to plot the resulting indicator, we can do:

        class System(Strategy):
            def init(self):
                self.sma = resample_apply(
                    'D', SMA, self.data.Close, 10, plot=False)

    The above short snippet is roughly equivalent to:

        class System(Strategy):
            def init(self):
                # Strategy exposes `self.data` as raw NumPy arrays.
                # Let's convert closing prices back to pandas Series.
                close = self.data.Close.s

                # Resample to daily resolution. Aggregate groups
                # using their last value (i.e. closing price at the end
                # of the day). Notice `label='right'`. If it were set to
                # 'left' (default), the strategy would exhibit
                # look-ahead bias.
                daily = close.resample('D', label='right').agg('last')

                # We apply SMA(10) to daily close prices,
                # then reindex it back to original hourly index,
                # forward-filling the missing values in each day.
                # We make a separate function that returns the final
                # indicator array.
                def SMA(series, n):
                    from backtesting.test import SMA
                    return SMA(series, n).reindex(close.index).ffill()

                # The result equivalent to the short example above:
                self.sma = self.Indicator(SMA, daily, 10, plot=False)

    """
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
