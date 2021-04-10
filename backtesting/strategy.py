import sys
from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from backtesting._util import _Data, _as_str, try_, _Indicator
from backtesting.broker import Broker
from backtesting.order import Order, _Orders
from backtesting.position import Position
from backtesting.trade import Trade


__pdoc__ = {'Strategy.__init__': False}


class Strategy(metaclass=ABCMeta):
    def __init__(self, broker, data, params):
        self._indicators = []
        self._broker: Broker = broker
        self._data: _Data = data
        self._params = self._check_params(params)

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    def __str__(self):
        params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(), map(_as_str, self._params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'."
                    'Strategy class should define parameters as class variables before they '
                    'can be optimized or run with.'
                )
            setattr(self, k, v)
        return params

    def I(
        self,
        func: Callable,
        *args,
        name=None,
        plot=True,
        overlay=None,
        color=None,
        scatter=False,
        **kwargs,
    ) -> np.ndarray:
        if name is None:
            params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
            func_name = _as_str(func)
            name = f'{func_name}({params})' if params else f'{func_name}'
        else:
            name = name.format(
                *map(_as_str, args),
                **dict(zip(kwargs.keys(), map(_as_str, kwargs.values()))),
            )

        try:
            value = func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f'I "{name}" errored with exception: {e}')

        if isinstance(value, pd.DataFrame):
            value = value.values.T

        if value is not None:
            value = try_(lambda: np.asarray(value, order='C'), None)
        is_arraylike = value is not None

        # Optionally flip the array if the user returned e.g. `df.values`
        if is_arraylike and np.argmax(value.shape) == 0:
            value = value.T

        if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[-1] != len(self._data.Close):
            raise ValueError(
                'Indicators must return (optionally a tuple of) numpy.arrays of same '
                f'length as `data` (data shape: {self._data.Close.shape}; indicator "{name}"'
                f'shape: {getattr(value, "shape" , "")}, returned value: {value})'
            )

        if plot and overlay is None and np.issubdtype(value.dtype, np.number):
            x = value / self._data.Close
            # By default, overlay if strong majority of indicator values
            # is within 30% of Close
            with np.errstate(invalid='ignore'):
                overlay = ((x < 1.4) & (x > 0.6)).mean() > 0.6

        value = _Indicator(
            value,
            name=name,
            plot=plot,
            overlay=overlay,
            color=color,
            scatter=scatter,
            index=self.data.index,
        )
        self._indicators.append(value)
        return value

    @abstractmethod
    def init(self):
        """
        Initialize the strategy.
        Override this method.
        Declare indicators (with `backtesting.backtesting.Strategy.I`).
        Precompute what needs to be precomputed or can be precomputed
        in a vectorized fashion before the strategy starts.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().init()
        """

    @abstractmethod
    def next(self):
        """
        Main strategy runtime method, called as each new
        `backtesting.backtesting.Strategy.data`
        instance (row; full candlestick bar) becomes available.
        This is the main method where strategy decisions
        upon data precomputed in `backtesting.backtesting.Strategy.init`
        take place.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().next()
        """

    class __FullEquity(float):
        def __repr__(self):
            return '.9999'

    _FULL_EQUITY = __FullEquity(1 - sys.float_info.epsilon)

    def buy(
        self,
        *,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
    ):
        """
        Place a new long order. For explanation of parameters, see `Order` and its properties.

        See also `Strategy.sell()`.
        """
        assert 0 < size < 1 or round(size) == size, 'size must be a positive fraction of equity, or a positive whole number of units'
        return self._broker.new_order(size, limit, stop, sl, tp)

    def sell(
        self,
        *,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
    ):
        """
        Place a new short order. For explanation of parameters, see `Order` and its properties.

        See also `Strategy.buy()`.
        """
        assert 0 < size < 1 or round(size) == size, 'size must be a positive fraction of equity, or a positive whole number of units'
        return self._broker.new_order(-size, limit, stop, sl, tp)

    @property
    def equity(self) -> float:
        """Current account equity (cash plus assets)."""
        return self._broker.equity

    @property
    def data(self) -> _Data:
        """
        Price data, roughly as passed into
        `backtesting.backtesting.Backtest.__init__`,
        but with two significant exceptions:

        * `data` is _not_ a DataFrame, but a custom structure
          that serves customized numpy arrays for reasons of performance
          and convenience. Besides OHLCV columns, `.index` and length,
          it offers `.pip` property, the smallest price unit of change.
        * Within `backtesting.backtesting.Strategy.init`, `data` arrays
          are available in full length, as passed into
          `backtesting.backtesting.Backtest.__init__`
          (for precomputing indicators and such). However, within
          `backtesting.backtesting.Strategy.next`, `data` arrays are
          only as long as the current iteration, simulating gradual
          price point revelation. In each call of
          `backtesting.backtesting.Strategy.next` (iteratively called by
          `backtesting.backtesting.Backtest` internally),
          the last array value (e.g. `data.Close[-1]`)
          is always the _most recent_ value.
        * If you need data arrays (e.g. `data.Close`) to be indexed
          **Pandas series**, you can call their `.s` accessor
          (e.g. `data.Close.s`). If you need the whole of data
          as a **DataFrame**, use `.df` accessor (i.e. `data.df`).
        """
        return self._data

    @property
    def position(self) -> 'Position':
        """Instance of `backtesting.backtesting.Position`."""
        return self._broker.position

    @property
    def orders(self) -> 'Tuple[Order, ...]':
        """List of orders (see `Order`) waiting for execution."""
        return _Orders(self._broker.orders)

    @property
    def trades(self) -> 'Tuple[Trade, ...]':
        """List of active trades (see `Trade`)."""
        return tuple(self._broker.trades)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        """List of settled trades (see `Trade`)."""
        return tuple(self._broker.closed_trades)
