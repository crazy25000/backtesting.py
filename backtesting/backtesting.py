import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from itertools import repeat, product, compress
from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm

from backtesting.broker import Broker
from backtesting.strategy import Strategy
from ._plotting import plot
from ._util import _Indicator, _Data
from .backtesting_helpers import (
    get_grid_frac,
    get_max_tries,
    AttrDict,
    _tuple,
    validate_and_get_data,
    validate_instance_types,
    loop_through_data, construct_dimensions,
)


class Backtest:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Type[Strategy],
        *,
        cash: float = 10_000,
        commission: float = 0.0,
        margin: float = 1.0,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
    ):
        validate_instance_types(commission, data, strategy)
        self._data = validate_and_get_data(cash, data)
        self._broker = partial(
            Broker,
            cash=cash,
            commission=commission,
            margin=margin,
            trade_on_close=trade_on_close,
            hedging=hedging,
            exclusive_orders=exclusive_orders,
            index=self._data.index,
        )
        self._strategy = strategy
        self._results: Union[pd.Series, None] = None

    def run(self, **kwargs) -> pd.Series:
        data = _Data(self._data.copy(deep=False))
        broker: Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)

        strategy.init()
        data._update()
        indicator_attrs = {attr: indicator for attr, indicator in strategy.__dict__.items() if isinstance(indicator, _Indicator)}.items()
        start = 1 + max(
            (np.isnan(indicator.astype(float)).argmin(axis=-1).max() for _, indicator in indicator_attrs),
            default=0,
        )

        self._results = loop_through_data(broker, data, indicator_attrs, start, strategy, self._data)

        return self._results

    def optimize(
        self,
        *,
        maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
        method: str = 'grid',
        max_tries: Union[int, float] = None,
        constraint: Callable[[dict], bool] = None,
        return_heatmap: bool = False,
        return_optimization: bool = False,
        random_state: int = None,
        **kwargs,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, dict]]:
        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            stats = self._results if self._results is not None else self.run()
            if maximize not in stats:
                raise ValueError('`maximize`, if str, must match a key in pd.Series result of backtest.run()')

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]

        elif not callable(maximize):
            raise TypeError(
                '`maximize` must be str (a field of backtest.run() result '
                'Series) or a function that accepts result Series '
                'and returns a number; the higher the better'
            )

        have_constraint = bool(constraint)
        if constraint is None:

            def constraint(_):
                return True

        elif not callable(constraint):
            raise TypeError(
                '`constraint` must be a function that accepts a dict '
                'of strategy parameters and returns a bool whether '
                'the combination of parameters is admissible or not'
            )

        if return_optimization and method != 'skopt':
            raise ValueError("return_optimization=True only valid if method='skopt'")

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(f"Optimization variable '{k}' is passed no " f'optimization values: {k}={v}')

        def _grid_size() -> np.int64:
            size = np.prod([len(_tuple(v)) for v in kwargs.values()])
            if size < 10_000 and have_constraint:
                size = sum(1 for p in product(*(zip(repeat(k), _tuple(v)) for k, v in kwargs.items())) if constraint(AttrDict(p)))
            return size

        def _optimize_grid() -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
            rand = np.random.RandomState(random_state).random
            grid_frac = get_grid_frac(max_tries, _grid_size)
            param_combos = [
                dict(params)  # back to dict so it pickles
                for params in (AttrDict(params) for params in product(*(zip(repeat(k), _tuple(v)) for k, v in kwargs.items())))
                if constraint(params) and rand() <= grid_frac  # type: ignore
            ]
            if not param_combos:
                raise ValueError('No admissible parameter combinations to test')

            if len(param_combos) > 300:
                warnings.warn(
                    f'Searching for best of {len(param_combos)} configurations.',
                    stacklevel=2,
                )

            heatmap = pd.Series(
                np.nan,
                name=maximize_key,
                index=pd.MultiIndex.from_tuples(
                    [p.values() for p in param_combos],
                    names=next(iter(param_combos)).keys(),
                ),
            )

            def _batch(seq: List[Dict[str, str]]):
                n = np.clip(len(seq) // (os.cpu_count() or 1), 5, 300)
                for i in range(0, len(seq), n):
                    yield seq[i : i + n]

            backtest_uuid = np.random.random()
            param_batches = list(_batch(param_combos))
            Backtest._mp_backtests[backtest_uuid] = (self, param_batches, maximize)  # type: ignore
            try:
                if mp.get_start_method(allow_none=False) == 'fork':
                    with ProcessPoolExecutor() as executor:
                        futures = [executor.submit(Backtest._mp_task, backtest_uuid, i) for i in range(len(param_batches))]
                        for future in _tqdm(as_completed(futures), total=len(futures), desc='Backtest.grid'):
                            batch_index, values = future.result()
                            for value, params in zip(values, param_batches[batch_index]):
                                heatmap[tuple(params.values())] = value
                else:
                    if os.name == 'posix':
                        warnings.warn('For multiprocessing support in `Backtest.optimize()` ' "set multiprocessing start method to 'fork'.")
                    for batch_index in _tqdm(range(len(param_batches))):
                        _, values = Backtest._mp_task(backtest_uuid, batch_index)
                        for value, params in zip(values, param_batches[batch_index]):
                            heatmap[tuple(params.values())] = value
            finally:
                del Backtest._mp_backtests[backtest_uuid]

            best_params = heatmap.idxmax()

            if pd.isnull(best_params):
                stats = self.run(**param_combos[0])
            else:
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        def _optimize_skopt() -> Union[pd.Series, Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, dict]]:
            try:
                from skopt import forest_minimize

                from skopt.utils import use_named_args
                from skopt.callbacks import DeltaXStopper
                from skopt.learning import ExtraTreesRegressor
            except ImportError:
                raise ImportError("Need package 'scikit-optimize' for method='skopt'. " 'pip install scikit-optimize')

            nonlocal max_tries
            max_tries = get_max_tries(max_tries, _grid_size)
            dimensions = construct_dimensions(kwargs)
            memoized_run = lru_cache()(lambda tup: self.run(**dict(tup)))

            INVALID = 1e300
            progress = _tqdm(dimensions, total=max_tries, desc='Backtest.optimize', leave=False)

            @use_named_args(dimensions=dimensions)
            def objective_function(**params):
                progress.update(1)

                # Check constraints
                # TODO: Adjust after https://github.com/scikit-optimize/scikit-optimize/pull/971
                if not constraint(AttrDict(params)):
                    return INVALID
                res = memoized_run(tuple(params.items()))
                value = -maximize(res)
                if np.isnan(value):
                    return INVALID
                return value

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The objective has been evaluated at this point before.')

                res = forest_minimize(
                    func=objective_function,
                    dimensions=dimensions,
                    n_calls=max_tries,
                    base_estimator=ExtraTreesRegressor(n_estimators=20, min_samples_leaf=2),
                    acq_func='LCB',
                    kappa=3,
                    n_initial_points=min(max_tries, 20 + 3 * len(kwargs)),
                    initial_point_generator='lhs',  # 'sobel' requires n_initial_points ~ 2**N
                    callback=DeltaXStopper(9e-7),
                    random_state=random_state,
                )

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(
                    dict(zip(map(tuple, res.x_iters), -res.func_vals)),
                    name=maximize_key,
                )
                heatmap.index.names = kwargs.keys()
                heatmap = heatmap[heatmap != -INVALID]
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                valid = res.func_vals != INVALID
                res.x_iters = list(compress(res.x_iters, valid))
                res.func_vals = res.func_vals[valid]
                output.append(res)
            progress.clear()
            progress.close()
            return stats if len(output) == 1 else tuple(output)

        if method == 'grid':
            output = _optimize_grid()
        elif method == 'skopt':
            output = _optimize_skopt()
        else:
            raise ValueError(f"Method should be 'grid' or 'skopt', not {method!r}")
        return output

    @staticmethod
    def _mp_task(backtest_uuid, batch_index):
        bt, param_batches, maximize_func = Backtest._mp_backtests[backtest_uuid]
        return batch_index, [
            maximize_func(stats) if stats['# Trades'] else np.nan for stats in (bt.run(**params) for params in param_batches[batch_index])
        ]

    _mp_backtests: Dict[float, Tuple['Backtest', List, Callable]] = {}

    def plot(
        self,
        *,
        results: pd.Series = None,
        filename=None,
        plot_width=None,
        plot_equity=True,
        plot_return=False,
        plot_pl=True,
        plot_volume=True,
        plot_drawdown=False,
        smooth_equity=False,
        relative_equity=True,
        superimpose: Union[bool, str] = True,
        resample=True,
        reverse_indicators=False,
        show_legend=True,
        open_browser=True,
    ):
        if results is None:
            if self._results is None:
                raise RuntimeError('First issue `backtest.run()` to obtain results.')
            results = self._results

        plot(
            results=results,
            df=self._data,
            indicators=results._strategy._indicators,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
        )


