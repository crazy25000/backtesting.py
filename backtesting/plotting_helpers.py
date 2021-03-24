from typing import Union, Callable

import pandas as pd

from backtesting._plotting import plot_heatmaps as _plot_heatmaps


def plot_heatmaps(
    heatmap: pd.Series,
    agg: Union[str, Callable] = 'max',
    *,
    ncols: int = 3,
    plot_width: int = 1200,
    filename: str = '',
    open_browser: bool = True,
):
    """
    Plots a grid of heatmaps, one for every pair of parameters in `heatmap`.

    `heatmap` is a Series as returned by
    `backtesting.backtesting.Backtest.optimize` when its parameter
    `return_heatmap=True`.

    When projecting the n-dimensional heatmap onto 2D, the values are
    aggregated by 'max' function by default. This can be tweaked
    with `agg` parameter, which accepts any argument pandas knows
    how to aggregate by.

    .. todo::
        Lay heatmaps out lower-triangular instead of in a simple grid.
        Like [`skopt.plots.plot_objective()`][plot_objective] does.

    [plot_objective]: \
        https://scikit-optimize.github.io/stable/modules/plots.html#plot-objective
    """
    return _plot_heatmaps(heatmap, agg, ncols, filename, plot_width, open_browser)
