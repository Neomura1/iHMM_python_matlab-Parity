"""Simple trace plot utility for scalar sequences (e.g., log-likelihood)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def traceplot_scalar(
    trace: Sequence[float],
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
    ylab: Optional[str] = None,
    xlab: str = "iteration",
    color: Optional[str] = None,
):
    """Plot a scalar trace over iterations.

    Parameters
    ----------
    trace : sequence of float
        Values per saved iteration (e.g., samples['loglik']).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None a new figure/axes is created.
    title : str, optional
        Title for the axes.
    ylab : str, optional
        Y axis label.
    xlab : str, default 'iteration'
        X axis label.
    color : str, optional
        Line color.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the trace is plotted on.
    """

    import matplotlib.pyplot as plt  # lazy import to avoid hard dependency upstream

    y = np.asarray(trace, dtype=float)
    x = np.arange(1, len(y) + 1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, color=color or "tab:blue", linewidth=1.5)
    ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    return ax

