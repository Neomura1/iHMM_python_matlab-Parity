"""Heatmap plotting for transition matrices Pi."""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_transition_heatmap(
    Pi: np.ndarray,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """Plot a heatmap of a transition matrix.

    Parameters
    ----------
    Pi : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None a new figure/axes is created.
    title : str, optional
        Title for the axes.
    cmap : str, default 'viridis'
        Matplotlib colormap.
    vmin, vmax : float
        Color scale limits.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the heatmap is plotted on.
    """

    import matplotlib.pyplot as plt

    Pi = np.asarray(Pi)
    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(Pi, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("to state k")
    ax.set_ylabel("from state j")
    ax.set_xticks(range(Pi.shape[1]))
    ax.set_yticks(range(Pi.shape[0]))
    if title:
        ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(j→k)")
    return ax

