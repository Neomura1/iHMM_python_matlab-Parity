"""Plot observed 1D sequence with state coloring."""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_state_sequence(
    Y: np.ndarray,
    z: np.ndarray,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
    cmap: str = "tab20",
    s: float = 12.0,
):
    """Plot y over time with a light gray line and colored state scatter.

    Parameters
    ----------
    Y : ndarray, shape (T, D)
        Observation sequence; only the first dimension is plotted.
    z : ndarray, shape (T,)
        Integer state labels 0..K-1.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None a new figure/axes is created.
    title : str, optional
        Title for the axes.
    cmap : str, default 'tab20'
        Colormap used to map states to colors.
    s : float, default 12.0
        Marker size.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the plot is drawn on.
    sc : PathCollection
        The scatter artist (useful for colorbars or further styling).
    """

    import matplotlib.pyplot as plt

    Y = np.asarray(Y)
    z = np.asarray(z, dtype=int)
    T = Y.shape[0]
    x = np.arange(T)
    y = Y[:, 0] if Y.ndim == 2 else np.asarray(Y).ravel()

    if ax is None:
        fig, ax = plt.subplots()

    # Background light-gray line without markers
    ax.plot(x, y, color="0.8", linewidth=1.0, zorder=1)
    # Colored scatter by state labels
    sc = ax.scatter(x, y, c=z, cmap=cmap, s=s, edgecolors="none", zorder=2)
    ax.set_xlabel("time t")
    ax.set_ylabel("y[:,0]")
    if title:
        ax.set_title(title)
    # Build legend with unique states present
    states = np.unique(z)
    handles = []
    labels = []
    import matplotlib as mpl
    cmap_obj = mpl.cm.get_cmap(cmap)
    for st in states:
        handles.append(
            mpl.lines.Line2D([0], [0], marker='o', color='w', label=f'state {st}',
                              markerfacecolor=cmap_obj(st % cmap_obj.N), markersize=6)
        )
        labels.append(f"state {st}")
    if len(handles) <= 20:  # avoid massive legends
        ax.legend(handles, labels, loc='best', fontsize='small', frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    return ax, sc
