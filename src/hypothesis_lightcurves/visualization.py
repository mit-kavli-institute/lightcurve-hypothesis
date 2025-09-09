"""Visualization utilities for lightcurves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes  # type: ignore[import-not-found]
    from matplotlib.figure import Figure  # type: ignore[import-not-found]

    from hypothesis_lightcurves.models import Lightcurve


def plot_lightcurve(
    lc: Lightcurve,
    ax: Axes | None = None,
    show_errors: bool = True,
    color: str = "C0",
    alpha: float = 1.0,
    marker: str = "o",
    markersize: float = 4,
    linewidth: float = 0,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "Flux",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a lightcurve with optional error bars.

    Parameters
    ----------
    lc : Lightcurve
        The lightcurve to plot.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, creates new figure and axes.
    show_errors : bool, default=True
        Whether to show error bars if available.
    color : str, default="C0"
        Color for the plot points and error bars.
    alpha : float, default=1.0
        Transparency level (0=transparent, 1=opaque).
    marker : str, default="o"
        Marker style for data points.
    markersize : float, default=4
        Size of the markers.
    linewidth : float, default=0
        Width of connecting lines (0 for no lines).
    label : str or None, optional
        Label for the legend.
    title : str or None, optional
        Title for the plot.
    xlabel : str, default="Time"
        Label for x-axis.
    ylabel : str, default="Flux"
        Label for y-axis.
    **kwargs : Any
        Additional keyword arguments passed to errorbar/plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import periodic_lightcurves
    >>> lc = periodic_lightcurves().example()
    >>> fig, ax = plot_lightcurve(lc, title="Periodic Lightcurve")
    """
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Plot with or without error bars
    if show_errors and lc.flux_err is not None:
        ax.errorbar(
            lc.time,
            lc.flux,
            yerr=lc.flux_err,
            fmt=marker,
            color=color,
            alpha=alpha,
            markersize=markersize,
            linewidth=linewidth,
            label=label,
            capsize=2,
            **kwargs,
        )
    else:
        ax.plot(
            lc.time,
            lc.flux,
            marker=marker,
            color=color,
            alpha=alpha,
            markersize=markersize,
            linewidth=linewidth,
            linestyle="-" if linewidth > 0 else "none",
            label=label,
            **kwargs,
        )

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add legend if label was provided
    if label:
        ax.legend()

    return fig, ax


def plot_multiple_lightcurves(
    lightcurves: list[Lightcurve],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    title: str | None = None,
    ncols: int = 1,
    sharex: bool = True,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """Plot multiple lightcurves in subplots.

    Parameters
    ----------
    lightcurves : list of Lightcurve
        List of lightcurves to plot.
    labels : list of str or None, optional
        Labels for each lightcurve.
    colors : list of str or None, optional
        Colors for each lightcurve.
    title : str or None, optional
        Overall figure title.
    ncols : int, default=1
        Number of columns in subplot grid.
    sharex : bool, default=True
        Whether to share x-axis across subplots.
    sharey : bool, default=False
        Whether to share y-axis across subplots.
    figsize : tuple of float or None, optional
        Figure size (width, height). If None, computed automatically.
    **kwargs : Any
        Additional keyword arguments passed to plot_lightcurve.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : numpy.ndarray of matplotlib.axes.Axes
        Array of axes objects.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import lightcurves
    >>> lcs = [lightcurves().example() for _ in range(4)]
    >>> fig, axes = plot_multiple_lightcurves(lcs, ncols=2)
    """
    import matplotlib.pyplot as plt

    n_lcs = len(lightcurves)
    nrows = (n_lcs + ncols - 1) // ncols

    if figsize is None:
        figsize = (10 * ncols, 6 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False
    )
    axes = axes.flatten()

    if labels is None:
        labels = [f"LC {i+1}" for i in range(n_lcs)]

    if colors is None:
        colors = [f"C{i % 10}" for i in range(n_lcs)]

    for i, (lc, label, color) in enumerate(zip(lightcurves, labels, colors, strict=False)):
        ax = axes[i]
        plot_lightcurve(lc, ax=ax, color=color, label=label, **kwargs)
        ax.set_title(label)

    # Hide unused subplots
    for i in range(n_lcs, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    return fig, axes[:n_lcs]


def plot_lightcurve_comparison(
    lc1: Lightcurve,
    lc2: Lightcurve,
    label1: str = "Original",
    label2: str = "Modified",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Plot two lightcurves for comparison with residuals.

    Creates a three-panel plot showing:
    1. Both lightcurves overlaid
    2. Individual lightcurves
    3. Residuals (if time arrays match)

    Parameters
    ----------
    lc1 : Lightcurve
        First lightcurve (reference).
    lc2 : Lightcurve
        Second lightcurve to compare.
    label1 : str, default="Original"
        Label for first lightcurve.
    label2 : str, default="Modified"
        Label for second lightcurve.
    title : str or None, optional
        Overall figure title.
    figsize : tuple of float, default=(12, 8)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : tuple of matplotlib.axes.Axes
        Tuple of (top_ax, middle_ax, bottom_ax).

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import periodic_lightcurves
    >>> from hypothesis_lightcurves.modifiers import add_noise
    >>> lc1 = periodic_lightcurves().example()
    >>> lc2 = add_noise(lc1, noise_level=5.0)
    >>> fig, axes = plot_lightcurve_comparison(lc1, lc2, label2="With Noise")
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)

    # Create gridspec for custom layout
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Top panel: overlay
    plot_lightcurve(lc1, ax=ax1, color="C0", alpha=0.7, label=label1)
    plot_lightcurve(lc2, ax=ax1, color="C1", alpha=0.7, label=label2)
    ax1.set_title("Overlay Comparison")
    ax1.legend()
    ax1.set_xlabel("")

    # Middle panel: side by side with offset
    offset = np.median(lc1.flux) * 1.5
    plot_lightcurve(lc1, ax=ax2, color="C0", label=label1)
    lc2_offset = lc2.copy()
    lc2_offset.flux = lc2_offset.flux + offset
    plot_lightcurve(lc2_offset, ax=ax2, color="C1", label=f"{label2} (offset)")
    ax2.set_title("Individual Lightcurves")
    ax2.legend()
    ax2.set_xlabel("")

    # Bottom panel: residuals if possible
    if np.array_equal(lc1.time, lc2.time):
        residuals = lc2.flux - lc1.flux
        ax3.plot(lc1.time, residuals, "k-", alpha=0.7, linewidth=1)
        ax3.scatter(lc1.time, residuals, c="k", s=10, alpha=0.5)
        ax3.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax3.set_ylabel("Residuals")
        ax3.set_title(f"Residuals ({label2} - {label1})")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "Cannot compute residuals:\nTime arrays do not match",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_ylabel("Residuals")

    ax3.set_xlabel("Time")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    return fig, (ax1, ax2, ax3)


def create_gallery_figure(
    generators: dict[str, Lightcurve],
    title: str = "Lightcurve Generator Gallery",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create a gallery figure showing examples from different generators.

    Parameters
    ----------
    generators : dict of str to Lightcurve
        Dictionary mapping generator names to example lightcurves.
    title : str, default="Lightcurve Generator Gallery"
        Title for the gallery figure.
    figsize : tuple of float or None, optional
        Figure size. If None, computed based on number of generators.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The gallery figure.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import (
    ...     periodic_lightcurves,
    ...     transient_lightcurves,
    ...     lightcurves
    ... )
    >>> examples = {
    ...     "Basic": lightcurves().example(),
    ...     "Periodic": periodic_lightcurves().example(),
    ...     "Transient": transient_lightcurves().example(),
    ... }
    >>> fig = create_gallery_figure(examples)
    """
    import matplotlib.pyplot as plt

    n_generators = len(generators)
    ncols = min(3, n_generators)
    nrows = (n_generators + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (name, lc) in enumerate(generators.items()):
        ax = axes[i]
        plot_lightcurve(lc, ax=ax, markersize=2, alpha=0.7)
        ax.set_title(name, fontsize=12, fontweight="bold")

        # Add metadata info if available
        if lc.metadata:
            info_text = "\n".join(
                [
                    f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in list(lc.metadata.items())[:3]
                ]
            )
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.7},
            )

    # Hide unused subplots
    for i in range(n_generators, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    return fig
