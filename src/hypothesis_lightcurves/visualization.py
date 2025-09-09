"""Visualization utilities for lightcurves.

This module provides functions for visualizing lightcurves, including
single plots, comparisons, and gallery generation for documentation.
"""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hypothesis_lightcurves.models import Lightcurve

# Registry for custom visualizers
VISUALIZERS: dict[str, Callable] = {}


def register_visualizer(generator_name: str) -> Callable:
    """Register a custom visualizer for a specific generator type.

    This decorator allows new generator types to register specialized
    visualization functions that will be automatically used when plotting
    lightcurves from those generators.

    Parameters
    ----------
    generator_name : str
        Name of the generator type (e.g., 'periodic', 'transient').

    Returns
    -------
    Callable
        Decorator function for registering the visualizer.

    Examples
    --------
    >>> @register_visualizer('exotic')
    ... def plot_exotic(lc, ax=None, **kwargs):
    ...     # Custom visualization logic
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        VISUALIZERS[generator_name] = func
        return func

    return decorator


def plot_lightcurve(
    lightcurve: Lightcurve,
    ax: Axes | None = None,
    show_errors: bool = True,
    color: str = "C0",
    alpha: float = 1.0,
    label: str | None = None,
    marker: str = ".",
    markersize: float = 4,
    linestyle: str = "-",
    linewidth: float = 1,
    title: str | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a single lightcurve.

    Parameters
    ----------
    lightcurve : Lightcurve
        The lightcurve to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    show_errors : bool, default=True
        Whether to show error bars if available.
    color : str, default='C0'
        Color for the plot.
    alpha : float, default=1.0
        Transparency level.
    label : str, optional
        Label for the plot legend.
    marker : str, default='.'
        Marker style for data points.
    markersize : float, default=4
        Size of markers.
    linestyle : str, default='-'
        Line style for connecting points.
    linewidth : float, default=1
        Width of connecting lines.
    title : str, optional
        Title for the plot.
    **kwargs
        Additional keyword arguments passed to plot functions.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted lightcurve.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import lightcurves
    >>> lc = lightcurves().example()
    >>> ax = plot_lightcurve(lc, title="Random Lightcurve")
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the main lightcurve
    if linestyle != "none":
        ax.plot(
            lightcurve.time,
            lightcurve.flux,
            color=color,
            alpha=alpha,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            **kwargs,
        )

    if marker != "none":
        ax.scatter(
            lightcurve.time,
            lightcurve.flux,
            color=color,
            alpha=alpha,
            marker=marker,
            s=markersize**2,
            **kwargs,
        )

    # Add error bars if available
    if show_errors and lightcurve.flux_err is not None:
        ax.errorbar(
            lightcurve.time,
            lightcurve.flux,
            yerr=lightcurve.flux_err,
            fmt="none",
            color=color,
            alpha=alpha * 0.5,
            capsize=2,
            **kwargs,
        )

    # Set labels and title
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Flux", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add legend if label was provided
    if label:
        ax.legend()

    return ax


def plot_comparison(
    lightcurves: list[Lightcurve],
    labels: list[str] | None = None,
    title: str = "Lightcurve Comparison",
    figsize: tuple[float, float] = (12, 8),
    ncols: int = 2,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs: Any,
) -> Figure:
    """Plot multiple lightcurves for comparison.

    Parameters
    ----------
    lightcurves : list of Lightcurve
        List of lightcurves to compare.
    labels : list of str, optional
        Labels for each lightcurve.
    title : str, default="Lightcurve Comparison"
        Overall title for the figure.
    figsize : tuple of float, default=(12, 8)
        Figure size in inches.
    ncols : int, default=2
        Number of columns in subplot grid.
    sharex : bool, default=False
        Whether to share x-axis across subplots.
    sharey : bool, default=False
        Whether to share y-axis across subplots.
    **kwargs
        Additional keyword arguments passed to plot_lightcurve.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all comparison plots.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import (
    ...     lightcurves, periodic_lightcurves, transient_lightcurves
    ... )
    >>> lcs = [
    ...     lightcurves().example(),
    ...     periodic_lightcurves().example(),
    ...     transient_lightcurves().example()
    ... ]
    >>> fig = plot_comparison(lcs, labels=['Random', 'Periodic', 'Transient'])
    >>> plt.show()
    """
    n_lightcurves = len(lightcurves)
    nrows = (n_lightcurves + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False
    )

    fig.suptitle(title, fontsize=16, y=1.02)

    if labels is None:
        labels = [f"Lightcurve {i+1}" for i in range(n_lightcurves)]

    for i, (lc, label) in enumerate(zip(lightcurves, labels, strict=False)):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        # Check if there's a custom visualizer for this type
        if lc.metadata and "generator_type" in lc.metadata:
            generator_type = lc.metadata["generator_type"]
            if generator_type in VISUALIZERS:
                VISUALIZERS[generator_type](lc, ax=ax, **kwargs)
            else:
                plot_lightcurve(lc, ax=ax, title=label, **kwargs)
        else:
            plot_lightcurve(lc, ax=ax, title=label, **kwargs)

    # Hide empty subplots
    for i in range(n_lightcurves, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def create_gallery_plot(
    n_examples: int = 6,
    generator_func: Callable | None = None,
    title: str = "Lightcurve Gallery",
    figsize: tuple[float, float] = (15, 10),
    seed: int | None = None,
    **generator_kwargs: Any,
) -> Figure:
    """Create a gallery plot showing multiple examples from a generator.

    Parameters
    ----------
    n_examples : int, default=6
        Number of examples to generate and plot.
    generator_func : Callable, optional
        Generator function to use. If None, uses basic lightcurves.
    title : str, default="Lightcurve Gallery"
        Title for the gallery.
    figsize : tuple of float, default=(15, 10)
        Figure size in inches.
    seed : int, optional
        Random seed for reproducibility.
    **generator_kwargs
        Keyword arguments passed to the generator function.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the gallery.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import periodic_lightcurves
    >>> fig = create_gallery_plot(
    ...     n_examples=9,
    ...     generator_func=periodic_lightcurves,
    ...     title="Periodic Lightcurve Gallery",
    ...     min_period=1.0,
    ...     max_period=5.0
    ... )
    >>> plt.show()
    """
    if seed is not None:
        np.random.seed(seed)

    if generator_func is None:
        from hypothesis_lightcurves.generators import lightcurves

        generator_func = lightcurves

    # Generate examples
    strategy = generator_func(**generator_kwargs)
    examples = [strategy.example() for _ in range(n_examples)]

    # Create comparison plot
    fig = plot_comparison(
        examples,
        labels=[f"Example {i+1}" for i in range(n_examples)],
        title=title,
        figsize=figsize,
        ncols=3,
    )

    return fig


def plot_with_annotations(
    lightcurve: Lightcurve,
    ax: Axes | None = None,
    annotate_metadata: bool = True,
    annotate_statistics: bool = True,
    **kwargs: Any,
) -> Axes:
    """Plot a lightcurve with educational annotations.

    Parameters
    ----------
    lightcurve : Lightcurve
        The lightcurve to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    annotate_metadata : bool, default=True
        Whether to annotate metadata information.
    annotate_statistics : bool, default=True
        Whether to annotate statistical properties.
    **kwargs
        Additional keyword arguments passed to plot_lightcurve.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the annotated plot.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import periodic_lightcurves
    >>> lc = periodic_lightcurves().example()
    >>> ax = plot_with_annotations(lc)
    >>> plt.show()
    """
    ax = plot_lightcurve(lightcurve, ax=ax, **kwargs)

    # Prepare annotation text
    annotations = []

    if annotate_statistics:
        annotations.extend(
            [
                f"N points: {lightcurve.n_points}",
                f"Duration: {lightcurve.duration:.2f}",
                f"Mean flux: {lightcurve.mean_flux:.2f}",
                f"Std flux: {lightcurve.std_flux:.2f}",
            ]
        )

    if annotate_metadata and lightcurve.metadata:
        for key, value in lightcurve.metadata.items():
            if isinstance(value, float):
                annotations.append(f"{key}: {value:.3f}")
            else:
                annotations.append(f"{key}: {value}")

    # Add text box with annotations
    if annotations:
        text = "\n".join(annotations)
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    return ax


def plot_phase_folded(
    lightcurve: Lightcurve,
    period: float,
    ax: Axes | None = None,
    n_phase_bins: int = 50,
    show_binned: bool = True,
    **kwargs: Any,
) -> Axes:
    """Plot a phase-folded lightcurve.

    Parameters
    ----------
    lightcurve : Lightcurve
        The lightcurve to phase-fold.
    period : float
        Period to fold on.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    n_phase_bins : int, default=50
        Number of phase bins for binned curve.
    show_binned : bool, default=True
        Whether to show binned phase curve.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the phase-folded plot.

    Examples
    --------
    >>> from hypothesis_lightcurves.generators import periodic_lightcurves
    >>> lc = periodic_lightcurves(min_period=2.0, max_period=2.0).example()
    >>> ax = plot_phase_folded(lc, period=2.0)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate phase
    phase = (lightcurve.time % period) / period

    # Plot folded lightcurve
    ax.scatter(phase, lightcurve.flux, alpha=0.5, s=4, label="Data")

    # Add a second cycle for clarity
    ax.scatter(phase + 1, lightcurve.flux, alpha=0.5, s=4, color="C0")

    if show_binned:
        # Bin the phase-folded data
        phase_bins = np.linspace(0, 1, n_phase_bins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        binned_flux = []

        for i in range(len(bin_centers)):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.any(mask):
                binned_flux.append(np.mean(lightcurve.flux[mask]))
            else:
                binned_flux.append(np.nan)

        # Plot binned curve
        ax.plot(bin_centers, binned_flux, "r-", linewidth=2, label="Binned")
        ax.plot(bin_centers + 1, binned_flux, "r-", linewidth=2)

    ax.set_xlabel("Phase", fontsize=12)
    ax.set_ylabel("Flux", fontsize=12)
    ax.set_title(f"Phase-folded at P = {period:.3f}", fontsize=14)
    ax.set_xlim(-0.1, 2.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


# Register default visualizers for built-in generator types
@register_visualizer("periodic")
def plot_periodic_specialized(lc: Lightcurve, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Specialized plotter for periodic lightcurves."""
    ax = plot_lightcurve(lc, ax=ax, **kwargs)

    if lc.metadata and "period" in lc.metadata:
        period = lc.metadata["period"]
        ax.axvline(period, color="red", linestyle="--", alpha=0.5, label=f"P={period:.2f}")
        ax.legend()

    return ax


@register_visualizer("transient")
def plot_transient_specialized(lc: Lightcurve, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Specialized plotter for transient lightcurves."""
    ax = plot_lightcurve(lc, ax=ax, **kwargs)

    if lc.metadata:
        if "peak_time" in lc.metadata:
            peak_time = lc.metadata["peak_time"]
            ax.axvline(peak_time, color="red", linestyle="--", alpha=0.5, label="Peak")

        # Mark rise and decay regions
        if "rise_time" in lc.metadata and "peak_time" in lc.metadata:
            rise_start = max(0, peak_time - 3 * lc.metadata["rise_time"])
            ax.axvspan(rise_start, peak_time, alpha=0.1, color="blue", label="Rise")

        if "decay_time" in lc.metadata and "peak_time" in lc.metadata:
            decay_end = peak_time + 3 * lc.metadata["decay_time"]
            ax.axvspan(peak_time, decay_end, alpha=0.1, color="orange", label="Decay")

        ax.legend()

    return ax
