"""Hypothesis strategies for generating lightcurves."""

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from hypothesis_lightcurves.models import Lightcurve
from hypothesis_lightcurves.modifiers import (
    add_noise,
    add_outliers,
    add_periodic_signal,
    add_transient_event,
)

# Type variable for strategy resolution
T = TypeVar("T")


def resolve_strategy(draw: st.DrawFn, value_or_strategy: T | st.SearchStrategy[T]) -> T:
    """Resolve a value that might be a strategy or a static value.

    Parameters
    ----------
    draw : hypothesis.strategies.DrawFn
        Hypothesis draw function for resolving strategies.
    value_or_strategy : T or hypothesis.strategies.SearchStrategy[T]
        Either a static value or a hypothesis strategy that generates values of type T.

    Returns
    -------
    T
        The resolved value, either the static value itself or a value drawn from the strategy.

    Notes
    -----
    This helper function allows generator parameters to accept both static values
    and Hypothesis strategies, providing flexibility in test generation.
    """
    if isinstance(value_or_strategy, st.SearchStrategy):
        return draw(value_or_strategy)  # type: ignore[no-any-return]
    return value_or_strategy


@st.composite
def baseline_lightcurves(
    draw: st.DrawFn,
    n_points: int | st.SearchStrategy[int] | None = None,
    min_points: int = 10,
    max_points: int = 1000,
    min_time: float | st.SearchStrategy[float] = 0.0,
    max_time: float | st.SearchStrategy[float] = 100.0,
    duration: float | st.SearchStrategy[float] | None = None,
    start_time: float | st.SearchStrategy[float] | None = None,
    baseline_type: str | st.SearchStrategy[str] = "flat",
    baseline_flux: float | st.SearchStrategy[float] = 100.0,
    time_sampling: str | st.SearchStrategy[str] = "uniform",
) -> Lightcurve:
    """Generate baseline lightcurves with simple patterns.

    This generator supports both static values and hypothesis strategies for all parameters,
    providing maximum flexibility for test generation.

    Parameters
    ----------
    draw : hypothesis.strategies.DrawFn
        Hypothesis draw function for generating random values.
    n_points : int or SearchStrategy[int] or None, optional
        Number of data points. If None, drawn from [min_points, max_points].
    min_points : int, default=10
        Minimum number of data points when n_points is None.
    max_points : int, default=1000
        Maximum number of data points when n_points is None.
    min_time : float or SearchStrategy[float], default=0.0
        Minimum time value for the lightcurve.
    max_time : float or SearchStrategy[float], default=100.0
        Maximum time value for the lightcurve.
    duration : float or SearchStrategy[float] or None, optional
        Duration of lightcurve. If None, computed from time range.
    start_time : float or SearchStrategy[float] or None, optional
        Start time of lightcurve. If None, uses min_time.
    baseline_type : str or SearchStrategy[str], default="flat"
        Type of baseline: 'flat', 'random_walk', or 'smooth'.
    baseline_flux : float or SearchStrategy[float], default=100.0
        Base flux level for the baseline.
    time_sampling : str or SearchStrategy[str], default="uniform"
        Time sampling pattern: 'uniform', 'random', or 'irregular'.

    Returns
    -------
    Lightcurve
        A baseline lightcurve with the specified characteristics.

    Examples
    --------
    Using static values:

    >>> from hypothesis import given
    >>> @given(lc=baseline_lightcurves(n_points=100, baseline_type="flat"))
    >>> def test_flat_baseline(lc):
    ...     assert lc.n_points == 100
    ...     assert lc.std_flux < lc.mean_flux * 0.1  # Should be relatively flat

    Using strategies for flexible testing:

    >>> from hypothesis import strategies as st
    >>> @given(lc=baseline_lightcurves(
    ...     n_points=st.integers(50, 200),
    ...     baseline_type=st.sampled_from(["flat", "smooth"]),
    ...     baseline_flux=st.floats(80, 120)
    ... ))
    >>> def test_variable_baseline(lc):
    ...     assert 50 <= lc.n_points <= 200
    ...     assert lc.metadata['baseline_type'] in ["flat", "smooth"]

    Notes
    -----
    - All parameters can accept either static values or Hypothesis strategies
    - The 'irregular' time sampling creates clustered observations
    - Metadata is automatically populated with generation parameters
    """
    # Resolve n_points
    if n_points is None:
        n_points_value = draw(st.integers(min_value=min_points, max_value=max_points))
    else:
        n_points_value = resolve_strategy(draw, n_points)

    # Resolve other parameters
    min_time_value = resolve_strategy(draw, min_time)
    max_time_value = resolve_strategy(draw, max_time)
    baseline_type_value = resolve_strategy(draw, baseline_type)
    baseline_flux_value = resolve_strategy(draw, baseline_flux)
    time_sampling_value = resolve_strategy(draw, time_sampling)

    # Generate time array based on sampling pattern
    if time_sampling_value == "uniform":
        # Resolve duration and start_time
        if duration is None:
            duration_value = draw(
                st.floats(
                    min_value=(max_time_value - min_time_value) / 2,
                    max_value=max_time_value - min_time_value,
                )
            )
        else:
            duration_value = resolve_strategy(draw, duration)

        if start_time is None:
            start_time_value = draw(
                st.floats(
                    min_value=min_time_value,
                    max_value=min(min_time_value + 10, max_time_value - duration_value),
                )
            )
        else:
            start_time_value = resolve_strategy(draw, start_time)

        time = np.linspace(start_time_value, start_time_value + duration_value, n_points_value)

    elif time_sampling_value == "random":
        time = draw(
            npst.arrays(
                dtype=np.float64,
                shape=n_points_value,
                elements=st.floats(min_value=min_time_value, max_value=max_time_value),
                unique=True,
            )
        )
        time = np.sort(time)

    else:  # irregular
        # Create irregular sampling with clusters
        n_clusters = draw(st.integers(min_value=3, max_value=10))
        points_per_cluster = n_points_value // n_clusters
        remaining_points = n_points_value % n_clusters
        time_list: list[float] = []
        for i in range(n_clusters):
            # Add extra points to the first few clusters to ensure we get exactly n_points_value
            extra_points = 1 if i < remaining_points else 0
            cluster_size = points_per_cluster + extra_points
            cluster_center = draw(st.floats(min_value=min_time_value, max_value=max_time_value))
            cluster_width = draw(st.floats(min_value=0.5, max_value=5.0))
            cluster_times = np.random.normal(cluster_center, cluster_width, cluster_size)
            time_list.extend(cluster_times)
        time = np.sort(np.array(time_list))

    # Generate baseline flux based on type
    if baseline_type_value == "flat":
        # Allow some variation around baseline_flux
        flux_variation = draw(st.floats(min_value=0.8, max_value=1.2))
        flux = np.full(n_points_value, baseline_flux_value * flux_variation)

    elif baseline_type_value == "random_walk":
        # Generate smooth random walk
        step_size = draw(st.floats(min_value=0.001, max_value=0.01))
        steps = np.random.normal(0, step_size * baseline_flux_value, n_points_value)
        flux = baseline_flux_value + np.cumsum(steps)

    else:  # smooth
        # Generate smooth baseline using low-frequency sinusoids
        n_components = draw(st.integers(min_value=1, max_value=3))
        flux = np.full(n_points_value, baseline_flux_value)
        time_range = max_time_value - min_time_value
        for _ in range(n_components):
            period = draw(st.floats(min_value=time_range / 2, max_value=time_range * 2))
            amplitude = draw(st.floats(min_value=0.01, max_value=0.05)) * baseline_flux_value
            phase = draw(st.floats(min_value=0, max_value=2 * np.pi))
            flux += amplitude * np.sin(2 * np.pi * time / period + phase)

    return Lightcurve(
        time=time,
        flux=flux,
        flux_err=None,
        metadata={
            "baseline_type": baseline_type_value,
            "time_sampling": time_sampling_value,
            "baseline_flux": baseline_flux_value,
        },
        modifications=["baseline"],
    )


@st.composite
def modified_lightcurves(
    draw: st.DrawFn,
    base_strategy: st.SearchStrategy[Lightcurve] | None = None,
    modifiers: list[Callable[[Lightcurve], Lightcurve]] | None = None,
    modifier_params: dict | None = None,
) -> Lightcurve:
    """Generate lightcurves by applying modifiers to a baseline.

    Args:
        draw: Hypothesis draw function
        base_strategy: Strategy for generating the baseline lightcurve
        modifiers: List of modifier functions to potentially apply
        modifier_params: Optional parameters for modifiers

    Returns:
        A modified Lightcurve object
    """
    # Use default baseline if not provided
    if base_strategy is None:
        base_strategy = baseline_lightcurves()

    # Start with baseline
    lc = draw(base_strategy)

    # Default modifiers if not provided
    if modifiers is None:
        modifiers = [add_noise, add_periodic_signal, add_outliers]  # type: ignore[list-item]

    # Apply random subset of modifiers
    n_modifiers = draw(st.integers(min_value=0, max_value=len(modifiers)))
    selected_modifiers = draw(
        st.lists(
            st.sampled_from(modifiers),
            min_size=n_modifiers,
            max_size=n_modifiers,
            unique=True,
        )
    )

    # Apply each modifier with random parameters
    for modifier in selected_modifiers:
        modifier_name = modifier.__name__
        if modifier_name == "add_noise":
            lc = modifier(  # type: ignore[call-arg]
                lc,
                noise_type=draw(st.sampled_from(["gaussian", "poisson", "uniform"])),
                level=draw(st.floats(min_value=0.001, max_value=0.1)) * lc.std_flux,
            )
        elif modifier_name == "add_periodic_signal":
            duration = lc.duration
            lc = modifier(  # type: ignore[call-arg]
                lc,
                period=draw(st.floats(min_value=duration / 20, max_value=duration / 2)),
                amplitude=draw(st.floats(min_value=0.01, max_value=0.3)) * lc.std_flux,
                phase=draw(st.floats(min_value=0, max_value=2 * np.pi)),
            )
        elif modifier_name == "add_transient_event":
            lc = modifier(  # type: ignore[call-arg]
                lc,
                peak_time=draw(st.floats(min_value=lc.time.min(), max_value=lc.time.max())),
                peak_flux=draw(st.floats(min_value=0.5, max_value=3.0)) * lc.std_flux,
                rise_time=draw(st.floats(min_value=0.01, max_value=0.1)) * lc.duration,
                decay_time=draw(st.floats(min_value=0.05, max_value=0.3)) * lc.duration,
                event_type=draw(st.sampled_from(["burst", "dip"])),
            )
        elif modifier_name == "add_outliers":
            lc = modifier(  # type: ignore[call-arg]
                lc,
                fraction=draw(st.floats(min_value=0.001, max_value=0.05)),
                amplitude=draw(st.floats(min_value=3.0, max_value=10.0)),
            )
        elif modifier_name == "add_gaps":
            lc = modifier(  # type: ignore[call-arg]
                lc,
                n_gaps=draw(st.integers(min_value=1, max_value=5)),
                gap_fraction=draw(st.floats(min_value=0.05, max_value=0.3)),
            )
        elif modifier_name == "add_trend":
            lc = modifier(  # type: ignore[call-arg]
                lc,
                trend_type=draw(st.sampled_from(["linear", "quadratic", "exponential"])),
                coefficient=draw(st.floats(min_value=-0.5, max_value=0.5)),
            )
        elif modifier_name == "add_flares":
            lc = modifier(  # type: ignore[call-arg]
                lc,
                n_flares=draw(st.integers(min_value=1, max_value=10)),
                min_amplitude=draw(st.floats(min_value=0.1, max_value=0.5)),
                max_amplitude=draw(st.floats(min_value=0.5, max_value=2.0)),
            )
        else:
            # Custom modifier - use provided params or defaults
            if modifier_params and modifier.__name__ in modifier_params:
                lc = modifier(lc, **modifier_params[modifier.__name__])
            else:
                lc = modifier(lc)

    return lc


# Backward compatibility - keep original functions but implement using new pattern


@st.composite
def lightcurves(
    draw: st.DrawFn,
    min_points: int = 10,
    max_points: int = 1000,
    min_time: float = 0.0,
    max_time: float = 100.0,
    min_flux: float = 0.0,
    max_flux: float = 1e6,
    with_errors: bool = False,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> Lightcurve:
    """Generate random lightcurves for property-based testing.

    This function maintains backward compatibility with the original API.

    Args:
        draw: Hypothesis draw function
        min_points: Minimum number of data points
        max_points: Maximum number of data points
        min_time: Minimum time value
        max_time: Maximum time value
        min_flux: Minimum flux value
        max_flux: Maximum flux value
        with_errors: Whether to include flux errors
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values

    Returns:
        A randomly generated Lightcurve object
    """
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))

    # Generate time array (sorted)
    time = draw(
        npst.arrays(
            dtype=np.float64,
            shape=n_points,
            elements=st.floats(
                min_value=min_time,
                max_value=max_time,
                allow_nan=allow_nan,
                allow_infinity=allow_inf,
            ),
            unique=True,
        )
    )
    time = np.sort(time)

    # Generate flux array
    flux = draw(
        npst.arrays(
            dtype=np.float64,
            shape=n_points,
            elements=st.floats(
                min_value=min_flux,
                max_value=max_flux,
                allow_nan=allow_nan,
                allow_infinity=allow_inf,
            ),
        )
    )

    # Optionally generate flux errors
    flux_err: np.ndarray | None = None
    if with_errors:
        flux_err = draw(
            npst.arrays(
                dtype=np.float64,
                shape=n_points,
                elements=st.floats(
                    min_value=0.0,
                    max_value=max_flux * 0.1,  # Errors up to 10% of max flux
                    allow_nan=allow_nan,
                    allow_infinity=False,  # Errors shouldn't be infinite
                ),
            )
        )

    return Lightcurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        modifications=["random"],
    )


@st.composite
def periodic_lightcurves(
    draw: st.DrawFn,
    min_points: int = 100,
    max_points: int = 1000,
    min_period: float = 0.1,
    max_period: float = 10.0,
    min_amplitude: float = 0.01,
    max_amplitude: float = 1.0,
    with_noise: bool = True,
) -> Lightcurve:
    """Generate periodic lightcurves with sinusoidal patterns.

    This function maintains backward compatibility with the original API,
    but now uses the composable pattern internally.

    Args:
        draw: Hypothesis draw function
        min_points: Minimum number of data points
        max_points: Maximum number of data points
        min_period: Minimum period value
        max_period: Maximum period value
        min_amplitude: Minimum amplitude
        max_amplitude: Maximum amplitude
        with_noise: Whether to add noise to the signal

    Returns:
        A randomly generated periodic Lightcurve object
    """
    # Create baseline using the new flexible API
    baseline_flux = draw(st.floats(min_value=10, max_value=100))
    lc = draw(
        baseline_lightcurves(
            n_points=st.integers(min_value=min_points, max_value=max_points),
            baseline_type="flat",
            baseline_flux=baseline_flux,
            time_sampling="uniform",
        )
    )

    # Add periodic signal
    period = draw(st.floats(min_value=min_period, max_value=max_period))
    amplitude = draw(st.floats(min_value=min_amplitude, max_value=max_amplitude)) * baseline_flux
    phase = draw(st.floats(min_value=0, max_value=2 * np.pi))

    lc = add_periodic_signal(lc, period=period, amplitude=amplitude, phase=phase)

    # Add noise if requested
    if with_noise:
        noise_level = draw(st.floats(min_value=0.001, max_value=0.01)) * amplitude
        lc = add_noise(lc, noise_type="gaussian", level=noise_level)

    return lc


@st.composite
def transient_lightcurves(
    draw: st.DrawFn,
    min_points: int = 50,
    max_points: int = 500,
    min_peak_time: float = 10.0,
    max_peak_time: float = 50.0,
    min_rise_time: float = 1.0,
    max_rise_time: float = 10.0,
    min_decay_time: float = 5.0,
    max_decay_time: float = 50.0,
) -> Lightcurve:
    """Generate transient lightcurves (e.g., supernovae-like).

    This function maintains backward compatibility with the original API,
    but now uses the composable pattern internally.

    Args:
        draw: Hypothesis draw function
        min_points: Minimum number of data points
        max_points: Maximum number of data points
        min_peak_time: Minimum time of peak
        max_peak_time: Maximum time of peak
        min_rise_time: Minimum rise time
        max_rise_time: Maximum rise time
        min_decay_time: Minimum decay time
        max_decay_time: Maximum decay time

    Returns:
        A randomly generated transient Lightcurve object
    """
    # Create baseline
    baseline_flux = draw(st.floats(min_value=10, max_value=100))
    total_duration = max_peak_time + max_decay_time * 3

    lc = draw(
        baseline_lightcurves(
            n_points=st.integers(min_value=min_points, max_value=max_points),
            min_time=0.0,
            max_time=total_duration,
            baseline_type="flat",
            baseline_flux=baseline_flux,
            time_sampling="uniform",
        )
    )

    # Add transient event
    peak_time = draw(st.floats(min_value=min_peak_time, max_value=max_peak_time))
    peak_flux = draw(st.floats(min_value=100, max_value=10000))
    rise_time = draw(st.floats(min_value=min_rise_time, max_value=max_rise_time))
    decay_time = draw(st.floats(min_value=min_decay_time, max_value=max_decay_time))

    lc = add_transient_event(
        lc,
        peak_time=peak_time,
        peak_flux=peak_flux,
        rise_time=rise_time,
        decay_time=decay_time,
        event_type="burst",
    )

    return lc
