"""Hypothesis strategies for generating lightcurves."""

from typing import Optional, Callable

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from hypothesis_lightcurves.models import Lightcurve
from hypothesis_lightcurves.modifiers import (
    add_periodic_signal,
    add_transient_event,
    add_noise,
    add_outliers,
    add_gaps,
    add_trend,
    add_flares,
)


@st.composite
def baseline_lightcurves(
    draw: st.DrawFn,
    min_points: int = 10,
    max_points: int = 1000,
    min_time: float = 0.0,
    max_time: float = 100.0,
    baseline_type: str = "flat",
    baseline_flux: float = 100.0,
    time_sampling: str = "uniform",
) -> Lightcurve:
    """Generate baseline lightcurves with simple patterns.
    
    Args:
        draw: Hypothesis draw function
        min_points: Minimum number of data points
        max_points: Maximum number of data points
        min_time: Minimum time value
        max_time: Maximum time value
        baseline_type: Type of baseline ('flat', 'random_walk', 'smooth')
        baseline_flux: Base flux level for flat baseline
        time_sampling: Time sampling pattern ('uniform', 'random', 'irregular')
    
    Returns:
        A baseline Lightcurve object
    """
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    
    # Generate time array based on sampling pattern
    if time_sampling == "uniform":
        duration = draw(st.floats(min_value=max_time/2, max_value=max_time))
        start_time = draw(st.floats(min_value=min_time, max_value=min_time + 10))
        time = np.linspace(start_time, start_time + duration, n_points)
    elif time_sampling == "random":
        time = draw(
            npst.arrays(
                dtype=np.float64,
                shape=n_points,
                elements=st.floats(min_value=min_time, max_value=max_time),
                unique=True,
            )
        )
        time = np.sort(time)
    else:  # irregular
        # Create irregular sampling with clusters
        n_clusters = draw(st.integers(min_value=3, max_value=10))
        points_per_cluster = n_points // n_clusters
        time_list = []
        for i in range(n_clusters):
            cluster_center = draw(st.floats(min_value=min_time, max_value=max_time))
            cluster_width = draw(st.floats(min_value=0.5, max_value=5.0))
            cluster_times = np.random.normal(cluster_center, cluster_width, points_per_cluster)
            time_list.extend(cluster_times)
        time = np.sort(np.array(time_list[:n_points]))
    
    # Generate baseline flux based on type
    if baseline_type == "flat":
        flux_value = draw(st.floats(min_value=baseline_flux * 0.8, max_value=baseline_flux * 1.2))
        flux = np.full(n_points, flux_value)
    elif baseline_type == "random_walk":
        # Generate smooth random walk
        step_size = draw(st.floats(min_value=0.001, max_value=0.01))
        steps = np.random.normal(0, step_size * baseline_flux, n_points)
        flux = baseline_flux + np.cumsum(steps)
    else:  # smooth
        # Generate smooth baseline using low-frequency sinusoids
        n_components = draw(st.integers(min_value=1, max_value=3))
        flux = np.full(n_points, baseline_flux)
        for _ in range(n_components):
            period = draw(st.floats(min_value=max_time/2, max_value=max_time*2))
            amplitude = draw(st.floats(min_value=0.01, max_value=0.05)) * baseline_flux
            phase = draw(st.floats(min_value=0, max_value=2*np.pi))
            flux += amplitude * np.sin(2 * np.pi * time / period + phase)
    
    return Lightcurve(
        time=time,
        flux=flux,
        flux_err=None,
        metadata={"baseline_type": baseline_type, "time_sampling": time_sampling},
        modifications=["baseline"],
    )


@st.composite
def modified_lightcurves(
    draw: st.DrawFn,
    base_strategy: Optional[st.SearchStrategy[Lightcurve]] = None,
    modifiers: Optional[list[Callable[[Lightcurve], Lightcurve]]] = None,
    modifier_params: Optional[dict] = None,
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
        modifiers = [add_noise, add_periodic_signal, add_outliers]
    
    # Apply random subset of modifiers
    n_modifiers = draw(st.integers(min_value=0, max_value=len(modifiers)))
    selected_modifiers = draw(st.lists(
        st.sampled_from(modifiers),
        min_size=n_modifiers,
        max_size=n_modifiers,
        unique=True,
    ))
    
    # Apply each modifier with random parameters
    for modifier in selected_modifiers:
        if modifier == add_noise:
            lc = modifier(
                lc,
                noise_type=draw(st.sampled_from(["gaussian", "poisson", "uniform"])),
                level=draw(st.floats(min_value=0.001, max_value=0.1)) * lc.std_flux,
            )
        elif modifier == add_periodic_signal:
            duration = lc.duration
            lc = modifier(
                lc,
                period=draw(st.floats(min_value=duration/20, max_value=duration/2)),
                amplitude=draw(st.floats(min_value=0.01, max_value=0.3)) * lc.std_flux,
                phase=draw(st.floats(min_value=0, max_value=2*np.pi)),
            )
        elif modifier == add_transient_event:
            lc = modifier(
                lc,
                peak_time=draw(st.floats(min_value=lc.time.min(), max_value=lc.time.max())),
                peak_flux=draw(st.floats(min_value=0.5, max_value=3.0)) * lc.std_flux,
                rise_time=draw(st.floats(min_value=0.01, max_value=0.1)) * lc.duration,
                decay_time=draw(st.floats(min_value=0.05, max_value=0.3)) * lc.duration,
                event_type=draw(st.sampled_from(["burst", "dip"])),
            )
        elif modifier == add_outliers:
            lc = modifier(
                lc,
                fraction=draw(st.floats(min_value=0.001, max_value=0.05)),
                amplitude=draw(st.floats(min_value=3.0, max_value=10.0)),
            )
        elif modifier == add_gaps:
            lc = modifier(
                lc,
                n_gaps=draw(st.integers(min_value=1, max_value=5)),
                gap_fraction=draw(st.floats(min_value=0.05, max_value=0.3)),
            )
        elif modifier == add_trend:
            lc = modifier(
                lc,
                trend_type=draw(st.sampled_from(["linear", "quadratic", "exponential"])),
                coefficient=draw(st.floats(min_value=-0.5, max_value=0.5)),
            )
        elif modifier == add_flares:
            lc = modifier(
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
    flux_err: Optional[np.ndarray] = None
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
    # Create baseline
    baseline_flux = draw(st.floats(min_value=10, max_value=100))
    lc = draw(baseline_lightcurves(
        min_points=min_points,
        max_points=max_points,
        baseline_type="flat",
        baseline_flux=baseline_flux,
        time_sampling="uniform",
    ))
    
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
    
    lc = draw(baseline_lightcurves(
        min_points=min_points,
        max_points=max_points,
        min_time=0.0,
        max_time=total_duration,
        baseline_type="flat",
        baseline_flux=baseline_flux,
        time_sampling="uniform",
    ))
    
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
    
    # Add some noise
    noise_level = peak_flux * 0.01
    lc = add_noise(lc, noise_type="gaussian", level=noise_level)
    
    return lc