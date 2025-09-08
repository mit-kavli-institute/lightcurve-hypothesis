"""Hypothesis strategies for generating lightcurves."""

from typing import Optional

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from hypothesis_lightcurves.models import Lightcurve


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
    
    return Lightcurve(time=time, flux=flux, flux_err=flux_err)


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
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    period = draw(st.floats(min_value=min_period, max_value=max_period))
    amplitude = draw(st.floats(min_value=min_amplitude, max_value=max_amplitude))
    phase = draw(st.floats(min_value=0, max_value=2 * np.pi))
    baseline = draw(st.floats(min_value=0, max_value=100))
    
    # Generate evenly spaced time array
    duration = draw(st.floats(min_value=period * 2, max_value=period * 20))
    time = np.linspace(0, duration, n_points)
    
    # Generate periodic signal
    flux = baseline + amplitude * np.sin(2 * np.pi * time / period + phase)
    
    # Add noise if requested
    if with_noise:
        noise_level = draw(st.floats(min_value=0.001, max_value=amplitude * 0.1))
        noise = np.random.normal(0, noise_level, n_points)
        flux += noise
        flux_err = np.full(n_points, noise_level)
    else:
        flux_err = None
    
    return Lightcurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        metadata={"period": period, "amplitude": amplitude, "phase": phase},
    )


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
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    peak_time = draw(st.floats(min_value=min_peak_time, max_value=max_peak_time))
    rise_time = draw(st.floats(min_value=min_rise_time, max_value=max_rise_time))
    decay_time = draw(st.floats(min_value=min_decay_time, max_value=max_decay_time))
    peak_flux = draw(st.floats(min_value=100, max_value=10000))
    baseline = draw(st.floats(min_value=0, max_value=100))
    
    # Generate time array
    total_duration = peak_time + decay_time * 3
    time = np.linspace(0, total_duration, n_points)
    
    # Generate transient profile
    flux = np.zeros(n_points) + baseline
    for i, t in enumerate(time):
        if t < peak_time:
            # Rising phase
            flux[i] += peak_flux * np.exp(-(peak_time - t) / rise_time)
        else:
            # Decay phase
            flux[i] += peak_flux * np.exp(-(t - peak_time) / decay_time)
    
    # Add some noise
    noise_level = peak_flux * 0.01
    flux += np.random.normal(0, noise_level, n_points)
    
    return Lightcurve(
        time=time,
        flux=flux,
        flux_err=np.full(n_points, noise_level),
        metadata={
            "peak_time": peak_time,
            "rise_time": rise_time,
            "decay_time": decay_time,
            "peak_flux": peak_flux,
        },
    )