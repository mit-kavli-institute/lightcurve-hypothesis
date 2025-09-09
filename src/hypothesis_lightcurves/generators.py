"""Hypothesis strategies for generating synthetic lightcurves.

This module provides Hypothesis strategies for generating various types of
lightcurves for property-based testing. It includes both simple random
lightcurves and specialized generators for periodic and transient phenomena.
"""

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

    This is the most general lightcurve generator, producing random time series
    with configurable constraints. It's useful for testing code that should work
    with any valid lightcurve, regardless of its physical properties.

    Parameters
    ----------
    draw : hypothesis.strategies.DrawFn
        Hypothesis draw function for generating random values.
    min_points : int, default=10
        Minimum number of data points to generate.
    max_points : int, default=1000
        Maximum number of data points to generate.
    min_time : float, default=0.0
        Minimum allowed time value.
    max_time : float, default=100.0
        Maximum allowed time value.
    min_flux : float, default=0.0
        Minimum allowed flux value.
    max_flux : float, default=1e6
        Maximum allowed flux value.
    with_errors : bool, default=False
        Whether to include flux error values.
    allow_nan : bool, default=False
        Whether to allow NaN values in the data.
    allow_inf : bool, default=False
        Whether to allow infinite values in the data.

    Returns
    -------
    Lightcurve
        A randomly generated lightcurve object.

    Examples
    --------
    >>> from hypothesis import given
    >>> from hypothesis_lightcurves.generators import lightcurves
    >>>
    >>> @given(lc=lightcurves())
    ... def test_lightcurve_properties(lc):
    ...     assert lc.n_points >= 10
    ...     assert lc.n_points <= 1000
    ...     assert np.all(np.isfinite(lc.flux))

    Notes
    -----
    The generated time array is always sorted in ascending order.
    Error values, when generated, are limited to 10% of the maximum flux value.

    See Also
    --------
    periodic_lightcurves : Generate lightcurves with periodic signals.
    transient_lightcurves : Generate lightcurves with transient events.
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

    Creates lightcurves with a sinusoidal variation, useful for testing
    algorithms that detect or analyze periodic signals, such as those from
    variable stars, exoplanet transits, or rotating objects.

    Parameters
    ----------
    draw : hypothesis.strategies.DrawFn
        Hypothesis draw function for generating random values.
    min_points : int, default=100
        Minimum number of data points to generate.
    max_points : int, default=1000
        Maximum number of data points to generate.
    min_period : float, default=0.1
        Minimum period of the sinusoidal signal.
    max_period : float, default=10.0
        Maximum period of the sinusoidal signal.
    min_amplitude : float, default=0.01
        Minimum amplitude of the sinusoidal variation.
    max_amplitude : float, default=1.0
        Maximum amplitude of the sinusoidal variation.
    with_noise : bool, default=True
        Whether to add Gaussian noise to the signal.

    Returns
    -------
    Lightcurve
        A lightcurve with periodic sinusoidal variation.

    Notes
    -----
    The generated lightcurve follows the model:

    .. math::

        f(t) = A_0 + A \\sin(2\\pi t / P + \\phi) + \\epsilon

    where :math:`A_0` is the baseline flux, :math:`A` is the amplitude,
    :math:`P` is the period, :math:`\\phi` is the phase, and :math:`\\epsilon`
    is optional Gaussian noise.

    The metadata dictionary contains the following keys:
    - 'period': The period of the signal
    - 'amplitude': The amplitude of the signal
    - 'phase': The phase offset in radians

    Examples
    --------
    >>> from hypothesis import given, settings
    >>> from hypothesis_lightcurves.generators import periodic_lightcurves
    >>>
    >>> @given(lc=periodic_lightcurves(min_period=1.0, max_period=2.0))
    >>> @settings(max_examples=10)
    ... def test_period_range(lc):
    ...     assert 1.0 <= lc.metadata['period'] <= 2.0

    See Also
    --------
    lightcurves : General random lightcurve generator.
    transient_lightcurves : Generator for transient events.
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
    """Generate transient lightcurves with rise and decay phases.

    Creates lightcurves that model transient astronomical events such as
    supernovae, novae, or stellar flares. These events are characterized
    by a rapid rise to peak brightness followed by a slower decay.

    Parameters
    ----------
    draw : hypothesis.strategies.DrawFn
        Hypothesis draw function for generating random values.
    min_points : int, default=50
        Minimum number of data points to generate.
    max_points : int, default=500
        Maximum number of data points to generate.
    min_peak_time : float, default=10.0
        Minimum time of peak brightness.
    max_peak_time : float, default=50.0
        Maximum time of peak brightness.
    min_rise_time : float, default=1.0
        Minimum e-folding rise time.
    max_rise_time : float, default=10.0
        Maximum e-folding rise time.
    min_decay_time : float, default=5.0
        Minimum e-folding decay time.
    max_decay_time : float, default=50.0
        Maximum e-folding decay time.

    Returns
    -------
    Lightcurve
        A lightcurve with a transient event.

    Notes
    -----
    The transient profile follows an exponential rise and decay model:

    - For t < peak_time: :math:`f(t) = f_0 + A \\exp(-(t_p - t)/\\tau_r)`
    - For t >= peak_time: :math:`f(t) = f_0 + A \\exp(-(t - t_p)/\\tau_d)`

    where :math:`f_0` is the baseline flux, :math:`A` is the peak amplitude,
    :math:`t_p` is the peak time, :math:`\\tau_r` is the rise time, and
    :math:`\\tau_d` is the decay time.

    The metadata dictionary contains:
    - 'peak_time': Time of peak brightness
    - 'rise_time': E-folding rise time
    - 'decay_time': E-folding decay time
    - 'peak_flux': Peak flux value above baseline

    Examples
    --------
    >>> from hypothesis import given
    >>> from hypothesis_lightcurves.generators import transient_lightcurves
    >>>
    >>> @given(lc=transient_lightcurves())
    ... def test_transient_properties(lc):
    ...     peak_idx = np.argmax(lc.flux)
    ...     # Peak should be near the specified peak_time
    ...     assert abs(lc.time[peak_idx] - lc.metadata['peak_time']) < 5.0

    See Also
    --------
    lightcurves : General random lightcurve generator.
    periodic_lightcurves : Generator for periodic signals.
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
