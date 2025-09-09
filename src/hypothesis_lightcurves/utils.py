"""Utility functions for lightcurve analysis and manipulation.

This module provides common operations for working with lightcurves, including
resampling, binning, gap addition, and basic periodogram analysis.
"""

import numpy as np
import numpy.typing as npt

from hypothesis_lightcurves.models import Lightcurve


def resample_lightcurve(
    lightcurve: Lightcurve, n_points: int, method: str = "linear"
) -> Lightcurve:
    """Resample a lightcurve to a different number of points.

    This function is useful for standardizing lightcurves to a common
    sampling rate or reducing the data volume while preserving the
    overall shape of the lightcurve.

    Parameters
    ----------
    lightcurve : Lightcurve
        The input lightcurve to resample.
    n_points : int
        The desired number of points in the resampled lightcurve.
    method : {'linear', 'nearest'}, default='linear'
        The interpolation method to use:
        - 'linear': Linear interpolation between points
        - 'nearest': Nearest-neighbor interpolation

    Returns
    -------
    Lightcurve
        A new lightcurve with `n_points` evenly spaced samples.

    Raises
    ------
    ValueError
        If an unknown interpolation method is specified.

    Notes
    -----
    The resampled time array spans from the minimum to maximum time
    of the original lightcurve with uniform spacing. Flux errors,
    if present, are also interpolated using the same method.

    Examples
    --------
    >>> import numpy as np
    >>> from hypothesis_lightcurves.models import Lightcurve
    >>> from hypothesis_lightcurves.utils import resample_lightcurve
    >>>
    >>> # Create a lightcurve with 1000 points
    >>> time = np.linspace(0, 10, 1000)
    >>> flux = 100 + 5 * np.sin(2 * np.pi * time)
    >>> lc = Lightcurve(time=time, flux=flux)
    >>>
    >>> # Resample to 100 points
    >>> resampled = resample_lightcurve(lc, 100)
    >>> assert resampled.n_points == 100

    See Also
    --------
    bin_lightcurve : Bin lightcurve data by averaging.
    """
    new_time = np.linspace(lightcurve.time.min(), lightcurve.time.max(), n_points)

    if method == "linear":
        new_flux = np.interp(new_time, lightcurve.time, lightcurve.flux)
        if lightcurve.flux_err is not None:
            new_flux_err = np.interp(new_time, lightcurve.time, lightcurve.flux_err)
        else:
            new_flux_err = None
    elif method == "nearest":
        indices = np.searchsorted(lightcurve.time, new_time)
        indices = np.clip(indices, 0, len(lightcurve.time) - 1)
        new_flux = lightcurve.flux[indices]
        if lightcurve.flux_err is not None:
            new_flux_err = lightcurve.flux_err[indices]
        else:
            new_flux_err = None
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return Lightcurve(
        time=new_time,
        flux=new_flux,
        flux_err=new_flux_err,
        metadata=lightcurve.metadata,
    )


def add_gaps(
    lightcurve: Lightcurve, n_gaps: int = 1, gap_fraction: float = 0.1, seed: int | None = None
) -> Lightcurve:
    """Add observational gaps to a lightcurve by removing data points.

    This function simulates observational gaps that occur in real astronomical
    data due to weather, daylight, or other observing constraints.

    Parameters
    ----------
    lightcurve : Lightcurve
        The input lightcurve to modify.
    n_gaps : int, default=1
        The number of gaps to add to the lightcurve.
    gap_fraction : float, default=0.1
        The total fraction of data points to remove. This is divided
        equally among all gaps.
    seed : int, optional
        Random seed for reproducible gap placement.

    Returns
    -------
    Lightcurve
        A new lightcurve with gaps (missing data points).

    Notes
    -----
    The gaps are placed randomly throughout the lightcurve. Each gap
    removes a contiguous block of data points. The total number of
    removed points is approximately `gap_fraction * n_points`.

    Examples
    --------
    >>> from hypothesis_lightcurves.utils import add_gaps
    >>>
    >>> # Remove 20% of data in 2 gaps
    >>> lc_with_gaps = add_gaps(lc, n_gaps=2, gap_fraction=0.2, seed=42)
    >>>
    >>> # Verify approximately 80% of points remain
    >>> remaining_fraction = lc_with_gaps.n_points / lc.n_points
    >>> assert 0.75 < remaining_fraction < 0.85

    See Also
    --------
    resample_lightcurve : Change the sampling rate of a lightcurve.
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.ones(len(lightcurve.time), dtype=bool)
    points_per_gap = int(len(lightcurve.time) * gap_fraction / n_gaps)

    if points_per_gap == 0:
        return lightcurve.copy()  # No gaps to add

    for _ in range(n_gaps):
        if points_per_gap >= len(lightcurve.time[mask]):
            break  # Can't add more gaps
        gap_start = np.random.randint(0, len(lightcurve.time) - points_per_gap)
        mask[gap_start : gap_start + points_per_gap] = False

    return Lightcurve(
        time=lightcurve.time[mask],
        flux=lightcurve.flux[mask],
        flux_err=lightcurve.flux_err[mask] if lightcurve.flux_err is not None else None,
        metadata=lightcurve.metadata,
    )


def calculate_periodogram(
    lightcurve: Lightcurve, periods: npt.NDArray[np.float64] | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate a simple Lomb-Scargle-like periodogram.

    Computes the power spectrum of a lightcurve at specified periods
    using a simplified periodogram method. This is useful for detecting
    periodic signals in the data.

    Parameters
    ----------
    lightcurve : Lightcurve
        The input lightcurve to analyze.
    periods : numpy.ndarray, optional
        Array of periods at which to calculate the power.
        If None, automatically generates a range based on the data.

    Returns
    -------
    periods : numpy.ndarray
        The array of test periods.
    power : numpy.ndarray
        The calculated power at each period.

    Notes
    -----
    This implements a simplified periodogram calculation, not the full
    Lomb-Scargle algorithm. The lightcurve is normalized before analysis
    to remove the mean and scale by the standard deviation.

    The power is calculated as:

    .. math::

        P(\\omega) = \\frac{1}{N} \\sqrt{A^2 + B^2}

    where :math:`A = \\sum f_i \\cos(\\omega t_i)` and
    :math:`B = \\sum f_i \\sin(\\omega t_i)`, and :math:`\\omega = 2\\pi/P`.

    Examples
    --------
    >>> import numpy as np
    >>> from hypothesis_lightcurves.utils import calculate_periodogram
    >>>
    >>> # Create a periodic lightcurve
    >>> time = np.linspace(0, 10, 100)
    >>> flux = 100 + 5 * np.sin(2 * np.pi * time / 2.5)
    >>> lc = Lightcurve(time=time, flux=flux)
    >>>
    >>> # Calculate periodogram
    >>> periods = np.linspace(0.5, 5.0, 100)
    >>> test_periods, power = calculate_periodogram(lc, periods)
    >>>
    >>> # Peak should be near period = 2.5
    >>> peak_period = test_periods[np.argmax(power)]
    >>> assert abs(peak_period - 2.5) < 0.1

    See Also
    --------
    scipy.signal.lombscargle : Full Lomb-Scargle periodogram implementation.
    """
    if periods is None:
        # Generate default period range
        duration = lightcurve.duration
        min_period = duration / 100  # At least 100 cycles
        max_period = duration / 2  # At least 2 cycles
        periods = np.linspace(min_period, max_period, 1000)

    normalized = lightcurve.normalize()
    power = np.zeros(len(periods))

    for i, period in enumerate(periods):
        phase = (normalized.time % period) / period * 2 * np.pi
        a = np.sum(normalized.flux * np.cos(phase))
        b = np.sum(normalized.flux * np.sin(phase))
        power[i] = np.sqrt(a**2 + b**2) / len(normalized.flux)

    return periods, power


def bin_lightcurve(lightcurve: Lightcurve, bin_size: float) -> Lightcurve:
    """Bin a lightcurve by averaging flux values in time bins.

    Binning reduces the time resolution of a lightcurve by averaging
    multiple data points within each time bin. This can improve the
    signal-to-noise ratio and reduce data volume.

    Parameters
    ----------
    lightcurve : Lightcurve
        The input lightcurve to bin.
    bin_size : float
        The size of each time bin in the same units as the time array.

    Returns
    -------
    Lightcurve
        A new lightcurve with binned data. Each bin is represented
        by a single point at the mean time with the mean flux.

    Notes
    -----
    - Empty bins are omitted from the output.
    - The time value for each bin is the mean of the original time values
      within that bin.
    - Flux errors are combined in quadrature and scaled by the square root
      of the number of points in each bin.

    Examples
    --------
    >>> from hypothesis_lightcurves.utils import bin_lightcurve
    >>>
    >>> # Create a noisy lightcurve
    >>> time = np.linspace(0, 10, 1000)
    >>> flux = 100 + np.random.normal(0, 1, 1000)
    >>> lc = Lightcurve(time=time, flux=flux)
    >>>
    >>> # Bin to 0.1 time units
    >>> binned = bin_lightcurve(lc, bin_size=0.1)
    >>>
    >>> # Should have approximately 100 bins
    >>> assert 95 < binned.n_points < 105

    See Also
    --------
    resample_lightcurve : Resample to a specific number of points.
    numpy.histogram : NumPy's binning function.
    """
    min_time = lightcurve.time.min()
    max_time = lightcurve.time.max()

    bins = np.arange(min_time, max_time + bin_size, bin_size)
    bin_indices = np.digitize(lightcurve.time, bins) - 1

    binned_time = []
    binned_flux = []
    binned_flux_err = []

    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if np.any(mask):
            binned_time.append(np.mean(lightcurve.time[mask]))
            binned_flux.append(np.mean(lightcurve.flux[mask]))

            if lightcurve.flux_err is not None:
                # Combine errors in quadrature and scale by sqrt(n)
                n_points = np.sum(mask)
                combined_err = np.sqrt(np.sum(lightcurve.flux_err[mask] ** 2)) / np.sqrt(n_points)
                binned_flux_err.append(combined_err)

    return Lightcurve(
        time=np.array(binned_time),
        flux=np.array(binned_flux),
        flux_err=np.array(binned_flux_err) if lightcurve.flux_err is not None else None,
        metadata=lightcurve.metadata,
    )
