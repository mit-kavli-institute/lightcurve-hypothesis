"""Utility functions for lightcurve analysis and manipulation."""

from typing import Tuple

import numpy as np
import numpy.typing as npt

from hypothesis_lightcurves.models import Lightcurve


def resample_lightcurve(
    lightcurve: Lightcurve, n_points: int, method: str = "linear"
) -> Lightcurve:
    """Resample a lightcurve to a different number of points.
    
    Args:
        lightcurve: Input lightcurve
        n_points: Number of points in the resampled lightcurve
        method: Interpolation method ('linear' or 'nearest')
    
    Returns:
        Resampled lightcurve
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
    lightcurve: Lightcurve, n_gaps: int = 1, gap_fraction: float = 0.1
) -> Lightcurve:
    """Add gaps to a lightcurve by removing data points.
    
    Args:
        lightcurve: Input lightcurve
        n_gaps: Number of gaps to add
        gap_fraction: Fraction of data to remove for each gap
    
    Returns:
        Lightcurve with gaps
    """
    mask = np.ones(len(lightcurve.time), dtype=bool)
    points_per_gap = int(len(lightcurve.time) * gap_fraction / n_gaps)
    
    for _ in range(n_gaps):
        gap_start = np.random.randint(0, len(lightcurve.time) - points_per_gap)
        mask[gap_start : gap_start + points_per_gap] = False
    
    return Lightcurve(
        time=lightcurve.time[mask],
        flux=lightcurve.flux[mask],
        flux_err=lightcurve.flux_err[mask] if lightcurve.flux_err is not None else None,
        metadata=lightcurve.metadata,
    )


def calculate_periodogram(
    lightcurve: Lightcurve, periods: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate a simple Lomb-Scargle-like periodogram.
    
    Args:
        lightcurve: Input lightcurve
        periods: Array of periods to test
    
    Returns:
        Tuple of (periods, power) arrays
    """
    normalized = lightcurve.normalize()
    power = np.zeros(len(periods))
    
    for i, period in enumerate(periods):
        phase = (normalized.time % period) / period * 2 * np.pi
        a = np.sum(normalized.flux * np.cos(phase))
        b = np.sum(normalized.flux * np.sin(phase))
        power[i] = np.sqrt(a**2 + b**2) / len(normalized.flux)
    
    return periods, power


def bin_lightcurve(lightcurve: Lightcurve, bin_size: float) -> Lightcurve:
    """Bin a lightcurve by averaging flux in time bins.
    
    Args:
        lightcurve: Input lightcurve
        bin_size: Size of time bins
    
    Returns:
        Binned lightcurve
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
                combined_err = np.sqrt(np.sum(lightcurve.flux_err[mask] ** 2)) / n_points
                binned_flux_err.append(combined_err)
    
    return Lightcurve(
        time=np.array(binned_time),
        flux=np.array(binned_flux),
        flux_err=np.array(binned_flux_err) if lightcurve.flux_err is not None else None,
        metadata=lightcurve.metadata,
    )