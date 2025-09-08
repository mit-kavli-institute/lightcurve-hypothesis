"""Modifier functions for transforming lightcurves."""

from typing import Literal

import numpy as np

from hypothesis_lightcurves.models import Lightcurve


def add_periodic_signal(
    lightcurve: Lightcurve,
    period: float,
    amplitude: float,
    phase: float = 0.0,
) -> Lightcurve:
    """Add a periodic sinusoidal signal to a lightcurve.

    Args:
        lightcurve: Input lightcurve
        period: Period of the signal
        amplitude: Amplitude of the signal
        phase: Phase offset in radians

    Returns:
        Modified lightcurve with periodic signal
    """
    lc = lightcurve.copy()
    signal = amplitude * np.sin(2 * np.pi * lc.time / period + phase)
    lc.flux = lc.flux + signal

    # Update metadata and modifications
    if lc.metadata is None:
        lc.metadata = {}
    lc.metadata.update(
        {
            "periodic_period": period,
            "periodic_amplitude": amplitude,
            "periodic_phase": phase,
        }
    )
    lc.modifications.append(f"periodic_signal(period={period:.2f}, amplitude={amplitude:.3f})")

    return lc


def add_noise(
    lightcurve: Lightcurve,
    noise_type: Literal["gaussian", "poisson", "uniform"] = "gaussian",
    level: float = 0.01,
    seed: int | None = None,
) -> Lightcurve:
    """Add noise to a lightcurve.

    Args:
        lightcurve: Input lightcurve
        noise_type: Type of noise to add
        level: Noise level (interpretation depends on noise type)
        seed: Random seed for reproducibility

    Returns:
        Modified lightcurve with added noise
    """
    lc = lightcurve.copy()

    if seed is not None:
        np.random.seed(seed)

    if noise_type == "gaussian":
        noise = np.random.normal(0, level, len(lc.flux))
        if lc.flux_err is None:
            lc.flux_err = np.full(len(lc.flux), level)
        else:
            # Combine errors in quadrature
            lc.flux_err = np.sqrt(lc.flux_err**2 + level**2)
    elif noise_type == "poisson":
        # Poisson noise scales with sqrt of signal
        noise = np.random.normal(0, level * np.sqrt(np.abs(lc.flux)))
        if lc.flux_err is None:
            lc.flux_err = level * np.sqrt(np.abs(lc.flux))
        else:
            lc.flux_err = np.sqrt(lc.flux_err**2 + (level * np.sqrt(np.abs(lc.flux))) ** 2)
    elif noise_type == "uniform":
        noise = np.random.uniform(-level, level, len(lc.flux))
        if lc.flux_err is None:
            lc.flux_err = np.full(len(lc.flux), level / np.sqrt(3))  # std of uniform dist
        else:
            lc.flux_err = np.sqrt(lc.flux_err**2 + (level / np.sqrt(3)) ** 2)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    lc.flux = lc.flux + noise

    # Update metadata and modifications
    if lc.metadata is None:
        lc.metadata = {}
    lc.metadata[f"noise_{noise_type}_level"] = level
    lc.modifications.append(f"noise({noise_type}, level={level:.3f})")

    return lc


def add_transient_event(
    lightcurve: Lightcurve,
    peak_time: float,
    peak_flux: float,
    rise_time: float = 1.0,
    decay_time: float = 5.0,
    event_type: Literal["burst", "dip"] = "burst",
) -> Lightcurve:
    """Add a transient event (burst or dip) to a lightcurve.

    Args:
        lightcurve: Input lightcurve
        peak_time: Time of peak/minimum
        peak_flux: Peak amplitude (positive for burst, negative for dip)
        rise_time: Rise time scale
        decay_time: Decay time scale
        event_type: Type of event

    Returns:
        Modified lightcurve with transient event
    """
    lc = lightcurve.copy()

    # Generate transient profile
    transient = np.zeros(len(lc.time))
    for i, t in enumerate(lc.time):
        if t < peak_time:
            # Rising phase
            transient[i] = peak_flux * np.exp(-(peak_time - t) / rise_time)
        else:
            # Decay phase
            transient[i] = peak_flux * np.exp(-(t - peak_time) / decay_time)

    if event_type == "dip":
        transient = -np.abs(transient)

    lc.flux = lc.flux + transient

    # Update metadata and modifications
    if lc.metadata is None:
        lc.metadata = {}
    lc.metadata.update(
        {
            f"{event_type}_peak_time": peak_time,
            f"{event_type}_peak_flux": peak_flux,
            f"{event_type}_rise_time": rise_time,
            f"{event_type}_decay_time": decay_time,
        }
    )
    lc.modifications.append(f"transient_{event_type}(t={peak_time:.1f}, peak={peak_flux:.1f})")

    return lc


def add_outliers(
    lightcurve: Lightcurve,
    fraction: float = 0.01,
    amplitude: float = 5.0,
    seed: int | None = None,
) -> Lightcurve:
    """Add outlier points to a lightcurve.

    Args:
        lightcurve: Input lightcurve
        fraction: Fraction of points to make outliers
        amplitude: Outlier amplitude (in units of flux std)
        seed: Random seed for reproducibility

    Returns:
        Modified lightcurve with outliers
    """
    lc = lightcurve.copy()

    if seed is not None:
        np.random.seed(seed)

    n_outliers = int(len(lc.flux) * fraction)
    if n_outliers > 0:
        outlier_indices = np.random.choice(len(lc.flux), n_outliers, replace=False)
        outlier_values = np.random.choice([-1, 1], n_outliers) * amplitude * lc.std_flux
        lc.flux[outlier_indices] += outlier_values

        # Update metadata and modifications
        if lc.metadata is None:
            lc.metadata = {}
        lc.metadata["outlier_fraction"] = fraction
        lc.metadata["outlier_amplitude"] = amplitude
        lc.metadata["outlier_indices"] = outlier_indices.tolist()
        lc.modifications.append(f"outliers(fraction={fraction:.3f}, amplitude={amplitude:.1f})")

    return lc


def add_gaps(
    lightcurve: Lightcurve,
    n_gaps: int = 1,
    gap_fraction: float = 0.1,
    seed: int | None = None,
) -> Lightcurve:
    """Add observational gaps to a lightcurve by removing data points.

    Args:
        lightcurve: Input lightcurve
        n_gaps: Number of gaps to add
        gap_fraction: Total fraction of data to remove
        seed: Random seed for reproducibility

    Returns:
        Modified lightcurve with gaps
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.ones(len(lightcurve.time), dtype=bool)
    points_per_gap = int(len(lightcurve.time) * gap_fraction / n_gaps)

    gap_starts = []
    for _ in range(n_gaps):
        if points_per_gap > 0 and np.sum(mask) > points_per_gap:
            # Find a valid gap start position
            valid_positions = np.where(mask)[0]
            if len(valid_positions) > points_per_gap:
                gap_start_idx = np.random.choice(len(valid_positions) - points_per_gap)
                gap_start = valid_positions[gap_start_idx]
                mask[gap_start : gap_start + points_per_gap] = False
                gap_starts.append(int(gap_start))

    # Create new lightcurve with gaps
    lc = Lightcurve(
        time=lightcurve.time[mask],
        flux=lightcurve.flux[mask],
        flux_err=lightcurve.flux_err[mask] if lightcurve.flux_err is not None else None,
        metadata=lightcurve.metadata.copy() if lightcurve.metadata else {},
        modifications=lightcurve.modifications.copy(),
    )

    # Update metadata and modifications
    if lc.metadata is None:
        lc.metadata = {}
    lc.metadata["gap_count"] = n_gaps
    lc.metadata["gap_fraction"] = gap_fraction
    lc.metadata["gap_starts"] = gap_starts
    lc.modifications.append(f"gaps(n={n_gaps}, fraction={gap_fraction:.2f})")

    return lc


def add_trend(
    lightcurve: Lightcurve,
    trend_type: Literal["linear", "quadratic", "exponential"] = "linear",
    coefficient: float = 0.01,
) -> Lightcurve:
    """Add a trend to a lightcurve.

    Args:
        lightcurve: Input lightcurve
        trend_type: Type of trend to add
        coefficient: Trend coefficient

    Returns:
        Modified lightcurve with trend
    """
    lc = lightcurve.copy()

    # Normalize time to [0, 1] for stable coefficients
    t_norm = (lc.time - lc.time.min()) / (lc.time.max() - lc.time.min())

    if trend_type == "linear":
        trend = coefficient * t_norm * lc.mean_flux
    elif trend_type == "quadratic":
        trend = coefficient * t_norm**2 * lc.mean_flux
    elif trend_type == "exponential":
        trend = lc.mean_flux * (np.exp(coefficient * t_norm) - 1)
    else:
        raise ValueError(f"Unknown trend type: {trend_type}")

    lc.flux = lc.flux + trend

    # Update metadata and modifications
    if lc.metadata is None:
        lc.metadata = {}
    lc.metadata[f"trend_{trend_type}_coefficient"] = coefficient
    lc.modifications.append(f"trend({trend_type}, coef={coefficient:.3f})")

    return lc


def add_flares(
    lightcurve: Lightcurve,
    n_flares: int = 3,
    min_amplitude: float = 0.5,
    max_amplitude: float = 2.0,
    min_duration: float = 0.01,
    max_duration: float = 0.05,
    seed: int | None = None,
) -> Lightcurve:
    """Add stellar flare events to a lightcurve.

    Args:
        lightcurve: Input lightcurve
        n_flares: Number of flares to add
        min_amplitude: Minimum flare amplitude (relative to mean flux)
        max_amplitude: Maximum flare amplitude (relative to mean flux)
        min_duration: Minimum flare duration (fraction of total duration)
        max_duration: Maximum flare duration (fraction of total duration)
        seed: Random seed for reproducibility

    Returns:
        Modified lightcurve with flares
    """
    lc = lightcurve.copy()

    if seed is not None:
        np.random.seed(seed)

    total_duration = lc.duration
    flare_info = []

    for _ in range(n_flares):
        # Random flare parameters
        flare_time = np.random.uniform(lc.time.min(), lc.time.max())
        flare_amplitude = np.random.uniform(min_amplitude, max_amplitude) * lc.mean_flux
        flare_duration = np.random.uniform(min_duration, max_duration) * total_duration

        # Fast rise, slow decay profile
        rise_time = flare_duration * 0.1  # 10% of duration for rise
        decay_time = flare_duration * 0.9  # 90% of duration for decay

        # Add flare profile
        for i, t in enumerate(lc.time):
            dt = t - flare_time
            if 0 <= dt < rise_time:
                # Fast rise
                lc.flux[i] += flare_amplitude * (dt / rise_time)
            elif rise_time <= dt < flare_duration:
                # Exponential decay
                decay_phase = (dt - rise_time) / decay_time
                lc.flux[i] += flare_amplitude * np.exp(-3 * decay_phase)

        flare_info.append(
            {
                "time": flare_time,
                "amplitude": flare_amplitude,
                "duration": flare_duration,
            }
        )

    # Update metadata and modifications
    if lc.metadata is None:
        lc.metadata = {}
    lc.metadata["flare_count"] = n_flares
    lc.metadata["flare_info"] = flare_info
    lc.modifications.append(f"flares(n={n_flares})")

    return lc
