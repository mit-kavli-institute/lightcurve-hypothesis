"""
Ensemble Statistics
====================

This example demonstrates statistical analysis of ensemble lightcurve generation,
showing distributions and properties across many generated examples.
"""

import matplotlib.pyplot as plt
import numpy as np
from hypothesis_lightcurves.generators import (
    lightcurves,
    periodic_lightcurves,
    transient_lightcurves,
)
from hypothesis_lightcurves.utils import calculate_periodogram
from hypothesis_lightcurves.visualization import plot_lightcurve

# %%
# Statistical properties of random lightcurves
# ---------------------------------------------
# Analyze the distribution of properties across many random lightcurves.

np.random.seed(42)
n_samples = 100

# Generate ensemble
ensemble = [lightcurves(min_points=100, max_points=500).example() for _ in range(n_samples)]

# Extract properties
n_points = [lc.n_points for lc in ensemble]
durations = [lc.duration for lc in ensemble]
mean_fluxes = [lc.mean_flux for lc in ensemble]
std_fluxes = [lc.std_flux for lc in ensemble]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Number of points distribution
axes[0, 0].hist(n_points, bins=20, color="blue", alpha=0.7, edgecolor="black")
axes[0, 0].set_xlabel("Number of Points")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title(f"Point Count Distribution (n={n_samples})")
axes[0, 0].axvline(
    np.mean(n_points), color="red", linestyle="--", label=f"Mean: {np.mean(n_points):.0f}"
)
axes[0, 0].legend()

# Duration distribution
axes[0, 1].hist(durations, bins=20, color="green", alpha=0.7, edgecolor="black")
axes[0, 1].set_xlabel("Duration")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Duration Distribution")
axes[0, 1].axvline(
    np.mean(durations), color="red", linestyle="--", label=f"Mean: {np.mean(durations):.1f}"
)
axes[0, 1].legend()

# Mean flux distribution
axes[1, 0].hist(mean_fluxes, bins=20, color="orange", alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("Mean Flux")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Mean Flux Distribution")
axes[1, 0].axvline(
    np.mean(mean_fluxes), color="red", linestyle="--", label=f"Mean: {np.mean(mean_fluxes):.1f}"
)
axes[1, 0].legend()

# Standard deviation distribution
axes[1, 1].hist(std_fluxes, bins=20, color="purple", alpha=0.7, edgecolor="black")
axes[1, 1].set_xlabel("Flux Std Dev")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Flux Standard Deviation Distribution")
axes[1, 1].axvline(
    np.mean(std_fluxes), color="red", linestyle="--", label=f"Mean: {np.mean(std_fluxes):.1f}"
)
axes[1, 1].legend()

plt.suptitle(f"Ensemble Statistics of {n_samples} Random Lightcurves", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Period recovery statistics
# ---------------------------
# Test how well we can recover periods from generated periodic lightcurves.

n_tests = 50
true_periods = []
detected_periods = []
recovery_errors = []

for _ in range(n_tests):
    # Generate periodic lightcurve
    lc = periodic_lightcurves(
        min_period=1.0,
        max_period=5.0,
        min_amplitude=0.1,
        max_amplitude=0.3,
        with_noise=True,
        min_points=200,
        max_points=400,
    ).example()

    true_period = lc.metadata["period"]
    true_periods.append(true_period)

    # Try to recover the period
    test_periods = np.linspace(0.5, 10.0, 1000)
    periods, power = calculate_periodogram(lc, test_periods)
    detected_period = periods[np.argmax(power)]
    detected_periods.append(detected_period)

    # Calculate relative error
    error = abs(detected_period - true_period) / true_period * 100
    recovery_errors.append(error)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# True vs Detected periods
axes[0].scatter(true_periods, detected_periods, alpha=0.6, s=30)
axes[0].plot([0, 6], [0, 6], "r--", label="Perfect Recovery")
axes[0].set_xlabel("True Period")
axes[0].set_ylabel("Detected Period")
axes[0].set_title(f"Period Recovery (n={n_tests})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error distribution
axes[1].hist(recovery_errors, bins=15, color="green", alpha=0.7, edgecolor="black")
axes[1].set_xlabel("Relative Error (%)")
axes[1].set_ylabel("Count")
axes[1].set_title("Period Recovery Error Distribution")
axes[1].axvline(
    np.median(recovery_errors),
    color="red",
    linestyle="--",
    label=f"Median: {np.median(recovery_errors):.1f}%",
)
axes[1].legend()

# Error vs True Period
axes[2].scatter(true_periods, recovery_errors, alpha=0.6, s=30, color="purple")
axes[2].set_xlabel("True Period")
axes[2].set_ylabel("Relative Error (%)")
axes[2].set_title("Recovery Error vs True Period")
axes[2].axhline(5, color="red", linestyle="--", alpha=0.5, label="5% threshold")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Period Recovery Statistics from {n_tests} Periodic Lightcurves", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

print(f"Recovery success rate (< 5% error): {sum(e < 5 for e in recovery_errors)/n_tests*100:.1f}%")
print(f"Mean recovery error: {np.mean(recovery_errors):.2f}%")
print(f"Median recovery error: {np.median(recovery_errors):.2f}%")

# %%
# Transient peak detection accuracy
# ----------------------------------
# Analyze how accurately we can detect transient peaks.

n_transients = 50
peak_time_errors = []
peak_flux_errors = []

for _ in range(n_transients):
    lc = transient_lightcurves(
        min_peak_time=20,
        max_peak_time=60,
        min_rise_time=1.0,
        max_rise_time=5.0,
        min_decay_time=5.0,
        max_decay_time=20.0,
    ).example()

    true_peak_time = lc.metadata["peak_time"]
    true_peak_flux = lc.metadata["peak_flux"]

    # Detect peak
    peak_idx = np.argmax(lc.flux)
    detected_peak_time = lc.time[peak_idx]
    detected_peak_flux = lc.flux[peak_idx] - np.min(lc.flux)

    # Calculate errors
    time_error = abs(detected_peak_time - true_peak_time)
    flux_error = abs(detected_peak_flux - true_peak_flux) / true_peak_flux * 100

    peak_time_errors.append(time_error)
    peak_flux_errors.append(flux_error)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Peak time error distribution
axes[0].hist(peak_time_errors, bins=15, color="blue", alpha=0.7, edgecolor="black")
axes[0].set_xlabel("Peak Time Error")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Peak Time Detection Error (n={n_transients})")
axes[0].axvline(
    np.median(peak_time_errors),
    color="red",
    linestyle="--",
    label=f"Median: {np.median(peak_time_errors):.2f}",
)
axes[0].legend()

# Peak flux error distribution
axes[1].hist(peak_flux_errors, bins=15, color="orange", alpha=0.7, edgecolor="black")
axes[1].set_xlabel("Peak Flux Error (%)")
axes[1].set_ylabel("Count")
axes[1].set_title("Peak Flux Detection Error")
axes[1].axvline(
    np.median(peak_flux_errors),
    color="red",
    linestyle="--",
    label=f"Median: {np.median(peak_flux_errors):.1f}%",
)
axes[1].legend()

plt.suptitle(f"Transient Peak Detection Statistics (n={n_transients})", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Ensemble visualization
# -----------------------
# Visualize multiple examples from each generator type.

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Row 1: Random lightcurves
for i in range(3):
    lc = lightcurves(with_errors=True).example()
    plot_lightcurve(lc, ax=axes[0, i], color=f"C{i}", marker="", linewidth=1)
    axes[0, i].set_title(f"Random {i+1}")

# Row 2: Periodic lightcurves
for i in range(3):
    lc = periodic_lightcurves(min_period=1.0 + i, max_period=1.0 + i, with_noise=True).example()
    plot_lightcurve(lc, ax=axes[1, i], color=f"C{i+3}", marker="", linewidth=1)
    axes[1, i].set_title(f"Periodic (P≈{1.0+i:.1f})")

# Row 3: Transient lightcurves
for i in range(3):
    lc = transient_lightcurves(min_rise_time=1.0 + i * 2, max_rise_time=1.0 + i * 2).example()
    plot_lightcurve(lc, ax=axes[2, i], color=f"C{i+6}", marker=".", markersize=2, linestyle="")
    axes[2, i].set_title(f"Transient (τ_r≈{1.0+i*2:.1f})")

plt.suptitle("Ensemble Examples from Different Generators", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Signal-to-noise ratio analysis
# -------------------------------
# Analyze SNR for periodic signals with different noise levels.

periods_test = 2.5
amplitudes = [0.05, 0.1, 0.2, 0.5]
n_realizations = 20

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, amplitude in enumerate(amplitudes):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    snr_values = []

    for _ in range(n_realizations):
        lc = periodic_lightcurves(
            min_period=periods_test,
            max_period=periods_test,
            min_amplitude=amplitude,
            max_amplitude=amplitude,
            with_noise=True,
            min_points=300,
            max_points=300,
        ).example()

        # Estimate SNR
        if lc.flux_err is not None:
            noise_level = np.mean(lc.flux_err)
        else:
            # Estimate noise from high-frequency components
            noise_level = np.std(np.diff(lc.flux)) / np.sqrt(2)

        signal_amplitude = amplitude
        snr = signal_amplitude / noise_level if noise_level > 0 else np.inf
        snr_values.append(snr)

    # Plot one example
    plot_lightcurve(lc, ax=ax, color="gray", alpha=0.5, marker=".", markersize=1, linestyle="")

    # Add SNR info
    mean_snr = np.mean(snr_values)
    ax.set_title(f"Amplitude={amplitude:.2f}, Mean SNR={mean_snr:.1f}")

    # Add text box with statistics
    stats_text = f"SNR: {mean_snr:.1f} ± {np.std(snr_values):.1f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

plt.suptitle("Signal-to-Noise Ratio Analysis for Different Amplitudes", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Coverage test for parameter ranges
# -----------------------------------
# Verify that generated parameters cover the requested ranges.

n_samples = 100
param_coverage = {
    "periods": [],
    "amplitudes": [],
    "peak_times": [],
    "rise_times": [],
}

# Generate periodic lightcurves
for _ in range(n_samples):
    lc = periodic_lightcurves(
        min_period=1.0, max_period=5.0, min_amplitude=0.05, max_amplitude=0.5
    ).example()
    param_coverage["periods"].append(lc.metadata["period"])
    param_coverage["amplitudes"].append(lc.metadata["amplitude"])

# Generate transient lightcurves
for _ in range(n_samples):
    lc = transient_lightcurves(
        min_peak_time=10, max_peak_time=50, min_rise_time=1.0, max_rise_time=10.0
    ).example()
    param_coverage["peak_times"].append(lc.metadata["peak_time"])
    param_coverage["rise_times"].append(lc.metadata["rise_time"])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Periods
axes[0, 0].hist(param_coverage["periods"], bins=20, color="blue", alpha=0.7, edgecolor="black")
axes[0, 0].axvline(1.0, color="red", linestyle="--", label="Min")
axes[0, 0].axvline(5.0, color="red", linestyle="--", label="Max")
axes[0, 0].set_xlabel("Period")
axes[0, 0].set_title("Period Coverage (requested: 1.0-5.0)")
axes[0, 0].legend()

# Amplitudes
axes[0, 1].hist(param_coverage["amplitudes"], bins=20, color="green", alpha=0.7, edgecolor="black")
axes[0, 1].axvline(0.05, color="red", linestyle="--", label="Min")
axes[0, 1].axvline(0.5, color="red", linestyle="--", label="Max")
axes[0, 1].set_xlabel("Amplitude")
axes[0, 1].set_title("Amplitude Coverage (requested: 0.05-0.5)")
axes[0, 1].legend()

# Peak times
axes[1, 0].hist(param_coverage["peak_times"], bins=20, color="orange", alpha=0.7, edgecolor="black")
axes[1, 0].axvline(10, color="red", linestyle="--", label="Min")
axes[1, 0].axvline(50, color="red", linestyle="--", label="Max")
axes[1, 0].set_xlabel("Peak Time")
axes[1, 0].set_title("Peak Time Coverage (requested: 10-50)")
axes[1, 0].legend()

# Rise times
axes[1, 1].hist(param_coverage["rise_times"], bins=20, color="purple", alpha=0.7, edgecolor="black")
axes[1, 1].axvline(1.0, color="red", linestyle="--", label="Min")
axes[1, 1].axvline(10.0, color="red", linestyle="--", label="Max")
axes[1, 1].set_xlabel("Rise Time")
axes[1, 1].set_title("Rise Time Coverage (requested: 1.0-10.0)")
axes[1, 1].legend()

plt.suptitle(f"Parameter Coverage Test (n={n_samples} each)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
