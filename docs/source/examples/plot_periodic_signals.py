"""
Periodic Lightcurve Examples
=============================

This example demonstrates the generation and visualization of periodic lightcurves,
which are useful for testing algorithms that detect periodic signals in astronomical data.
"""

import matplotlib.pyplot as plt
import numpy as np
from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.utils import calculate_periodogram
from hypothesis_lightcurves.visualization import (
    create_gallery_plot,
    plot_lightcurve,
    plot_phase_folded,
    plot_with_annotations,
)

# %%
# Simple periodic lightcurve
# ---------------------------
# Generate a periodic lightcurve with known period and amplitude.

np.random.seed(42)
lc = periodic_lightcurves(
    min_period=2.5,
    max_period=2.5,  # Fixed period for demonstration
    min_amplitude=0.1,
    max_amplitude=0.2,
    with_noise=False,
).example()

fig, ax = plt.subplots(figsize=(12, 6))
plot_with_annotations(lc, ax=ax, color="darkblue", marker="", linewidth=2)
ax.set_title(f"Periodic Lightcurve (P={lc.metadata['period']:.2f})", fontsize=14)
plt.show()

# %%
# Effect of noise on periodic signals
# ------------------------------------
# Compare clean and noisy periodic signals.

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Clean signal
lc_clean = periodic_lightcurves(
    min_period=3.0,
    max_period=3.0,
    min_amplitude=0.15,
    max_amplitude=0.15,
    with_noise=False,
    min_points=200,
    max_points=200,
).example()

plot_lightcurve(
    lc_clean, ax=axes[0], title="Clean Periodic Signal", color="navy", marker="", linewidth=2
)

# Noisy signal with same parameters
np.random.seed(42)  # Same seed for consistent period/amplitude
lc_noisy = periodic_lightcurves(
    min_period=3.0,
    max_period=3.0,
    min_amplitude=0.15,
    max_amplitude=0.15,
    with_noise=True,
    min_points=200,
    max_points=200,
).example()

plot_lightcurve(
    lc_noisy,
    ax=axes[1],
    title="Noisy Periodic Signal",
    color="darkred",
    marker=".",
    markersize=3,
    linestyle="",
)

plt.suptitle("Impact of Noise on Periodic Signals", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Different periods and amplitudes
# ---------------------------------
# Showcase various combinations of periods and amplitudes.

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

params = [
    (0.5, 0.05, "Short period, small amplitude"),
    (0.5, 0.3, "Short period, large amplitude"),
    (2.0, 0.05, "Medium period, small amplitude"),
    (2.0, 0.3, "Medium period, large amplitude"),
    (5.0, 0.05, "Long period, small amplitude"),
    (5.0, 0.3, "Long period, large amplitude"),
]

for idx, (period, amplitude, title) in enumerate(params):
    row = idx // 2
    col = idx % 2

    lc = periodic_lightcurves(
        min_period=period,
        max_period=period,
        min_amplitude=amplitude,
        max_amplitude=amplitude,
        with_noise=False,
        min_points=300,
        max_points=300,
    ).example()

    plot_lightcurve(lc, ax=axes[row, col], title=title, color=f"C{idx}", marker="", linewidth=1.5)
    axes[row, col].set_ylim(lc.mean_flux - amplitude * 2, lc.mean_flux + amplitude * 2)

plt.suptitle("Periodic Signals with Various Parameters", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Phase-folded lightcurve
# ------------------------
# Demonstrate phase-folding to reveal the periodic pattern.

lc_fold = periodic_lightcurves(
    min_period=1.7,
    max_period=1.7,
    min_amplitude=0.2,
    max_amplitude=0.2,
    with_noise=True,
    min_points=500,
    max_points=500,
).example()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original lightcurve
plot_lightcurve(
    lc_fold,
    ax=axes[0],
    title="Original Lightcurve",
    color="darkgreen",
    marker=".",
    markersize=2,
    linestyle="",
)

# Phase-folded
true_period = lc_fold.metadata["period"]
plot_phase_folded(lc_fold, period=true_period, ax=axes[1])
axes[1].set_title(f"Phase-folded at True Period (P={true_period:.3f})")

plt.tight_layout()
plt.show()

# %%
# Periodogram analysis
# ---------------------
# Show how periodogram can recover the true period.

lc_periodogram = periodic_lightcurves(
    min_period=2.3,
    max_period=2.3,
    min_amplitude=0.15,
    max_amplitude=0.15,
    with_noise=True,
    min_points=200,
    max_points=200,
).example()

# Calculate periodogram
test_periods = np.linspace(0.5, 5.0, 1000)
periods, power = calculate_periodogram(lc_periodogram, test_periods)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Lightcurve
plot_lightcurve(
    lc_periodogram,
    ax=axes[0],
    title="Periodic Lightcurve for Analysis",
    color="purple",
    marker=".",
    markersize=3,
    linestyle="-",
    linewidth=0.5,
)

# Periodogram
axes[1].plot(periods, power, "b-", linewidth=1.5)
true_period = lc_periodogram.metadata["period"]
axes[1].axvline(true_period, color="red", linestyle="--", label=f"True Period = {true_period:.3f}")
detected_period = periods[np.argmax(power)]
axes[1].axvline(
    detected_period, color="green", linestyle="--", label=f"Detected Period = {detected_period:.3f}"
)
axes[1].set_xlabel("Period", fontsize=12)
axes[1].set_ylabel("Power", fontsize=12)
axes[1].set_title("Periodogram Analysis", fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Gallery of periodic lightcurves
# --------------------------------
# Show diversity of periodic lightcurves that can be generated.

fig = create_gallery_plot(
    n_examples=9,
    generator_func=periodic_lightcurves,
    title="Gallery of Periodic Lightcurves",
    figsize=(15, 10),
    seed=456,
    min_period=0.5,
    max_period=5.0,
    with_noise=True,
)
plt.show()

# %%
# Multi-period comparison
# ------------------------
# Compare lightcurves with different periods side by side.

periods_to_compare = [0.5, 1.0, 2.0, 4.0]
fig, axes = plt.subplots(len(periods_to_compare), 1, figsize=(12, 10), sharex=True)

for i, period in enumerate(periods_to_compare):
    lc = periodic_lightcurves(
        min_period=period,
        max_period=period,
        min_amplitude=0.1,
        max_amplitude=0.1,
        with_noise=False,
        min_points=500,
        max_points=500,
    ).example()

    plot_lightcurve(lc, ax=axes[i], color=f"C{i}", marker="", linewidth=1.5)
    axes[i].set_title(f"Period = {period:.1f}", fontsize=12)
    axes[i].set_ylabel("Flux", fontsize=10)

axes[-1].set_xlabel("Time", fontsize=12)
plt.suptitle("Comparison of Different Periods", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
