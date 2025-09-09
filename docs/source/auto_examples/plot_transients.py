"""
Transient Lightcurve Examples
==============================

This example demonstrates the generation and visualization of transient lightcurves,
which model events like supernovae, novae, and stellar flares.
"""

import matplotlib.pyplot as plt
import numpy as np
from hypothesis_lightcurves.generators import transient_lightcurves
from hypothesis_lightcurves.visualization import (
    create_gallery_plot,
    plot_lightcurve,
    plot_with_annotations,
)

# %%
# Basic transient event
# ----------------------
# Generate and visualize a basic transient lightcurve.

np.random.seed(42)
lc = transient_lightcurves().example()

fig, ax = plt.subplots(figsize=(12, 6))
plot_with_annotations(lc, ax=ax, color="darkred", marker=".", markersize=3)
ax.set_title("Transient Event Example", fontsize=14)

# Mark the peak
peak_idx = np.argmax(lc.flux)
ax.plot(lc.time[peak_idx], lc.flux[peak_idx], "r*", markersize=15, label="Peak")
ax.legend()

plt.show()

# %%
# Different rise and decay times
# -------------------------------
# Compare transients with different temporal characteristics.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fast rise, fast decay (like a flare)
lc_flare = transient_lightcurves(
    min_rise_time=0.5,
    max_rise_time=0.5,
    min_decay_time=1.0,
    max_decay_time=1.0,
    min_peak_time=10,
    max_peak_time=10,
).example()
plot_lightcurve(
    lc_flare,
    ax=axes[0, 0],
    title="Flare-like (Fast rise, Fast decay)",
    color="orange",
    marker=".",
    markersize=2,
)

# Fast rise, slow decay (like a Type Ia supernova)
lc_sn_ia = transient_lightcurves(
    min_rise_time=2.0,
    max_rise_time=2.0,
    min_decay_time=15.0,
    max_decay_time=15.0,
    min_peak_time=20,
    max_peak_time=20,
).example()
plot_lightcurve(
    lc_sn_ia,
    ax=axes[0, 1],
    title="SN Ia-like (Fast rise, Slow decay)",
    color="blue",
    marker=".",
    markersize=2,
)

# Slow rise, slow decay (like a Type II supernova)
lc_sn_ii = transient_lightcurves(
    min_rise_time=5.0,
    max_rise_time=5.0,
    min_decay_time=20.0,
    max_decay_time=20.0,
    min_peak_time=25,
    max_peak_time=25,
).example()
plot_lightcurve(
    lc_sn_ii,
    ax=axes[1, 0],
    title="SN II-like (Slow rise, Slow decay)",
    color="green",
    marker=".",
    markersize=2,
)

# Slow rise, fast decay (unusual)
lc_unusual = transient_lightcurves(
    min_rise_time=8.0,
    max_rise_time=8.0,
    min_decay_time=2.0,
    max_decay_time=2.0,
    min_peak_time=20,
    max_peak_time=20,
).example()
plot_lightcurve(
    lc_unusual,
    ax=axes[1, 1],
    title="Unusual (Slow rise, Fast decay)",
    color="purple",
    marker=".",
    markersize=2,
)

plt.suptitle("Transient Events with Different Timescales", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Peak time variation
# -------------------
# Show how peak time affects the lightcurve.

peak_times = [10, 25, 40, 55]
fig, axes = plt.subplots(1, len(peak_times), figsize=(16, 4))

for i, peak_time in enumerate(peak_times):
    lc = transient_lightcurves(
        min_peak_time=peak_time,
        max_peak_time=peak_time,
        min_rise_time=3.0,
        max_rise_time=3.0,
        min_decay_time=10.0,
        max_decay_time=10.0,
        min_points=200,
        max_points=200,
    ).example()

    plot_lightcurve(lc, ax=axes[i], color=f"C{i}", marker="", linewidth=2)
    axes[i].set_title(f"Peak at t={peak_time}", fontsize=12)
    axes[i].axvline(peak_time, color="red", linestyle="--", alpha=0.5)

plt.suptitle("Transients with Different Peak Times", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# %%
# Amplitude variations
# --------------------
# Compare transients with different peak amplitudes.

fig, ax = plt.subplots(figsize=(12, 7))

amplitudes = [100, 500, 1000, 5000]
colors = ["blue", "green", "orange", "red"]

for amp, color in zip(amplitudes, colors, strict=False):
    lc = transient_lightcurves(
        min_peak_time=25,
        max_peak_time=25,
        min_rise_time=3.0,
        max_rise_time=3.0,
        min_decay_time=10.0,
        max_decay_time=10.0,
        min_points=150,
        max_points=150,
    ).example()

    # Scale the flux to desired amplitude
    baseline = np.min(lc.flux)
    scale_factor = amp / (np.max(lc.flux) - baseline)
    scaled_flux = baseline + (lc.flux - baseline) * scale_factor

    ax.plot(
        lc.time, scaled_flux, color=color, linewidth=2, label=f"Peak amplitude ≈ {amp}", alpha=0.7
    )

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Flux", fontsize=12)
ax.set_title("Transients with Different Amplitudes", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %%
# Evolution of a transient
# -------------------------
# Show the characteristic phases of a transient event.

lc_evolution = transient_lightcurves(
    min_peak_time=30,
    max_peak_time=30,
    min_rise_time=5.0,
    max_rise_time=5.0,
    min_decay_time=15.0,
    max_decay_time=15.0,
    min_points=300,
    max_points=300,
).example()

fig, ax = plt.subplots(figsize=(14, 7))

# Plot the full lightcurve
plot_lightcurve(lc_evolution, ax=ax, color="black", marker="", linewidth=2)

# Highlight different phases
peak_time = lc_evolution.metadata["peak_time"]
rise_time = lc_evolution.metadata["rise_time"]
decay_time = lc_evolution.metadata["decay_time"]

# Pre-explosion
ax.axvspan(0, peak_time - 3 * rise_time, alpha=0.2, color="gray", label="Pre-explosion")

# Rise phase
ax.axvspan(peak_time - 3 * rise_time, peak_time, alpha=0.2, color="blue", label="Rise phase")

# Peak
peak_idx = np.argmax(lc_evolution.flux)
ax.plot(lc_evolution.time[peak_idx], lc_evolution.flux[peak_idx], "r*", markersize=20, label="Peak")

# Decay phase
ax.axvspan(peak_time, peak_time + 3 * decay_time, alpha=0.2, color="orange", label="Decay phase")

# Late time
ax.axvspan(
    peak_time + 3 * decay_time, lc_evolution.time[-1], alpha=0.2, color="purple", label="Late time"
)

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Flux", fontsize=12)
ax.set_title("Phases of a Transient Event", fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.show()

# %%
# Gallery of transient events
# ----------------------------
# Showcase the diversity of transient lightcurves.

fig = create_gallery_plot(
    n_examples=9,
    generator_func=transient_lightcurves,
    title="Gallery of Transient Events",
    figsize=(15, 10),
    seed=789,
)
plt.show()

# %%
# Comparison with noise
# ---------------------
# Show how noise affects transient detection.

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Generate base transient
np.random.seed(100)
base_params = dict(
    min_peak_time=25,
    max_peak_time=25,
    min_rise_time=3.0,
    max_rise_time=3.0,
    min_decay_time=12.0,
    max_decay_time=12.0,
    min_points=200,
    max_points=200,
)

# Clean transient
lc_clean = transient_lightcurves(**base_params).example()
# Remove noise for clean version
lc_clean.flux = lc_clean.flux - np.random.normal(0, lc_clean.flux_err[0], len(lc_clean.flux))
lc_clean.flux_err = None

plot_lightcurve(lc_clean, ax=axes[0], title="Clean Transient", color="navy", marker="", linewidth=2)

# Low noise
lc_low_noise = transient_lightcurves(**base_params).example()
plot_lightcurve(
    lc_low_noise,
    ax=axes[1],
    title="Low Noise",
    color="darkgreen",
    marker=".",
    markersize=2,
    linestyle="",
)

# High noise (add extra noise)
lc_high_noise = transient_lightcurves(**base_params).example()
extra_noise = np.random.normal(0, np.std(lc_high_noise.flux) * 0.2, len(lc_high_noise.flux))
lc_high_noise.flux += extra_noise

plot_lightcurve(
    lc_high_noise,
    ax=axes[2],
    title="High Noise",
    color="darkred",
    marker=".",
    markersize=2,
    linestyle="",
)

plt.suptitle("Impact of Noise on Transient Detection", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
