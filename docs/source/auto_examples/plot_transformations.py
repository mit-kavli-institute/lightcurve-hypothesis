"""
Lightcurve Transformations
===========================

This example demonstrates various transformations that can be applied to lightcurves,
including resampling, binning, gap addition, and normalization.
"""

import matplotlib.pyplot as plt
import numpy as np
from hypothesis_lightcurves.generators import periodic_lightcurves, transient_lightcurves
from hypothesis_lightcurves.utils import (
    add_gaps,
    bin_lightcurve,
    calculate_periodogram,
    resample_lightcurve,
)
from hypothesis_lightcurves.visualization import plot_lightcurve

# %%
# Resampling demonstration
# -------------------------
# Show how resampling affects lightcurve resolution.

np.random.seed(42)
original = periodic_lightcurves(
    min_points=500, max_points=500, min_period=2.5, max_period=2.5, with_noise=True
).example()

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Original
plot_lightcurve(
    original,
    ax=axes[0],
    title=f"Original ({original.n_points} points)",
    color="navy",
    marker=".",
    markersize=2,
    linestyle="-",
    linewidth=0.5,
)

# Downsampled
downsampled = resample_lightcurve(original, n_points=50)
plot_lightcurve(
    downsampled,
    ax=axes[1],
    title=f"Downsampled ({downsampled.n_points} points)",
    color="darkgreen",
    marker="o",
    markersize=5,
    linestyle="-",
    linewidth=1,
)

# Upsampled
upsampled = resample_lightcurve(original, n_points=1000)
plot_lightcurve(
    upsampled,
    ax=axes[2],
    title=f"Upsampled ({upsampled.n_points} points)",
    color="darkred",
    marker="",
    linestyle="-",
    linewidth=1,
)

plt.suptitle("Effect of Resampling on Lightcurves", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Binning for noise reduction
# ----------------------------
# Demonstrate how binning can reduce noise in lightcurves.

# Generate noisy periodic lightcurve
noisy_lc = periodic_lightcurves(
    min_points=1000,
    max_points=1000,
    min_period=3.0,
    max_period=3.0,
    min_amplitude=0.1,
    max_amplitude=0.1,
    with_noise=True,
).example()

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Original noisy data
plot_lightcurve(
    noisy_lc,
    ax=axes[0],
    title=f"Original Noisy Data ({noisy_lc.n_points} points)",
    color="gray",
    marker=".",
    markersize=1,
    linestyle="",
    alpha=0.5,
)

# Small bins
binned_small = bin_lightcurve(noisy_lc, bin_size=0.5)
plot_lightcurve(
    binned_small,
    ax=axes[1],
    title=f"Small Bins (size=0.5, {binned_small.n_points} bins)",
    color="blue",
    marker="o",
    markersize=4,
    linestyle="-",
    linewidth=1,
)

# Large bins
binned_large = bin_lightcurve(noisy_lc, bin_size=2.0)
plot_lightcurve(
    binned_large,
    ax=axes[2],
    title=f"Large Bins (size=2.0, {binned_large.n_points} bins)",
    color="red",
    marker="s",
    markersize=6,
    linestyle="-",
    linewidth=2,
)

plt.suptitle("Binning for Noise Reduction", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Adding observational gaps
# --------------------------
# Simulate realistic observational gaps in the data.

# Generate continuous lightcurve
continuous = transient_lightcurves(
    min_points=300, max_points=300, min_peak_time=30, max_peak_time=30
).example()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original continuous
plot_lightcurve(
    continuous,
    ax=axes[0, 0],
    title=f"Original Continuous ({continuous.n_points} points)",
    color="black",
    marker=".",
    markersize=2,
)

# Single gap
single_gap = add_gaps(continuous, n_gaps=1, gap_fraction=0.15, seed=42)
plot_lightcurve(
    single_gap,
    ax=axes[0, 1],
    title=f"Single Gap (15% removed, {single_gap.n_points} points)",
    color="blue",
    marker=".",
    markersize=3,
)

# Multiple small gaps
multi_small = add_gaps(continuous, n_gaps=3, gap_fraction=0.2, seed=43)
plot_lightcurve(
    multi_small,
    ax=axes[1, 0],
    title=f"3 Small Gaps (20% removed, {multi_small.n_points} points)",
    color="green",
    marker=".",
    markersize=3,
)

# Many gaps (heavily sampled)
many_gaps = add_gaps(continuous, n_gaps=5, gap_fraction=0.4, seed=44)
plot_lightcurve(
    many_gaps,
    ax=axes[1, 1],
    title=f"5 Gaps (40% removed, {many_gaps.n_points} points)",
    color="red",
    marker=".",
    markersize=3,
)

plt.suptitle("Effect of Observational Gaps", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Normalization comparison
# -------------------------
# Show the effect of normalization on different lightcurves.

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Generate lightcurves with different characteristics
lc_types = [
    (
        "Low baseline",
        periodic_lightcurves(
            min_points=200, max_points=200, min_period=2.0, max_period=2.0
        ).example(),
    ),
    (
        "High baseline",
        periodic_lightcurves(
            min_points=200, max_points=200, min_period=2.0, max_period=2.0
        ).example(),
    ),
    ("Transient", transient_lightcurves(min_points=200, max_points=200).example()),
]

# Scale the high baseline
lc_types[1] = (lc_types[1][0], lc_types[1][1])
lc_types[1][1].flux = lc_types[1][1].flux + 1000

for i, (name, lc) in enumerate(lc_types):
    # Original
    plot_lightcurve(
        lc, ax=axes[i, 0], title=f"{name} - Original", color=f"C{i}", marker="", linewidth=1.5
    )
    axes[i, 0].text(
        0.02,
        0.98,
        f"Mean: {lc.mean_flux:.1f}\nStd: {lc.std_flux:.1f}",
        transform=axes[i, 0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Normalized
    normalized = lc.normalize()
    plot_lightcurve(
        normalized,
        ax=axes[i, 1],
        title=f"{name} - Normalized",
        color=f"C{i}",
        marker="",
        linewidth=1.5,
    )
    axes[i, 1].text(
        0.02,
        0.98,
        f"Mean: {normalized.mean_flux:.1e}\nStd: {normalized.std_flux:.3f}",
        transform=axes[i, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    axes[i, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)

plt.suptitle("Normalization of Different Lightcurve Types", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Combined transformations
# -------------------------
# Apply multiple transformations to show cumulative effects.

# Start with a high-resolution periodic lightcurve
original_combined = periodic_lightcurves(
    min_points=1000,
    max_points=1000,
    min_period=2.0,
    max_period=2.0,
    min_amplitude=0.15,
    max_amplitude=0.15,
    with_noise=True,
).example()

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# Step 1: Original
plot_lightcurve(
    original_combined,
    ax=axes[0],
    title=f"1. Original ({original_combined.n_points} points)",
    color="black",
    marker="",
    linewidth=0.5,
    alpha=0.7,
)

# Step 2: Add gaps
with_gaps = add_gaps(original_combined, n_gaps=2, gap_fraction=0.2, seed=50)
plot_lightcurve(
    with_gaps,
    ax=axes[1],
    title=f"2. With Gaps ({with_gaps.n_points} points, 20% removed)",
    color="blue",
    marker=".",
    markersize=2,
    linestyle="",
)

# Step 3: Bin the gapped data
binned = bin_lightcurve(with_gaps, bin_size=1.0)
plot_lightcurve(
    binned,
    ax=axes[2],
    title=f"3. Binned (bin_size=1.0, {binned.n_points} bins)",
    color="green",
    marker="o",
    markersize=5,
    linestyle="-",
    linewidth=1.5,
)

# Step 4: Resample to regular grid
resampled = resample_lightcurve(binned, n_points=100)
plot_lightcurve(
    resampled,
    ax=axes[3],
    title=f"4. Resampled ({resampled.n_points} points)",
    color="orange",
    marker="s",
    markersize=4,
    linestyle="-",
    linewidth=1,
)

# Step 5: Normalize
normalized_final = resampled.normalize()
plot_lightcurve(
    normalized_final,
    ax=axes[4],
    title="5. Normalized (mean≈0, std≈1)",
    color="red",
    marker="",
    linestyle="-",
    linewidth=2,
)
axes[4].axhline(0, color="gray", linestyle="--", alpha=0.5)

plt.suptitle("Sequential Transformations Pipeline", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Effect on periodogram
# ---------------------
# Show how transformations affect period detection.

# Generate a periodic signal
lc_period = periodic_lightcurves(
    min_points=500,
    max_points=500,
    min_period=3.14159,  # Use pi for easy recognition
    max_period=3.14159,
    min_amplitude=0.2,
    max_amplitude=0.2,
    with_noise=True,
).example()

# Apply transformations
lc_gapped = add_gaps(lc_period, n_gaps=3, gap_fraction=0.3, seed=60)
lc_binned = bin_lightcurve(lc_period, bin_size=0.5)

# Calculate periodograms
test_periods = np.linspace(1.0, 6.0, 1000)
_, power_original = calculate_periodogram(lc_period, test_periods)
_, power_gapped = calculate_periodogram(lc_gapped, test_periods)
_, power_binned = calculate_periodogram(lc_binned, test_periods)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original
plot_lightcurve(
    lc_period, ax=axes[0, 0], title="Original Lightcurve", color="navy", marker="", linewidth=0.5
)
axes[0, 1].plot(test_periods, power_original, "navy", linewidth=1.5)
axes[0, 1].axvline(3.14159, color="red", linestyle="--", alpha=0.5, label="True Period")
axes[0, 1].set_title("Original Periodogram")
axes[0, 1].legend()

# With gaps
plot_lightcurve(
    lc_gapped,
    ax=axes[1, 0],
    title="With Gaps (30% removed)",
    color="green",
    marker=".",
    markersize=2,
    linestyle="",
)
axes[1, 1].plot(test_periods, power_gapped, "green", linewidth=1.5)
axes[1, 1].axvline(3.14159, color="red", linestyle="--", alpha=0.5, label="True Period")
axes[1, 1].set_title("Periodogram with Gaps")
axes[1, 1].legend()

# Binned
plot_lightcurve(
    lc_binned,
    ax=axes[2, 0],
    title="Binned (size=0.5)",
    color="orange",
    marker="o",
    markersize=4,
    linestyle="-",
)
axes[2, 1].plot(test_periods, power_binned, "orange", linewidth=1.5)
axes[2, 1].axvline(3.14159, color="red", linestyle="--", alpha=0.5, label="True Period")
axes[2, 1].set_title("Binned Periodogram")
axes[2, 1].legend()

# Format periodogram axes
for i in range(3):
    axes[i, 1].set_ylabel("Power")
    axes[i, 1].grid(True, alpha=0.3)
axes[2, 1].set_xlabel("Period")

plt.suptitle("Impact of Transformations on Period Detection", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
