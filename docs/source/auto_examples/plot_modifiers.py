"""
Lightcurve Modifiers
====================

This example shows how to apply various modifiers to lightcurves
to simulate realistic observational effects.
"""

# %%
# Import necessary libraries

import matplotlib.pyplot as plt

from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.modifiers import (
    add_gaps,
    add_noise,
    add_outliers,
    add_trend,
)
from hypothesis_lightcurves.visualization import plot_lightcurve_comparison

# %%
# Start with a clean periodic signal
# -----------------------------------

base_lc = periodic_lightcurves(
    min_points=400,
    max_points=400,
    min_period=1.4,
    max_period=1.6,
    min_amplitude=0.09,
    max_amplitude=0.11,
    with_noise=False,
).example()

fig, ax = base_lc.plot(title="Clean Periodic Signal")
plt.show()

# %%
# Add data gaps
# -------------
# Simulate observational gaps in the data

lc_with_gaps = add_gaps(base_lc, n_gaps=3, gap_fraction=0.1)

fig, axes = plot_lightcurve_comparison(
    base_lc, lc_with_gaps, label1="Original", label2="With Gaps", title="Effect of Data Gaps"
)
plt.show()

print(f"Original points: {base_lc.n_points}")
print(f"After gaps: {lc_with_gaps.n_points}")
print(f"Points removed: {base_lc.n_points - lc_with_gaps.n_points}")

# %%
# Add outliers
# ------------
# Introduce outlier contamination

lc_with_outliers = add_outliers(base_lc, fraction=0.05, amplitude=5.0)

fig, axes = plot_lightcurve_comparison(
    base_lc, lc_with_outliers, label1="Clean", label2="With Outliers", title="Outlier Contamination"
)
plt.show()

# %%
# Add systematic trends
# ---------------------
# Apply linear and quadratic trends

lc_linear = add_trend(base_lc, trend_type="linear", coefficient=0.15)
lc_quadratic = add_trend(base_lc, trend_type="quadratic", coefficient=0.2)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

base_lc.plot(ax=axes[0], title="Original Signal")
lc_linear.plot(ax=axes[1], title="With Linear Trend", color="C1")
lc_quadratic.plot(ax=axes[2], title="With Quadratic Trend", color="C2")

plt.tight_layout()
plt.show()

# %%
# Combine multiple effects
# -------------------------
# Create a realistic observational scenario

# Start with periodic signal
realistic_lc = periodic_lightcurves(
    min_points=500,
    max_points=500,
    min_period=0.7,
    max_period=0.8,
    min_amplitude=0.07,
    max_amplitude=0.09,
    with_noise=True,
).example()

# Apply modifications sequentially
realistic_lc = add_noise(realistic_lc, level=0.01)
realistic_lc = add_gaps(realistic_lc, n_gaps=2, gap_fraction=0.08)
realistic_lc = add_outliers(realistic_lc, fraction=0.02, amplitude=4)
realistic_lc = add_trend(realistic_lc, trend_type="linear", coefficient=0.05)

# Plot the result
fig, ax = realistic_lc.plot(title="Realistic Observational Data")
ax.text(
    0.02,
    0.98,
    f"Modifications: {', '.join(realistic_lc.modifications)}",
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
)
plt.show()

print("Applied modifications:")
for mod in realistic_lc.modifications:
    print(f"  - {mod}")

# %%
# Visualize the cumulative effect
# --------------------------------

# Create stages of modification
stages = []
lc = periodic_lightcurves(
    min_points=300,
    max_points=300,
    min_period=0.9,
    max_period=1.1,
    min_amplitude=0.11,
    max_amplitude=0.13,
    with_noise=False,
).example()
stages.append(("Original", lc.copy()))

lc = add_noise(lc, level=0.015)
stages.append(("+ Noise", lc.copy()))

lc = add_gaps(lc, n_gaps=2, gap_fraction=0.1)
stages.append(("+ Gaps", lc.copy()))

lc = add_outliers(lc, fraction=0.03, amplitude=5)
stages.append(("+ Outliers", lc.copy()))

# Plot all stages
fig, axes = plt.subplots(len(stages), 1, figsize=(12, 3 * len(stages)), sharex=True)

for ax, (label, stage_lc) in zip(axes, stages, strict=False):
    stage_lc.plot(ax=ax, title=label, markersize=3)

plt.tight_layout()
plt.show()
