"""
Basic Lightcurve Generation
============================

This example demonstrates the basic usage of hypothesis-lightcurves
to generate and visualize synthetic lightcurves.
"""

# %%
# Import necessary libraries

import matplotlib.pyplot as plt

from hypothesis_lightcurves.generators import lightcurves, periodic_lightcurves
from hypothesis_lightcurves.modifiers import add_noise

# %%
# Generate a basic lightcurve
# ----------------------------
# Create a simple random lightcurve using the basic generator

lc = lightcurves(min_points=200, max_points=200).example()
print(f"Generated lightcurve with {lc.n_points} points")
print(f"Duration: {lc.duration:.2f}")
print(f"Mean flux: {lc.mean_flux:.2f}")
print(f"Std flux: {lc.std_flux:.2f}")

# %%
# Plot the basic lightcurve

fig, ax = lc.plot(title="Basic Random Lightcurve")
plt.show()

# %%
# Generate a periodic lightcurve
# -------------------------------
# Create a lightcurve with a known period and amplitude

periodic_lc = periodic_lightcurves(
    min_points=300,
    max_points=300,
    min_period=2.4,
    max_period=2.6,
    min_amplitude=0.14,
    max_amplitude=0.16,
    with_noise=False,
).example()

fig, ax = periodic_lc.plot(title="Periodic Lightcurve (P=2.5)")
plt.show()

# %%
# Add realistic noise
# -------------------
# Apply noise to make the lightcurve more realistic

noisy_lc = add_noise(periodic_lc, level=0.02)

# Plot both for comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

periodic_lc.plot(ax=ax1, title="Original Periodic Signal")
noisy_lc.plot(ax=ax2, title="With Added Noise")

plt.tight_layout()
plt.show()

# %%
# Generate lightcurve with errors
# --------------------------------
# Create a lightcurve that includes measurement uncertainties

lc_with_errors = lightcurves(min_points=150, max_points=150, with_errors=True).example()

fig, ax = lc_with_errors.plot(title="Lightcurve with Measurement Errors", show_errors=True)
plt.show()

# %%
# Normalized lightcurve
# ----------------------
# Normalize a lightcurve to zero mean and unit variance

original = periodic_lightcurves(min_amplitude=0.4, max_amplitude=0.6, with_noise=False).example()
normalized = original.normalize()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

original.plot(ax=ax1, title=f"Original (mean={original.mean_flux:.1f})")
normalized.plot(ax=ax2, title=f"Normalized (mean={normalized.mean_flux:.1e})")

plt.tight_layout()
plt.show()

print(f"Original: mean={original.mean_flux:.2f}, std={original.std_flux:.2f}")
print(f"Normalized: mean={normalized.mean_flux:.2e}, std={normalized.std_flux:.2f}")
