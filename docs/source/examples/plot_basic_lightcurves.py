"""
Basic Lightcurve Examples
==========================

This example demonstrates the generation and visualization of basic random lightcurves
using the hypothesis_lightcurves package.
"""

import matplotlib.pyplot as plt
import numpy as np
from hypothesis_lightcurves.generators import lightcurves
from hypothesis_lightcurves.visualization import (
    create_gallery_plot,
    plot_lightcurve,
    plot_with_annotations,
)

# %%
# Generate a single random lightcurve
# ------------------------------------
# Let's start by generating and plotting a single random lightcurve.

np.random.seed(42)
lc = lightcurves(min_points=100, max_points=200).example()

fig, ax = plt.subplots(figsize=(10, 6))
plot_lightcurve(lc, ax=ax, title="Random Lightcurve Example", color="navy")
plt.show()

# %%
# Generate lightcurves with different parameters
# -----------------------------------------------
# We can control various parameters when generating lightcurves.

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sparse lightcurve
lc_sparse = lightcurves(min_points=20, max_points=30).example()
plot_lightcurve(lc_sparse, ax=axes[0, 0], title="Sparse Sampling", marker="o", markersize=6)

# Dense lightcurve
lc_dense = lightcurves(min_points=500, max_points=1000).example()
plot_lightcurve(lc_dense, ax=axes[0, 1], title="Dense Sampling", marker="", linestyle="-")

# High flux range
lc_bright = lightcurves(min_flux=1000, max_flux=10000).example()
plot_lightcurve(lc_bright, ax=axes[1, 0], title="High Flux Range", color="orange")

# With errors
lc_errors = lightcurves(with_errors=True).example()
plot_lightcurve(lc_errors, ax=axes[1, 1], title="With Measurement Errors", color="green")

plt.tight_layout()
plt.show()

# %%
# Lightcurve with annotations
# ----------------------------
# We can add statistical annotations to better understand the lightcurve properties.

lc_annotated = lightcurves(min_points=150, max_points=200, with_errors=True).example()

fig, ax = plt.subplots(figsize=(12, 7))
plot_with_annotations(lc_annotated, ax=ax, annotate_statistics=True, color="purple", alpha=0.7)
ax.set_title("Annotated Lightcurve with Statistics", fontsize=14)
plt.show()

# %%
# Gallery of random lightcurves
# ------------------------------
# Let's create a gallery showing the diversity of generated lightcurves.

fig = create_gallery_plot(
    n_examples=9,
    generator_func=lightcurves,
    title="Gallery of Random Lightcurves",
    figsize=(15, 10),
    seed=123,
    min_points=50,
    max_points=300,
)
plt.show()

# %%
# Comparing different flux distributions
# ---------------------------------------
# We can generate lightcurves with different flux characteristics.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Low baseline
lc_low = lightcurves(min_flux=0, max_flux=100).example()
plot_lightcurve(lc_low, ax=axes[0], title="Low Flux (0-100)", color="blue")
axes[0].set_ylim(-10, 110)

# Medium baseline
lc_med = lightcurves(min_flux=900, max_flux=1100).example()
plot_lightcurve(lc_med, ax=axes[1], title="Medium Flux (900-1100)", color="green")
axes[1].set_ylim(850, 1150)

# High baseline
lc_high = lightcurves(min_flux=9000, max_flux=11000).example()
plot_lightcurve(lc_high, ax=axes[2], title="High Flux (9000-11000)", color="red")
axes[2].set_ylim(8500, 11500)

plt.suptitle("Lightcurves with Different Flux Baselines", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
