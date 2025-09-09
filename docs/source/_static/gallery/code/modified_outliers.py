from hypothesis_lightcurves.generators import baseline_lightcurves
from hypothesis_lightcurves.modifiers import add_outliers

# Start with a smooth baseline
base_lc = baseline_lightcurves(baseline_type="smooth", n_points=200).example()

# Add outliers
lc_with_outliers = add_outliers(base_lc, fraction=0.05, amplitude=5.0)
fig, ax = lc_with_outliers.plot(title="Lightcurve with Outliers")
