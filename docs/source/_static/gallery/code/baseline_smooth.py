from hypothesis_lightcurves.generators import baseline_lightcurves

# Generate a smooth baseline
lc = baseline_lightcurves(baseline_type="smooth", n_points=250).example()
fig, ax = lc.plot(title="Smooth Baseline")
