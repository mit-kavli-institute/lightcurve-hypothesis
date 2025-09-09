from hypothesis_lightcurves.generators import baseline_lightcurves

# Generate a flat baseline lightcurve
lc = baseline_lightcurves(baseline_type="flat", n_points=200).example()
fig, ax = lc.plot(title="Flat Baseline")
