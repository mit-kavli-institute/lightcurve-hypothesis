from hypothesis_lightcurves.generators import lightcurves

# Generate a lightcurve with error bars
lc = lightcurves(with_errors=True).example()
fig, ax = lc.plot(title="Lightcurve with Errors")
