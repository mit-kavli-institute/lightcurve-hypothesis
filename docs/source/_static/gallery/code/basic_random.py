from hypothesis_lightcurves.generators import lightcurves

# Generate a basic random lightcurve
lc = lightcurves().example()
fig, ax = lc.plot(title="Basic Random Lightcurve")
