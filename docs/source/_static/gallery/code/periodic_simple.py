from hypothesis_lightcurves.generators import periodic_lightcurves

# Generate a simple periodic lightcurve
lc = periodic_lightcurves(
    min_points=200,
    max_points=200,
    min_period=2.4,
    max_period=2.6,
    min_amplitude=0.09,
    max_amplitude=0.11,
    with_noise=False,
).example()
fig, ax = lc.plot(title="Simple Periodic Signal")
