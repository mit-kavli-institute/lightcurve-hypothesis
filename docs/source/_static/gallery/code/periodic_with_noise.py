from hypothesis_lightcurves.generators import periodic_lightcurves

# Generate a periodic lightcurve with noise
lc = periodic_lightcurves(
    min_points=300,
    max_points=300,
    min_period=0.9,
    max_period=1.1,
    min_amplitude=0.15,
    max_amplitude=0.25,
    with_noise=True,
).example()
fig, ax = lc.plot(title="Periodic with Noise")
