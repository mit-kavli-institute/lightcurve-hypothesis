from hypothesis_lightcurves.generators import periodic_lightcurves

# Generate a multi-periodic signal by combining two periodic signals
lc1 = periodic_lightcurves(
    min_points=500,
    max_points=500,
    min_period=0.45,
    max_period=0.55,
    min_amplitude=0.08,
    max_amplitude=0.12,
    with_noise=False,
).example()
lc2 = periodic_lightcurves(
    min_points=500,
    max_points=500,
    min_period=1.2,
    max_period=1.4,
    min_amplitude=0.04,
    max_amplitude=0.06,
    with_noise=False,
).example()

# Combine the signals
lc2.time = lc1.time.copy()
lc1.flux = lc1.flux + lc2.flux - lc2.mean_flux
fig, ax = lc1.plot(title="Multi-periodic Signal")
