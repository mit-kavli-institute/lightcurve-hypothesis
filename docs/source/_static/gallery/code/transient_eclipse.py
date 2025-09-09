from hypothesis_lightcurves.generators import transient_lightcurves

# Generate an eclipse-like dip
lc = transient_lightcurves(
    min_points=200,
    max_points=200,
    min_peak_time=25.0,
    max_peak_time=35.0,
    min_rise_time=1.5,
    max_rise_time=2.5,
    min_decay_time=1.5,
    max_decay_time=2.5,
).example()
# Invert to create a dip
lc.flux = 2 * lc.mean_flux - lc.flux
fig, ax = lc.plot(title="Eclipse-like Dip")
