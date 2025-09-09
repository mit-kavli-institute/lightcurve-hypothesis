from hypothesis_lightcurves.generators import transient_lightcurves

# Generate a supernova-like transient
lc = transient_lightcurves(
    min_points=200,
    max_points=200,
    min_peak_time=45.0,
    max_peak_time=55.0,
    min_rise_time=8.0,
    max_rise_time=12.0,
    min_decay_time=25.0,
    max_decay_time=35.0,
).example()
fig, ax = lc.plot(title="Supernova-like Transient")
