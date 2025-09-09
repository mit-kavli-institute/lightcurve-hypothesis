from hypothesis_lightcurves.generators import transient_lightcurves

# Generate a stellar flare (short timescales)
lc = transient_lightcurves(
    min_points=150,
    max_points=150,
    min_peak_time=20.0,
    max_peak_time=30.0,
    min_rise_time=0.3,
    max_rise_time=0.7,
    min_decay_time=1.5,
    max_decay_time=2.5,
).example()
fig, ax = lc.plot(title="Stellar Flare")
