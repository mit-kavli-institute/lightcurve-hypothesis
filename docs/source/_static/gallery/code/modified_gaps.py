from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.modifiers import add_gaps

# Start with a periodic signal
base_lc = periodic_lightcurves(
    min_points=300,
    max_points=300,
    min_period=1.4,
    max_period=1.6,
    min_amplitude=0.14,
    max_amplitude=0.16,
    with_noise=False,
).example()

# Add data gaps
lc_with_gaps = add_gaps(base_lc, n_gaps=3, gap_fraction=0.1)
fig, ax = lc_with_gaps.plot(title="Lightcurve with Gaps")
