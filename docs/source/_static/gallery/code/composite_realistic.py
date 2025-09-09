from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.modifiers import add_gaps, add_noise, add_outliers

# Generate realistic observational data with multiple effects
base_lc = periodic_lightcurves(
    min_points=400,
    max_points=400,
    min_period=1.1,
    max_period=1.3,
    min_amplitude=0.07,
    max_amplitude=0.09,
    with_noise=False,
).example()

# Apply modifications sequentially
lc = add_noise(base_lc, level=0.01)
lc = add_gaps(lc, n_gaps=2, gap_fraction=0.08)
lc = add_outliers(lc, fraction=0.02, amplitude=4)

fig, ax = lc.plot(title="Realistic Observational Data")
