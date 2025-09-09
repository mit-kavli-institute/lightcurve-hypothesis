from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.modifiers import add_trend

# Start with a periodic signal
base_lc = periodic_lightcurves(
    min_points=250,
    max_points=250,
    min_period=0.7,
    max_period=0.9,
    min_amplitude=0.09,
    max_amplitude=0.11,
    with_noise=False,
).example()

# Add quadratic trend
lc_with_trend = add_trend(base_lc, trend_type="quadratic", coefficient=0.2)
fig, ax = lc_with_trend.plot(title="Lightcurve with Trend")
