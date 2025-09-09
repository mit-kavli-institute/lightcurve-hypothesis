from hypothesis_lightcurves.generators import baseline_lightcurves

# Generate a random walk baseline
lc = baseline_lightcurves(baseline_type="random_walk", n_points=300).example()
fig, ax = lc.plot(title="Random Walk Baseline")
