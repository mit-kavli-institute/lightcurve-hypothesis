Examples
========

This page provides detailed examples of using hypothesis-lightcurves for various testing scenarios.

Testing a Period-Finding Algorithm
-----------------------------------

Test that your period-finding algorithm works across a range of parameters:

.. code-block:: python

   from hypothesis import given, strategies as st, assume
   from hypothesis_lightcurves.generators import periodic_lightcurves
   import numpy as np

   @given(
       period=st.floats(min_value=0.1, max_value=100.0),
       amplitude=st.floats(min_value=0.01, max_value=1.0),
       n_points=st.integers(min_value=50, max_value=1000),
       noise_level=st.floats(min_value=0.0, max_value=0.1)
   )
   def test_period_recovery(period, amplitude, n_points, noise_level):
       """Test period recovery under various conditions."""
       # Skip cases where noise overwhelms signal
       assume(amplitude > 2 * noise_level)

       # Generate lightcurve
       lc = periodic_lightcurves(
           period=period,
           amplitude=amplitude,
           n_points=n_points,
           include_noise=True,
           noise_level=noise_level
       ).example()

       # Find period
       recovered_period = find_period_lomb_scargle(lc.time, lc.flux)

       # Check recovery (allow 1% error)
       relative_error = abs(recovered_period - period) / period
       assert relative_error < 0.01, f"Expected {period}, got {recovered_period}"

Testing Outlier Detection
-------------------------

Ensure your outlier detection works with various contamination levels:

.. code-block:: python

   from hypothesis_lightcurves.generators import baseline_lightcurves
   from hypothesis_lightcurves.modifiers import add_outliers

   @given(
       lc=baseline_lightcurves(baseline_type="flat"),
       outlier_fraction=st.floats(min_value=0.01, max_value=0.2),
       outlier_scale=st.floats(min_value=3.0, max_value=10.0)
   )
   def test_outlier_detection(lc, outlier_fraction, outlier_scale):
       """Test outlier detection with known contamination."""
       # Add outliers and track indices
       n_outliers = int(outlier_fraction * lc.n_points)
       outlier_indices = np.random.choice(lc.n_points, n_outliers, replace=False)

       lc_with_outliers = add_outliers(
           lc,
           fraction=outlier_fraction,
           outlier_scale=outlier_scale
       )

       # Detect outliers
       detected = detect_outliers_mad(
           lc_with_outliers.time,
           lc_with_outliers.flux
       )

       # Check detection performance
       true_positives = len(set(detected) & set(outlier_indices))
       precision = true_positives / len(detected) if detected else 0
       recall = true_positives / n_outliers if n_outliers > 0 else 1

       # Should achieve reasonable performance
       assert precision > 0.7, f"Precision too low: {precision}"
       assert recall > 0.7, f"Recall too low: {recall}"

Testing Gap Handling
--------------------

Test that your algorithm handles data gaps correctly:

.. code-block:: python

   from hypothesis_lightcurves.modifiers import add_gaps

   @given(
       lc=periodic_lightcurves(),
       n_gaps=st.integers(min_value=1, max_value=5),
       gap_fraction=st.floats(min_value=0.05, max_value=0.3)
   )
   def test_gap_interpolation(lc, n_gaps, gap_fraction):
       """Test interpolation across gaps."""
       # Add gaps
       lc_with_gaps = add_gaps(lc, n_gaps=n_gaps, gap_fraction=gap_fraction)

       # Interpolate
       interpolated = interpolate_gaps(
           lc_with_gaps.time,
           lc_with_gaps.flux,
           method='cubic'
       )

       # Check properties
       assert len(interpolated.time) >= len(lc_with_gaps.time)
       assert np.all(np.isfinite(interpolated.flux))
       assert np.all(np.diff(interpolated.time) > 0)

Testing Transient Detection
---------------------------

Test transient detection with various event types:

.. code-block:: python

   from hypothesis_lightcurves.generators import transient_lightcurves
   from hypothesis_lightcurves.modifiers import add_noise

   @given(
       peak_time=st.floats(min_value=20, max_value=80),
       rise_time=st.floats(min_value=1, max_value=10),
       decay_time=st.floats(min_value=5, max_value=50),
       peak_flux=st.floats(min_value=100, max_value=1000)
   )
   def test_supernova_detection(peak_time, rise_time, decay_time, peak_flux):
       """Test supernova detection and characterization."""
       # Generate transient
       lc = transient_lightcurves(
           transient_type="supernova",
           peak_time=peak_time,
           rise_time=rise_time,
           decay_time=decay_time,
           peak_flux=peak_flux
       ).example()

       # Add realistic noise
       lc = add_noise(lc, noise_level=0.01 * peak_flux)

       # Detect and characterize
       detection = detect_transient(lc.time, lc.flux)

       assert detection is not None
       assert abs(detection['peak_time'] - peak_time) < 1.0
       assert abs(detection['peak_flux'] - peak_flux) / peak_flux < 0.1

Testing Multi-band Observations
--------------------------------

Test algorithms that process multi-band data:

.. code-block:: python

   @given(
       g_band=lightcurves(),
       r_band=lightcurves(),
       i_band=lightcurves()
   )
   def test_multiband_analysis(g_band, r_band, i_band):
       """Test multi-band lightcurve analysis."""
       # Ensure same time sampling
       common_time = g_band.time
       r_band.time = common_time
       i_band.time = common_time

       # Compute colors
       g_r = g_band.flux - r_band.flux
       r_i = r_band.flux - i_band.flux

       # Classify based on colors
       classification = classify_by_color(g_r, r_i)

       assert classification in ['star', 'galaxy', 'quasar', 'unknown']

Testing Statistical Properties
-------------------------------

Ensure statistical measures are robust:

.. code-block:: python

   from hypothesis_lightcurves.utils import calculate_statistics

   @given(lc=lightcurves())
   def test_statistical_measures(lc):
       """Test statistical measure calculation."""
       stats = calculate_statistics(lc)

       # Basic checks
       assert np.isfinite(stats['mean'])
       assert np.isfinite(stats['std'])
       assert stats['std'] >= 0

       # Consistency checks
       assert stats['min'] <= stats['mean'] <= stats['max']
       assert stats['n_points'] == lc.n_points

       # Percentile checks
       assert stats['min'] <= stats['percentile_25']
       assert stats['percentile_25'] <= stats['median']
       assert stats['median'] <= stats['percentile_75']
       assert stats['percentile_75'] <= stats['max']

Complex Composite Testing
--------------------------

Test complete pipelines with realistic complications:

.. code-block:: python

   from hypothesis_lightcurves.generators import modified_lightcurves

   # Define a complex realistic scenario
   realistic_observation = modified_lightcurves(
       base_strategy=periodic_lightcurves(
           period=st.floats(0.5, 2.0),
           amplitude=st.floats(0.05, 0.2),
           n_points=st.integers(200, 500)
       ),
       modifications=[
           ("noise", {"noise_level": st.floats(0.001, 0.01)}),
           ("gaps", {
               "n_gaps": st.integers(2, 5),
               "gap_fraction": st.floats(0.05, 0.15)
           }),
           ("outliers", {
               "fraction": st.floats(0.01, 0.05),
               "outlier_scale": st.floats(3, 7)
           }),
           ("trend", {
               "trend_type": st.sampled_from(["linear", "quadratic"]),
               "trend_amplitude": st.floats(0.0, 0.1)
           })
       ]
   )

   @given(lc=realistic_observation)
   def test_complete_pipeline(lc):
       """Test complete analysis pipeline with realistic data."""
       # Preprocessing
       lc_clean = remove_outliers(lc)
       lc_detrended = remove_trend(lc_clean)
       lc_filled = interpolate_gaps(lc_detrended)

       # Analysis
       period = find_period(lc_filled)
       amplitude = measure_amplitude(lc_filled, period)
       classification = classify_variable(lc_filled, period, amplitude)

       # Validate results
       assert period > 0
       assert amplitude > 0
       assert classification is not None

       # Check that modifications are tracked
       assert "outliers_removed" in lc_clean.modifications
       assert "trend_removed" in lc_detrended.modifications
       assert "gaps_interpolated" in lc_filled.modifications

Performance Testing
-------------------

Test that your algorithms scale well:

.. code-block:: python

   import time

   @given(
       n_points=st.sampled_from([100, 1000, 10000, 100000])
   )
   def test_algorithm_scaling(n_points):
       """Test algorithm performance scaling."""
       lc = lightcurves(n_points=n_points).example()

       start = time.time()
       result = process_lightcurve(lc)
       elapsed = time.time() - start

       # Should scale roughly linearly
       expected_time = n_points * 1e-5  # 10 microseconds per point
       assert elapsed < expected_time * 2, f"Too slow for {n_points} points"

Testing Error Propagation
-------------------------

Ensure errors are properly propagated:

.. code-block:: python

   @given(
       lc=lightcurves(include_errors=True),
       operation=st.sampled_from(['normalize', 'bin', 'smooth'])
   )
   def test_error_propagation(lc, operation):
       """Test that errors are properly propagated."""
       if operation == 'normalize':
           result = lc.normalize()
       elif operation == 'bin':
           result = bin_lightcurve(lc, bin_size=10)
       else:  # smooth
           result = smooth_lightcurve(lc, window=5)

       # Errors should still be present and valid
       assert result.flux_err is not None
       assert np.all(result.flux_err > 0)
       assert np.all(np.isfinite(result.flux_err))

Custom Strategy Composition
---------------------------

Create domain-specific test strategies:

.. code-block:: python

   from hypothesis.strategies import composite

   @composite
   def eclipsing_binary_lightcurves(draw):
       """Generate eclipsing binary lightcurves."""
       # Draw parameters
       period = draw(st.floats(0.5, 10.0))
       primary_depth = draw(st.floats(0.1, 0.5))
       secondary_depth = draw(st.floats(0.05, primary_depth))
       eclipse_width = draw(st.floats(0.05, 0.2))

       # Generate base
       lc = draw(baseline_lightcurves(n_points=1000))

       # Add eclipses
       phase = (lc.time % period) / period

       # Primary eclipse at phase 0
       primary_mask = np.abs(phase) < eclipse_width / 2
       lc.flux[primary_mask] *= (1 - primary_depth)

       # Secondary eclipse at phase 0.5
       secondary_mask = np.abs(phase - 0.5) < eclipse_width / 2
       lc.flux[secondary_mask] *= (1 - secondary_depth)

       # Store parameters
       lc.metadata['period'] = period
       lc.metadata['primary_depth'] = primary_depth
       lc.metadata['secondary_depth'] = secondary_depth

       return lc

   @given(lc=eclipsing_binary_lightcurves())
   def test_eclipse_detection(lc):
       """Test eclipsing binary detection."""
       result = detect_eclipses(lc)
       assert result['is_eclipsing'] == True
       assert len(result['eclipse_times']) >= 2
