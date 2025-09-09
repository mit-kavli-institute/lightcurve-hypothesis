Examples
========

This section provides detailed examples of using ``hypothesis_lightcurves`` for various testing scenarios.

Testing a Period Detection Algorithm
-------------------------------------

Let's test a simple period detection algorithm using property-based testing:

.. code-block:: python

    import numpy as np
    from hypothesis import given, assume, settings
    from hypothesis_lightcurves.generators import periodic_lightcurves
    from hypothesis_lightcurves.utils import calculate_periodogram

    def detect_period_simple(lightcurve):
        """A simple period detection using periodogram."""
        periods = np.linspace(0.1, 20.0, 2000)
        test_periods, power = calculate_periodogram(lightcurve, periods)
        return test_periods[np.argmax(power)]

    @given(lc=periodic_lightcurves(
        min_period=0.5,
        max_period=10.0,
        min_amplitude=0.1,
        max_amplitude=10.0,
        with_noise=True
    ))
    @settings(max_examples=100)
    def test_period_detection_accuracy(lc):
        """Test that period detection is accurate within tolerance."""
        # Skip if too few points per period
        points_per_period = lc.n_points / (lc.duration / lc.metadata['period'])
        assume(points_per_period > 10)

        detected = detect_period_simple(lc)
        true_period = lc.metadata['period']

        # Should detect within 5% for well-sampled lightcurves
        relative_error = abs(detected - true_period) / true_period
        assert relative_error < 0.05, f"Detected {detected:.3f}, true {true_period:.3f}"

Testing Lightcurve Normalization
---------------------------------

Test that normalization works correctly for various input conditions:

.. code-block:: python

    from hypothesis import given, strategies as st
    from hypothesis_lightcurves.generators import lightcurves
    import numpy as np

    @given(
        lc=lightcurves(
            min_flux=st.floats(min_value=-1e6, max_value=1e6),
            max_flux=st.floats(min_value=-1e6, max_value=1e6),
            with_errors=True
        )
    )
    def test_normalization_properties(lc):
        """Test mathematical properties of normalization."""
        # Skip constant lightcurves
        if np.allclose(lc.flux, lc.flux[0]):
            return

        normalized = lc.normalize()

        # Mean should be approximately zero
        assert abs(normalized.mean_flux) < 1e-10

        # Standard deviation should be approximately one
        assert abs(normalized.std_flux - 1.0) < 1e-10

        # Errors should be scaled by the same factor
        if lc.flux_err is not None:
            scale_factor = lc.std_flux
            expected_err = lc.flux_err / scale_factor
            np.testing.assert_allclose(normalized.flux_err, expected_err)

        # Should preserve time array
        np.testing.assert_array_equal(normalized.time, lc.time)

Testing Transient Detection
----------------------------

Test algorithms that identify and characterize transient events:

.. code-block:: python

    from hypothesis import given, strategies as st
    from hypothesis_lightcurves.generators import transient_lightcurves
    import numpy as np

    def detect_transient(lightcurve, threshold=3.0):
        """Detect transient events above threshold sigma."""
        baseline = np.median(lightcurve.flux)
        noise = np.std(lightcurve.flux[lightcurve.flux < baseline * 1.5])

        # Points significantly above baseline
        transient_mask = lightcurve.flux > baseline + threshold * noise

        if not np.any(transient_mask):
            return None

        # Find the peak
        transient_indices = np.where(transient_mask)[0]
        peak_idx = transient_indices[np.argmax(lightcurve.flux[transient_mask])]

        return {
            'peak_time': lightcurve.time[peak_idx],
            'peak_flux': lightcurve.flux[peak_idx],
            'baseline': baseline,
        }

    @given(lc=transient_lightcurves(
        min_peak_time=10.0,
        max_peak_time=50.0,
        min_rise_time=0.5,
        max_rise_time=5.0,
        min_decay_time=2.0,
        max_decay_time=20.0
    ))
    def test_transient_detection(lc):
        """Test that we can detect and characterize transients."""
        result = detect_transient(lc)

        assert result is not None, "Should detect the transient"

        # Peak time should be close to true value
        true_peak = lc.metadata['peak_time']
        assert abs(result['peak_time'] - true_peak) < 2.0

        # Peak flux should be the maximum
        assert result['peak_flux'] == np.max(lc.flux)

Testing Binning Operations
---------------------------

Test that binning preserves signal properties:

.. code-block:: python

    from hypothesis import given, strategies as st
    from hypothesis_lightcurves.generators import periodic_lightcurves
    from hypothesis_lightcurves.utils import bin_lightcurve
    import numpy as np

    @given(
        lc=periodic_lightcurves(min_points=200, max_points=1000),
        bin_factor=st.floats(min_value=0.01, max_value=0.1)
    )
    def test_binning_preserves_period(lc, bin_factor):
        """Test that binning preserves periodic signals."""
        # Calculate bin size based on duration
        bin_size = lc.duration * bin_factor

        # Skip if bin size is too large
        if bin_size > lc.metadata['period'] / 4:
            return

        binned = bin_lightcurve(lc, bin_size)

        # Should have fewer points
        assert binned.n_points < lc.n_points

        # Should preserve the overall flux scale
        assert abs(binned.mean_flux - lc.mean_flux) / lc.mean_flux < 0.1

        # Period should still be detectable
        from hypothesis_lightcurves.utils import calculate_periodogram
        periods = np.linspace(0.5, 20.0, 500)
        _, power_original = calculate_periodogram(lc, periods)
        _, power_binned = calculate_periodogram(binned, periods)

        # Peak should be at similar location
        peak_original = periods[np.argmax(power_original)]
        peak_binned = periods[np.argmax(power_binned)]
        assert abs(peak_original - peak_binned) / peak_original < 0.2

Testing with Multiple Modifications
------------------------------------

Combine multiple operations to test robustness:

.. code-block:: python

    from hypothesis import given, strategies as st
    from hypothesis_lightcurves.generators import periodic_lightcurves
    from hypothesis_lightcurves.utils import (
        resample_lightcurve,
        add_gaps,
        bin_lightcurve
    )

    @given(
        lc=periodic_lightcurves(min_points=500, max_points=1000),
        resample_factor=st.floats(min_value=0.1, max_value=0.5),
        n_gaps=st.integers(min_value=0, max_value=3),
        gap_fraction=st.floats(min_value=0.05, max_value=0.2),
        do_binning=st.booleans()
    )
    def test_combined_operations(lc, resample_factor, n_gaps, gap_fraction, do_binning):
        """Test that lightcurve operations can be combined."""
        original_period = lc.metadata['period']

        # Apply various operations
        processed = lc.copy()

        # Resample
        new_points = int(lc.n_points * resample_factor)
        processed = resample_lightcurve(processed, new_points)

        # Add gaps
        if n_gaps > 0:
            processed = add_gaps(processed, n_gaps, gap_fraction)

        # Bin if requested
        if do_binning:
            bin_size = processed.duration / 50  # Target ~50 bins
            processed = bin_lightcurve(processed, bin_size)

        # Basic sanity checks
        assert processed.n_points > 10, "Should have enough points left"
        assert processed.duration > 0, "Should have positive duration"
        assert len(processed.time) == len(processed.flux), "Arrays should match"

        # Check modifications are tracked
        assert "normalized" in lc.normalize().modifications

Testing Edge Cases
------------------

Test handling of edge cases and unusual inputs:

.. code-block:: python

    from hypothesis import given, strategies as st
    from hypothesis_lightcurves.generators import lightcurves
    import numpy as np

    @given(lc=lightcurves(
        min_points=10,
        max_points=20,
        allow_nan=False,
        allow_inf=False
    ))
    def test_edge_cases(lc):
        """Test handling of various edge cases."""
        # Test with all same flux values
        constant_lc = Lightcurve(
            time=lc.time,
            flux=np.full_like(lc.flux, 100.0)
        )
        normalized = constant_lc.normalize()
        assert np.all(normalized.flux == 0)  # Should just subtract mean

        # Test with single point
        single = Lightcurve(
            time=np.array([1.0]),
            flux=np.array([100.0])
        )
        assert single.duration == 0
        assert single.n_points == 1

        # Test copy independence
        copy = lc.copy()
        copy.flux[0] = -999
        assert lc.flux[0] != -999  # Original unchanged
