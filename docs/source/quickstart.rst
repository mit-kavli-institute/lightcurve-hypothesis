Quick Start Guide
=================

This guide will help you get started with ``hypothesis_lightcurves`` for property-based testing of astronomical data processing code.

Basic Usage
-----------

Creating a Simple Lightcurve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to create a lightcurve is using the ``Lightcurve`` data model:

.. code-block:: python

    import numpy as np
    from hypothesis_lightcurves.models import Lightcurve

    # Create time and flux arrays
    time = np.linspace(0, 10, 100)
    flux = 100 + 5 * np.sin(2 * np.pi * time / 2.5)

    # Create the lightcurve
    lc = Lightcurve(time=time, flux=flux)

    # Access properties
    print(f"Number of points: {lc.n_points}")
    print(f"Duration: {lc.duration:.2f}")
    print(f"Mean flux: {lc.mean_flux:.2f}")

Property-Based Testing
----------------------

Using Hypothesis Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main power of this package comes from using Hypothesis strategies to generate test cases:

.. code-block:: python

    from hypothesis import given, settings
    from hypothesis_lightcurves.generators import lightcurves

    @given(lc=lightcurves(min_points=50, max_points=200))
    @settings(max_examples=100)
    def test_normalization_preserves_shape(lc):
        """Test that normalization preserves the lightcurve shape."""
        normalized = lc.normalize()

        # Check that normalization worked
        assert abs(normalized.mean_flux) < 1e-10
        assert abs(normalized.std_flux - 1.0) < 1e-10

        # Check that shape is preserved
        assert normalized.n_points == lc.n_points
        assert normalized.duration == lc.duration

Testing Periodic Signals
~~~~~~~~~~~~~~~~~~~~~~~~

Test algorithms that detect periodic signals:

.. code-block:: python

    from hypothesis import given
    from hypothesis_lightcurves.generators import periodic_lightcurves
    from hypothesis_lightcurves.utils import calculate_periodogram

    @given(lc=periodic_lightcurves(min_period=1.0, max_period=5.0))
    def test_period_detection(lc):
        """Test that we can recover the input period."""
        true_period = lc.metadata['period']

        # Calculate periodogram
        periods = np.linspace(0.5, 10.0, 1000)
        test_periods, power = calculate_periodogram(lc, periods)

        # Find the peak
        detected_period = test_periods[np.argmax(power)]

        # Should be within 10% of true period
        assert abs(detected_period - true_period) / true_period < 0.1

Testing Transient Events
~~~~~~~~~~~~~~~~~~~~~~~~

Test code that processes transient events:

.. code-block:: python

    from hypothesis import given
    from hypothesis_lightcurves.generators import transient_lightcurves

    @given(lc=transient_lightcurves())
    def test_transient_peak_detection(lc):
        """Test that we can find the peak of a transient."""
        peak_idx = np.argmax(lc.flux)
        detected_peak_time = lc.time[peak_idx]
        true_peak_time = lc.metadata['peak_time']

        # Peak detection should be reasonably accurate
        assert abs(detected_peak_time - true_peak_time) < 1.0

Lightcurve Manipulation
-----------------------

Resampling
~~~~~~~~~~

Change the sampling rate of a lightcurve:

.. code-block:: python

    from hypothesis_lightcurves.utils import resample_lightcurve

    # Reduce to 50 points
    resampled = resample_lightcurve(lc, n_points=50)
    assert resampled.n_points == 50

Adding Gaps
~~~~~~~~~~~

Simulate observational gaps:

.. code-block:: python

    from hypothesis_lightcurves.utils import add_gaps

    # Remove 20% of data in 3 gaps
    lc_with_gaps = add_gaps(lc, n_gaps=3, gap_fraction=0.2)

    # Verify approximately 80% of points remain
    remaining = lc_with_gaps.n_points / lc.n_points
    assert 0.75 < remaining < 0.85

Binning
~~~~~~~

Reduce noise by binning:

.. code-block:: python

    from hypothesis_lightcurves.utils import bin_lightcurve

    # Bin to 0.5 time units
    binned = bin_lightcurve(lc, bin_size=0.5)

    # Binned lightcurve has fewer points
    assert binned.n_points < lc.n_points

Advanced Testing Patterns
-------------------------

Combining Multiple Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test that your code handles various types of lightcurves:

.. code-block:: python

    from hypothesis import given, strategies as st
    from hypothesis_lightcurves.generators import (
        lightcurves,
        periodic_lightcurves,
        transient_lightcurves
    )

    # Create a strategy that generates any type of lightcurve
    any_lightcurve = st.one_of([
        lightcurves(),
        periodic_lightcurves(),
        transient_lightcurves(),
    ])

    @given(lc=any_lightcurve)
    def test_generic_processing(lc):
        """Test that processing works for any lightcurve type."""
        processed = my_processing_function(lc)
        assert processed.is_valid()

Testing with Realistic Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add noise and gaps to make tests more realistic:

.. code-block:: python

    from hypothesis import given, strategies as st
    from hypothesis_lightcurves.generators import periodic_lightcurves
    from hypothesis_lightcurves.utils import add_gaps

    @given(
        lc=periodic_lightcurves(with_noise=True),
        n_gaps=st.integers(0, 5),
        gap_frac=st.floats(0.0, 0.3)
    )
    def test_robust_period_detection(lc, n_gaps, gap_frac):
        """Test period detection with gaps and noise."""
        # Add gaps to simulate real observations
        if n_gaps > 0:
            lc = add_gaps(lc, n_gaps=n_gaps, gap_fraction=gap_frac)

        # Your period detection should still work
        detected_period = detect_period(lc)
        true_period = lc.metadata['period']

        # Allow more tolerance with gaps
        tolerance = 0.2 if n_gaps > 0 else 0.1
        assert abs(detected_period - true_period) / true_period < tolerance
