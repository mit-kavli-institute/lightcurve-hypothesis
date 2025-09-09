Quick Start Guide
=================

This guide will help you get started with hypothesis-lightcurves for testing astronomical data processing code.

Basic Concepts
--------------

hypothesis-lightcurves provides **Hypothesis strategies** for generating test lightcurves. A strategy is a recipe for generating random test data that satisfies certain constraints.

The main components are:

1. **Lightcurve model**: A dataclass representing time series data
2. **Generators**: Hypothesis strategies that create lightcurves
3. **Modifiers**: Functions that add realistic effects to lightcurves

Your First Test
---------------

Let's write a simple property-based test:

.. code-block:: python

   from hypothesis import given
   from hypothesis_lightcurves.generators import lightcurves
   import numpy as np

   @given(lc=lightcurves())
   def test_lightcurve_properties(lc):
       """Test that all lightcurves have valid properties."""
       # Time should be sorted
       assert np.all(np.diff(lc.time) >= 0)

       # Flux should be finite
       assert np.all(np.isfinite(lc.flux))

       # Arrays should have same length
       assert len(lc.time) == len(lc.flux)

       if lc.flux_err is not None:
           assert len(lc.flux_err) == len(lc.flux)

This test will run with many different randomly generated lightcurves, ensuring your assumptions hold for all of them.

Generating Different Types of Lightcurves
------------------------------------------

Periodic Lightcurves
^^^^^^^^^^^^^^^^^^^^

Generate lightcurves with periodic signals:

.. code-block:: python

   from hypothesis_lightcurves.generators import periodic_lightcurves

   @given(lc=periodic_lightcurves(period=1.5, amplitude=0.1))
   def test_period_detection(lc):
       """Test that we can recover known periods."""
       detected_period = my_period_finder(lc.time, lc.flux)
       assert abs(detected_period - 1.5) < 0.01

Transient Events
^^^^^^^^^^^^^^^^

Generate lightcurves with transient events like supernovae:

.. code-block:: python

   from hypothesis_lightcurves.generators import transient_lightcurves

   @given(lc=transient_lightcurves())
   def test_transient_detection(lc):
       """Test transient detection algorithm."""
       is_transient = my_transient_detector(lc.time, lc.flux)
       assert is_transient is True

Using Strategies for Parameters
--------------------------------

Instead of fixed values, use strategies to test ranges:

.. code-block:: python

   from hypothesis import strategies as st
   from hypothesis_lightcurves.generators import baseline_lightcurves

   @given(
       lc=baseline_lightcurves(
           n_points=st.integers(min_value=50, max_value=500),
           baseline_flux=st.floats(min_value=10, max_value=1000),
           baseline_type=st.sampled_from(["flat", "smooth", "random_walk"])
       )
   )
   def test_baseline_estimation(lc):
       """Test baseline estimation with various parameters."""
       estimated = estimate_baseline(lc.time, lc.flux)
       assert estimated is not None

Adding Realistic Effects
------------------------

Use modifiers to add complications:

.. code-block:: python

   from hypothesis_lightcurves.modifiers import add_noise, add_gaps, add_outliers

   @given(lc=lightcurves())
   def test_with_realistic_data(lc):
       """Test with realistic data complications."""
       # Add various effects
       lc = add_noise(lc, noise_level=0.01)
       lc = add_gaps(lc, n_gaps=3, gap_fraction=0.1)
       lc = add_outliers(lc, fraction=0.05)

       # Your pipeline should still work
       result = process_lightcurve(lc)
       assert result is not None

Composed Strategies
-------------------

Create custom strategies by composing existing ones:

.. code-block:: python

   from hypothesis_lightcurves.generators import modified_lightcurves

   # Create a strategy for realistic variable star observations
   realistic_variables = modified_lightcurves(
       base_strategy=periodic_lightcurves(
           period=st.floats(0.1, 10.0),
           amplitude=st.floats(0.01, 0.5)
       ),
       modifications=[
           ("noise", {"noise_level": 0.005}),
           ("gaps", {"n_gaps": st.integers(1, 5)}),
           ("outliers", {"fraction": 0.02})
       ]
   )

   @given(lc=realistic_variables)
   def test_variable_star_pipeline(lc):
       classification = classify_variable(lc)
       assert classification in ["RR Lyrae", "Cepheid", "Eclipse", "Unknown"]

Testing Edge Cases
------------------

Hypothesis will automatically find edge cases, but you can also guide it:

.. code-block:: python

   from hypothesis import assume

   @given(lc=lightcurves())
   def test_short_lightcurves(lc):
       # Focus on short lightcurves
       assume(lc.n_points < 50)

       # Should still handle short data
       result = analyze_lightcurve(lc)
       assert result is not None

Best Practices
--------------

1. **Start simple**: Begin with basic generators and add complexity gradually
2. **Test properties, not examples**: Focus on what should always be true
3. **Use assumptions carefully**: Filter inputs when needed but not excessively
4. **Let Hypothesis find bugs**: Don't over-constrain the input space
5. **Combine strategies**: Build complex test scenarios from simple parts

Debugging Failed Tests
----------------------

When a test fails, Hypothesis will show you the failing example:

.. code-block:: python

   @given(lc=lightcurves())
   def test_mean_flux(lc):
       # This test has a bug!
       assert lc.mean_flux > 0  # Will fail for negative flux

   # Hypothesis will find and minimize a counterexample
   # Falsifying example: lc=Lightcurve(
   #     time=array([0.]),
   #     flux=array([-1.]),
   #     ...
   # )

Hypothesis automatically simplifies failures to minimal examples, making debugging easier.

Next Steps
----------

* Read the :doc:`examples` for more complex scenarios
* Explore the :doc:`api` for all available generators and modifiers
* Learn about `Hypothesis <https://hypothesis.readthedocs.io/>`_ for advanced testing techniques
