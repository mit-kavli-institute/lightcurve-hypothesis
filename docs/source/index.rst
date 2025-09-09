.. hypothesis-lightcurves documentation master file

hypothesis-lightcurves
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   quickstart
   examples
   api

**hypothesis-lightcurves** is a Python package for generating synthetic astronomical lightcurves
using `Hypothesis <https://hypothesis.readthedocs.io/>`_ property-based testing strategies.
It provides data models and generators for creating realistic test data for astronomical
time series analysis pipelines.

Key Features
------------

* **Property-based testing**: Generate diverse lightcurves for comprehensive testing
* **Flexible generators**: Create baseline, periodic, and transient lightcurves
* **Composable modifiers**: Add noise, gaps, outliers, and other realistic effects
* **Type-safe**: Full type hints and runtime validation
* **Well-tested**: Extensive test coverage using the same tools we provide

Installation
------------

Install hypothesis-lightcurves using pip:

.. code-block:: bash

   pip install hypothesis-lightcurves

Or install from source:

.. code-block:: bash

   git clone https://github.com/mit-kavli-institute/lightcurve-hypothesis.git
   cd lightcurve-hypothesis
   pip install -e ".[dev]"

Quick Example
-------------

Generate a simple lightcurve for testing:

.. code-block:: python

   from hypothesis import given
   from hypothesis_lightcurves.generators import lightcurves

   @given(lc=lightcurves())
   def test_my_analysis_pipeline(lc):
       """Test that my pipeline handles any valid lightcurve."""
       result = my_pipeline(lc.time, lc.flux, lc.flux_err)
       assert result is not None
       assert not np.any(np.isnan(result))

Create a periodic lightcurve with specific properties:

.. code-block:: python

   from hypothesis import strategies as st
   from hypothesis_lightcurves.generators import periodic_lightcurves

   # Generate periodic signals with periods between 0.5 and 10 days
   periodic_strategy = periodic_lightcurves(
       period=st.floats(min_value=0.5, max_value=10.0),
       amplitude=st.floats(min_value=0.01, max_value=0.1),
       n_points=st.integers(min_value=100, max_value=1000)
   )

Why hypothesis-lightcurves?
----------------------------

Testing astronomical data processing pipelines is challenging because:

1. **Edge cases are common**: Real astronomical data contains gaps, outliers, and noise
2. **Parameter spaces are large**: Periods, amplitudes, and sampling patterns vary widely
3. **Assumptions break**: Pipelines often assume uniform sampling or Gaussian noise

This package helps you:

* **Find bugs faster**: Property-based testing explores edge cases automatically
* **Write better tests**: Focus on properties, not specific examples
* **Improve robustness**: Test with realistic data complications

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
