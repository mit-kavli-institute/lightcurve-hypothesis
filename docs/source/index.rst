.. hypothesis_lightcurves documentation master file

hypothesis_lightcurves
======================

A Python package for generating synthetic astronomical lightcurves using property-based testing with Hypothesis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   gallery
   api
   examples
   contributing

Overview
--------

``hypothesis_lightcurves`` provides tools for generating and manipulating synthetic astronomical lightcurves for testing purposes. It leverages the `Hypothesis <https://hypothesis.readthedocs.io/>`_ property-based testing framework to create diverse test cases for astronomical data processing pipelines.

Key Features
------------

* **Property-based testing strategies** for generating random lightcurves
* **Specialized generators** for periodic and transient phenomena
* **Utility functions** for common lightcurve operations
* **Type-safe** with full type hints using Python 3.11+ syntax
* **Well-documented** with comprehensive NumPy-style docstrings

Installation
------------

Install from source::

    pip install -e .

For development with all optional dependencies::

    pip install -e ".[dev]"

Quick Example
-------------

Generate a random lightcurve for testing:

.. code-block:: python

    from hypothesis import given
    from hypothesis_lightcurves.generators import lightcurves

    @given(lc=lightcurves())
    def test_my_analysis_function(lc):
        result = my_analysis_function(lc)
        assert result.is_valid()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
