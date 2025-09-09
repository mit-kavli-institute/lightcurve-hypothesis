Gallery
=======

Visual examples demonstrating the capabilities of ``hypothesis_lightcurves``.

The gallery below showcases various types of lightcurves that can be generated,
transformations that can be applied, and statistical analyses that can be performed.
These examples serve both as documentation and as visual validation of the generators.

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/index

Example Categories
------------------

**Basic Lightcurves**
   Random lightcurves with various sampling rates, flux ranges, and noise levels.

**Periodic Signals**
   Sinusoidal patterns useful for testing period detection algorithms.

**Transient Events**
   Rise and decay profiles modeling supernovae, novae, and stellar flares.

**Transformations**
   Resampling, binning, gap addition, and normalization operations.

**Ensemble Statistics**
   Statistical analysis across multiple generated lightcurves.

.. raw:: html

   <div style="clear: both"></div>

Browse Examples
---------------

.. include:: auto_examples/index.rst
   :start-after: .. _gallery_examples:

Interactive Features
--------------------

Each example includes:

* **Source code** - Complete Python code that you can copy and modify
* **Visualizations** - High-quality plots showing the results
* **Explanations** - Detailed descriptions of what each example demonstrates
* **Downloads** - Jupyter notebook and Python script versions

Using the Examples
------------------

To run these examples locally:

1. Install the package with visualization dependencies::

    pip install -e ".[docs]"

2. Navigate to any example script::

    cd docs/source/examples
    python plot_basic_lightcurves.py

3. Or run them in a Jupyter notebook::

    jupyter notebook plot_basic_lightcurves.ipynb

Extending the Gallery
---------------------

To add new examples:

1. Create a new Python file in ``docs/source/examples/`` starting with ``plot_``
2. Add docstring with title and description using the gallery format
3. Include matplotlib plots to visualize your example
4. Rebuild the documentation to include your example

The visualization system is designed to be extensible. New generator types can
register custom visualizers that will be automatically used in the gallery.

.. seealso::

   :doc:`api`
      Complete API documentation for all modules

   :doc:`examples`
      Code examples for testing scenarios

   :doc:`quickstart`
      Getting started guide
