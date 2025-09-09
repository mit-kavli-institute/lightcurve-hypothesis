Installation
============

Requirements
------------

hypothesis-lightcurves requires Python 3.11 or later. The main dependencies are:

* `hypothesis <https://hypothesis.readthedocs.io/>`_ >= 6.0 - Property-based testing framework
* `numpy <https://numpy.org/>`_ >= 1.24 - Numerical operations

Installing from PyPI
--------------------

The simplest way to install hypothesis-lightcurves is using pip:

.. code-block:: bash

   pip install hypothesis-lightcurves

Installing from Source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/mit-kavli-institute/lightcurve-hypothesis.git
   cd lightcurve-hypothesis
   pip install -e .

Development Installation
------------------------

If you want to contribute to hypothesis-lightcurves, install with development dependencies:

.. code-block:: bash

   git clone https://github.com/mit-kavli-institute/lightcurve-hypothesis.git
   cd lightcurve-hypothesis
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

This installs additional tools for development:

* **pytest** - Test runner
* **black** - Code formatter
* **ruff** - Linter
* **mypy** - Type checker
* **pre-commit** - Git hooks for code quality
* **nox** - Test automation

Documentation Dependencies
--------------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The documentation will be built in ``docs/build/html/``.

Verifying Installation
----------------------

After installation, you can verify everything is working:

.. code-block:: python

   import hypothesis_lightcurves
   from hypothesis_lightcurves.generators import lightcurves

   # Generate a test lightcurve
   lc = lightcurves().example()
   print(f"Generated lightcurve with {lc.n_points} points")

Running Tests
-------------

To run the test suite:

.. code-block:: bash

   # If installed from source with dev dependencies
   pytest

   # Or using nox for full test matrix
   nox -s tests

Optional Dependencies
---------------------

The package has several optional dependency groups:

* ``[dev]`` - Development tools (testing, linting, formatting)
* ``[docs]`` - Documentation building tools

Install multiple groups:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Troubleshooting
---------------

Common installation issues:

**Python version**: Ensure you have Python 3.11 or later:

.. code-block:: bash

   python --version

**NumPy compatibility**: If you encounter NumPy issues, try upgrading:

.. code-block:: bash

   pip install --upgrade numpy

**Development tools**: If pre-commit hooks fail, update them:

.. code-block:: bash

   pre-commit autoupdate
