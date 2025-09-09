Installation
============

Requirements
------------

* Python 3.11 or higher
* NumPy 1.24 or higher
* Hypothesis 6.0 or higher

Installing from Source
----------------------

Clone the repository and install in development mode::

    git clone https://github.com/williamfong/lightcurve-hypothesis.git
    cd lightcurve-hypothesis
    pip install -e .

Development Installation
------------------------

To install with all development dependencies::

    pip install -e ".[dev]"

This includes:

* **pytest** - Test runner
* **pytest-cov** - Coverage reporting
* **black** - Code formatter
* **ruff** - Linter
* **mypy** - Type checker
* **pre-commit** - Git hooks for code quality

After installing development dependencies, set up pre-commit hooks::

    pre-commit install

Documentation Dependencies
--------------------------

To build the documentation locally::

    pip install -e ".[docs]"

This includes:

* **sphinx** - Documentation generator
* **sphinx-rtd-theme** - Read the Docs theme
* **numpydoc** - NumPy-style docstring support
* **sphinx-autodoc-typehints** - Type hint support
* **myst-parser** - Markdown support

Verifying Installation
----------------------

After installation, verify everything is working::

    python -c "import hypothesis_lightcurves; print(hypothesis_lightcurves.__version__)"

This should print the current version number (e.g., ``0.1.0``).
