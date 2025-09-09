Contributing
============

We welcome contributions to ``hypothesis_lightcurves``! This guide will help you get started.

Development Setup
-----------------

1. Fork and clone the repository::

    git clone https://github.com/YOUR_USERNAME/lightcurve-hypothesis.git
    cd lightcurve-hypothesis

2. Create a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode with all dependencies::

    pip install -e ".[dev,docs]"

4. Set up pre-commit hooks::

    pre-commit install

Code Style
----------

We use several tools to maintain code quality:

* **black** for code formatting (line length: 100)
* **ruff** for linting
* **mypy** for type checking

Run all checks with::

    black src/ tests/
    ruff check src/ tests/
    mypy src/

Or use pre-commit to run all checks::

    pre-commit run --all-files

Testing
-------

We use pytest for testing. All new features should include tests.

Run tests::

    pytest

Run tests with coverage::

    pytest --cov=hypothesis_lightcurves --cov-report=term-missing

Writing Tests
~~~~~~~~~~~~~

Tests should use property-based testing where appropriate:

.. code-block:: python

    from hypothesis import given
    from hypothesis_lightcurves.generators import lightcurves

    @given(lc=lightcurves())
    def test_my_feature(lc):
        result = my_function(lc)
        assert some_property(result)

Documentation
-------------

Documentation uses Sphinx with NumPy-style docstrings.

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Build the documentation locally::

    cd docs
    make clean
    make html

View the documentation by opening ``docs/build/html/index.html``.

Writing Docstrings
~~~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

    def my_function(param1: float, param2: int) -> float:
        """Brief description of function.

        Longer description if needed, explaining what the
        function does in more detail.

        Parameters
        ----------
        param1 : float
            Description of param1.
        param2 : int
            Description of param2.

        Returns
        -------
        float
            Description of return value.

        Examples
        --------
        >>> result = my_function(1.0, 2)
        >>> print(result)
        3.0
        """

Making a Pull Request
---------------------

1. Create a new branch for your feature::

    git checkout -b my-feature-branch

2. Make your changes and commit them::

    git add .
    git commit -m "Add my new feature"

3. Run all tests and checks::

    pytest
    pre-commit run --all-files

4. Push to your fork::

    git push origin my-feature-branch

5. Open a pull request on GitHub

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

* Include tests for new functionality
* Update documentation as needed
* Follow the existing code style
* Write clear commit messages
* Keep pull requests focused on a single feature/fix

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

Report bugs at https://github.com/williamfong/lightcurve-hypothesis/issues

Include:

* Your operating system and Python version
* Detailed steps to reproduce the bug
* Any error messages or tracebacks

Feature Requests
~~~~~~~~~~~~~~~~

Suggest features at https://github.com/williamfong/lightcurve-hypothesis/issues

Explain:

* The use case for the feature
* How it would work
* Example code showing the desired API

Code Contributions
~~~~~~~~~~~~~~~~~~

Areas where contributions are especially welcome:

* New lightcurve generators for specific phenomena
* Additional utility functions
* Performance improvements
* Documentation improvements
* Test coverage improvements

Questions?
----------

Feel free to open an issue for any questions about contributing!
