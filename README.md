# hypothesis_lightcurves

[![CI](https://github.com/williamfong/lightcurve-hypothesis/actions/workflows/ci.yml/badge.svg)](https://github.com/williamfong/lightcurve-hypothesis/actions/workflows/ci.yml)
[![Documentation](https://github.com/williamfong/lightcurve-hypothesis/actions/workflows/docs.yml/badge.svg)](https://github.com/williamfong/lightcurve-hypothesis/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightcurve generation testing tool using [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing of astronomical data processing pipelines.

## Features

- **Property-based testing strategies** for generating synthetic lightcurves
- **Multiple lightcurve types**: periodic, transient, and random signals
- **Extensible visualization system** with automatic gallery generation
- **Comprehensive documentation** with examples and tutorials
- **Utility functions** for common lightcurve operations (binning, resampling, gap addition)

## Installation

```bash
pip install hypothesis_lightcurves
```

For development installation with all dependencies:

```bash
git clone https://github.com/williamfong/lightcurve-hypothesis.git
cd lightcurve-hypothesis
pip install -e ".[dev,docs]"
```

## Quick Start

```python
from hypothesis import given
from hypothesis_lightcurves.generators import lightcurves, periodic_lightcurves

@given(lc=lightcurves(min_points=50, max_points=200))
def test_my_analysis_function(lc):
    """Test that my analysis works on any lightcurve."""
    result = my_analysis_function(lc)
    assert result.is_valid()

@given(lc=periodic_lightcurves(min_period=1.0, max_period=10.0))
def test_period_detection(lc):
    """Test period detection algorithm."""
    detected_period = detect_period(lc)
    true_period = lc.metadata['period']
    assert abs(detected_period - true_period) / true_period < 0.1
```

## Documentation

Full documentation is available at [GitHub Pages](https://williamfong.github.io/lightcurve-hypothesis/) (once deployed).

To build documentation locally:

```bash
nox -s docs
# View at docs/build/html/index.html
```

## Development

This project uses [nox](https://nox.thea.codes/) for task automation:

```bash
# Run all tests
nox -s tests

# Run linters
nox -s lint

# Format code
nox -s format

# Build documentation
nox -s docs
```

## Contributing

Contributions are welcome! Please see the [Contributing Guide](docs/source/contributing.rst) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
