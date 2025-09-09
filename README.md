# hypothesis-lightcurves

A Python package for generating synthetic astronomical lightcurves using Hypothesis property-based testing strategies. This tool helps astronomers test their data processing pipelines with diverse, realistic test data.

## Features

- **Property-based testing** for astronomical time series
- **Flexible lightcurve generators** (baseline, periodic, transient)
- **Composable modifiers** (noise, gaps, outliers, trends)
- **Type-safe** with full type hints
- **NumPy-style documentation**

## Installation

```bash
pip install hypothesis-lightcurves
```

Or install from source:
```bash
git clone https://github.com/mit-kavli-institute/lightcurve-hypothesis.git
cd lightcurve-hypothesis
pip install -e ".[dev]"
```

## Quick Start

```python
from hypothesis import given
from hypothesis_lightcurves.generators import lightcurves

@given(lc=lightcurves())
def test_my_pipeline(lc):
    """Test that my pipeline handles any valid lightcurve."""
    result = process_lightcurve(lc.time, lc.flux)
    assert result is not None
```

## Documentation

Full documentation with API reference and examples is available:

- **Online**: [Read the Docs](https://hypothesis-lightcurves.readthedocs.io/) (coming soon)
- **Local build**:
  ```bash
  pip install -e ".[docs]"
  cd docs
  make html
  # Open docs/build/html/index.html in your browser
  ```

## Development

This project uses:
- **nox** for test automation
- **pre-commit** for code quality
- **Sphinx** with Furo theme for documentation

Run tests:
```bash
nox -s tests
```

Check code quality:
```bash
nox -s quality
```

Build documentation:
```bash
nox -s docs
```

## License

MIT License - see LICENSE file for details.
