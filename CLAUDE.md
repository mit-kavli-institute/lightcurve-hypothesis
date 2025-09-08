# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for generating and testing lightcurves using property-based testing with Hypothesis. The package provides data models and generators for creating synthetic astronomical lightcurves for testing purposes.

## Commands

### Development Setup
```bash
# Install package in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files (useful for initial setup)
pre-commit run --all-files
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=hypothesis_lightcurves --cov-report=term-missing

# Run a specific test file
pytest tests/test_generators.py

# Run tests matching a pattern
pytest -k "test_periodic"

# Run tests with verbose output
pytest -v
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/

# Run all quality checks
black src/ tests/ && ruff check src/ tests/ && mypy src/

# Run pre-commit hooks manually
pre-commit run --all-files

# Update pre-commit hooks to latest versions
pre-commit autoupdate
```

### Pre-commit Hooks
The project uses pre-commit to automatically run code quality checks before commits. The following hooks are configured:
- **black**: Formats code according to project style (line length 100, Python 3.11+)
- **ruff**: Lints code for errors and style issues
- **mypy**: Checks type annotations
- **Standard hooks**: trailing whitespace, end-of-file fixer, YAML/JSON/TOML validation, merge conflict detection

All tool configurations are read from `pyproject.toml` to ensure consistency.

## Architecture

### Package Structure
The package uses a src layout with the main code in `src/hypothesis_lightcurves/`:

- **models.py**: Contains the `Lightcurve` dataclass that represents astronomical time series data with time, flux, optional errors, and metadata.

- **generators.py**: Hypothesis strategies for generating test lightcurves:
  - `lightcurves()`: Basic random lightcurves
  - `periodic_lightcurves()`: Sinusoidal periodic signals
  - `transient_lightcurves()`: Transient events (supernova-like)

- **utils.py**: Utility functions for lightcurve manipulation:
  - `resample_lightcurve()`: Change sampling rate
  - `add_gaps()`: Introduce data gaps
  - `calculate_periodogram()`: Simple period analysis
  - `bin_lightcurve()`: Time binning

### Testing Strategy
Tests use Hypothesis for property-based testing, ensuring that generated lightcurves satisfy expected invariants across a wide range of random inputs. This approach is particularly valuable for testing astronomical data processing pipelines.

## Key Dependencies
- **hypothesis**: Property-based testing framework
- **numpy**: Numerical operations and array handling
- **pytest**: Test runner with coverage support

## Python Version
Requires Python 3.11 or higher to leverage modern Python features.