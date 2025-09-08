"""Nox configuration for running tests and quality checks."""

import nox


@nox.session(python=["3.11", "3.12"])
def tests(session):
    """Run the test suite."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-v", "--tb=short", *session.posargs)


@nox.session(python="3.11")
def coverage(session):
    """Run tests with coverage reporting."""
    session.install("-e", ".[dev]")
    session.install("pytest-cov")
    session.run(
        "pytest",
        "--cov=hypothesis_lightcurves",
        "--cov-report=term-missing",
        "--cov-report=html",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session):
    """Run linting with ruff."""
    session.install("ruff")
    session.run("ruff", "check", "src", "tests")


@nox.session(python="3.11")
def format(session):
    """Format code with black."""
    session.install("black")
    session.run("black", "--check", "src", "tests")


@nox.session(python="3.11")
def mypy(session):
    """Run type checking with mypy."""
    session.install("-e", ".[dev]")
    session.install("mypy")
    session.run("mypy", "src")


@nox.session(python="3.11")
def quality(session):
    """Run all quality checks."""
    session.install("-e", ".[dev]")
    session.install("black", "ruff", "mypy")
    
    # Format check
    session.run("black", "--check", "src", "tests")
    
    # Lint
    session.run("ruff", "check", "src", "tests")
    
    # Type check
    session.run("mypy", "src")


@nox.session(python="3.11")
def dev(session):
    """Set up development environment."""
    session.install("-e", ".[dev]")
    session.install("pre-commit")
    session.run("pre-commit", "install")
    print("\n✅ Development environment ready!")
    print("Run 'nox' to run all tests")
    print("Run 'nox -s tests' to run tests only")
    print("Run 'nox -s coverage' to run tests with coverage")
    print("Run 'nox -s quality' to run all quality checks")