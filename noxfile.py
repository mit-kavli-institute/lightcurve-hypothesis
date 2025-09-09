"""Nox configuration for running tests and building documentation."""

import nox


@nox.session(python=["3.11", "3.12", "3.13"])
def tests(session):
    """Run the test suite."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)


@nox.session(python="3.11")
def lint(session):
    """Run linters."""
    session.install("-e", ".[dev]")
    session.run("black", "--check", "src", "tests")
    session.run("ruff", "check", "src", "tests")
    session.run("mypy", "src")


@nox.session(python="3.11")
def format(session):
    """Format code with black."""
    session.install("-e", ".[dev]")
    session.run("black", "src", "tests")
    session.run("ruff", "check", "--fix", "src", "tests")


@nox.session(python="3.11")
def docs(session):
    """Build the documentation."""
    session.install("-e", ".[docs]")
    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
    print("\nDocumentation built successfully!")
    print("View it at: docs/build/html/index.html")
