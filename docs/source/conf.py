"""Sphinx configuration file for hypothesis_lightcurves documentation."""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("../../src"))

# Project information
project = "hypothesis_lightcurves"
copyright = f"{datetime.now().year}, William Fong"
author = "William Fong"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

# Add support for Markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "hypothesis": ("https://hypothesis.readthedocs.io/en/latest/", None),
}

# HTML output options
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Static files
html_static_path = ["_static"]
html_css_files = []

# Exclude patterns
exclude_patterns = []

# Autosummary configuration
autosummary_generate = True

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/plot_",  # pattern to match example files
    "ignore_pattern": r"__pycache__|\.py[cod]$",
    "plot_gallery": True,
    "download_all_examples": False,
    "show_memory": False,
    "remove_config_comments": True,
    "default_thumb_file": None,
    "matplotlib_animations": False,
    "image_scrapers": ("matplotlib",),
    "reset_modules": ("matplotlib", "seaborn"),
    "first_notebook_cell": None,
    "last_notebook_cell": None,
    "notebook_images": False,
    "abort_on_example_error": False,
    "expected_failing_examples": [],
    "min_reported_time": 0,
    "show_signature": False,
    "inspect_global_variables": False,
    "doc_module": ("hypothesis_lightcurves",),
    "reference_url": {
        "hypothesis_lightcurves": None,
    },
    "capture_repr": ("_repr_html_", "__repr__"),
    "ignore_repr_types": r"matplotlib\.text|matplotlib\.axes",
}
