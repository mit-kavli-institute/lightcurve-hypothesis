"""Sphinx configuration file for hypothesis_lightcurves documentation."""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath("../../src"))

# Project information
project = "hypothesis-lightcurves"
copyright = f"{datetime.now().year}, William Fong"
author = "William Fong"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "numpydoc",
    "sphinx_copybutton",
    "myst_parser",
]

# Add support for both RST and Markdown
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
autodoc_typehints = "description"
autodoc_type_aliases = {}

# Napoleon settings for NumPy style docstrings
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
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "hypothesis": ("https://hypothesis.readthedocs.io/en/latest/", None),
}

# HTML output options
html_theme = "furo"
html_title = "hypothesis-lightcurves"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Furo theme options
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/mit-kavli-institute/lightcurve-hypothesis",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# Autosummary configuration
autosummary_generate = False  # Disable for now to avoid issues
# autosummary_generate_overwrite = True

# MyST configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "tasklist",
]

# Exclude patterns
exclude_patterns = []

# Pygments style for code highlighting
pygments_style = "sphinx"
pygments_dark_style = "monokai"
