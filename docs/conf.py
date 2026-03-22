# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "pytc"
copyright = "2025, Ke Liao"
author = "Ke Liao"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST Markdown configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "pytc"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/nickirk/pytc",
    "source_branch": "main",
    "source_directory": "docs/",
}
