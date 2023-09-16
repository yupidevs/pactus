# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../pactus/"))
sys.path.insert(0, os.path.abspath("../../pactus/dataset/"))
sys.path.insert(0, os.path.abspath("../../pactus/models/"))

project = "pactus"
copyright = "2023, Jorge Morgado Vega"
author = "Jorge Morgado Vega, Gustavo Viera-López, Alfredo Reyes"
release = "0.4.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]
autodoc_typehints = "description"
add_module_names = False
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
