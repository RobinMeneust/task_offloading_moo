"""Sphinx configuration file for the task_offloading_moo package."""

import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_path, "src"))

sys.path.insert(0, src_path)


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "task_offloading_moo"
copyright = "2025, Maxime Cabrit, Antoine Charlet, Robin Meneust, Ethan Pinto"
author = "Maxime Cabrit, Antoine Charlet, Robin Meneust, Ethan Pinto"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "rst2pdf.pdfbuilder",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
