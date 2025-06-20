# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULCMore actions
#
# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
import os
from pathlib import Path

# Add the Python packages directory to the path
build_dir = Path(__file__).resolve().parents[3] / "build"
sys.path.insert(0, str(build_dir / "python_packages"))

# Output paths
html_dir = build_dir / "docs/book/autogen/html/ttir-builder"
md_dir = build_dir / "docs/src/autogen/md/ttir-builder"

# Create output directories
html_dir.mkdir(parents=True, exist_ok=True)
md_dir.mkdir(parents=True, exist_ok=True)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ttir-builder"
copyright = "2025, Julia Grim"
author = "Julia Grim"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "../../build/python_packages/ttir_builder/builder.py"
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "private-members": True,
    "special-members": "__init__",
    "member-order": "bysource",
}
autodoc_docstring_signature = True
autodoc_typehints = "description"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_class_member_order = "bysource"
autodoc_typehints = "description"
autodoc_docstring_signature = True
autodoc_preserve_defaults = True

# Autosummary settings
autosummary_generate = True  # ['generated/ttir_builder.TTIRBuilder']
# html_split_index = False

templates_path = ["_templates"]
exclude_patterns = ["test_optimizer_overrides.py"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("tt_theme.css")
