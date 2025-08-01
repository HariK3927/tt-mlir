# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULCMore actions
#
# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ttir-builder"
copyright = "2025 Tenstorrent AI ULC"
author = "Nick Smith"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "private-members": False,
}
autodoc_docstring_signature = True
autodoc_typehints = "description"
autodoc_member_order = "alphabetical"

# Napoleon settings
napoleon_numpy_docstring = True

# Autosummary settings
autosummary_generate = True

# Exclude patterns
exclude_patterns = ["modules.rst", "ttir_builder.rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"


def autodoc_skip_member(app, what, name, obj, skip, options):
    if hasattr(obj, "__autodoc_skip__") and obj.__autodoc_skip__:
        return True  # Skip this member
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
