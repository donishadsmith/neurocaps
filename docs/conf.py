# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinxext"))

from github_link import make_linkcode_resolve

# -- Project information -----------------------------------------------------

project = "neurocaps"
copyright = "2024-2025, NeuroCAPs Developers"
author = "Donisha Smith"

import neurocaps

# The full version, including alpha/beta/rc tags
release = neurocaps.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "sphinxcontrib.redirects",
    "nbsphinx",
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_default_options = {"members": False, "inherited-members": False}
numpydoc_show_class_members = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "none"

# Remove module name in signature
add_module_names = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_include_private_with_doc = False

nbsphinx_execute = "never"
nbsphinx_prompt_width = "0"
nbsphinx_codecell_lexer = "python3"

pygments_style = "sphinx"

sphinx_gallery_conf = {
    "thumbnail_size": (350, 250),
}

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_utils/*"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "NeuroCAPs",
    },
    "back_to_top_button": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/donishadsmith/neurocaps/stable/docs/_static/versions.json",
        "version_match": release,
    },
    "header_links_before_dropdown": 6,
    "secondary_sidebar_items": ["page-toc"],
    "navbar_align": "content",
    "navbar_persistent": [],
}

# Remove primary sidebar for certain pages
html_sidebars = {
    "contributing": [],
    "changelog": [],
}

nbsphinx_thumbnails = {
    "tutorials/tutorial-[457]": "_static/neurocaps.png",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

redirects_file = "redirects"

def setup(app):
    app.add_css_file("custom.css")
    app.add_css_file("theme_overrides.css")


# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "neurocaps",
    "https://github.com/donishadsmith/neurocaps/blob/{revision}/{package}/{path}#L{lineno}",
)
