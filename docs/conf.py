#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# General configuration
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "myst_parser",
]
source_suffix = ".rst"
master_doc = "index"

# nbsphinx configuration: do not execute notebooks during the build
# This avoids long build times from heavy notebooks
nbsphinx_execute = "never"

# Override nbsphinx's default which contains a lambda (unpickleable),
# so Sphinx can cache the environment and do fast incremental builds
nbsphinx_custom_formats = {}

# General information about the project
project = "SynPlanner"
copyright = "Laboratoire de Chemoinformatique"
author = "Tagir Akhmetshin / Dmitry Zankov"

# Software version and release
try:
    # Resolve version from installed distribution when docs are built in an env
    from importlib.metadata import version as _dist_version

    release = _dist_version("SynPlanner")
    version = ".".join(release.split(".")[:2])
except Exception:
    # Fallback for local builds without installed metadata
    try:
        from synplan import __version__ as _pkg_version

        release = _pkg_version
        version = ".".join(release.split(".")[:2])
    except Exception:
        release = "0.0.0+unknown"
        version = "0.0"

html_title = "SynPlanner"

# Options for HTML output
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 4,
    "show_prev_next": False,
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": os.environ.get("READTHEDOCS_CANONICAL_URL", "")
        + "_static/switcher.json",
        "version_match": os.environ.get("READTHEDOCS_VERSION", release),
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "secondary_sidebar_items": ["page-toc"],
    "header_links_before_dropdown": 7,
}

html_sidebars = {
    "get_started/*": [],  # hide left Section Navigation for Get started pages
    "development": [],
    "release_notes": [],
}

# Static assets (for version switcher JSON, images, etc.)
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Intersphinx mappings for resolving external references (optional, helpful later)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Autodoc settings
autodoc_type_aliases = {
    "Tree": "synplan.mcts.tree.Tree",
}

# Suppress warnings for modules that fail to import at doc-build time
# (e.g. decompositions.py executes code at import that requires runtime deps)
# Also suppress ambiguous cross-reference warnings for re-exported symbols (e.g. Tree)
suppress_warnings = ["autodoc.import_object", "ref.python"]
