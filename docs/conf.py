#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# General configuration
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'nbsphinx']
source_suffix = '.rst'
master_doc = 'index'

# General information about the project
project = 'SynPlanner'
copyright = 'Laboratoire de Chemoinformatique'
author = 'Tagir Akhmetshin / Dmitry Zankov'

# Software version and release
version = '1.0'
release = '1.0'

# Options for HTML output
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "show_nav_level": 4,
    "github_url": "https://github.com/Laboratoire-de-Chemoinformatique/synplanner",
    "show_prev_next": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_sidebars = {
    "**": [
        "localtoc.html",
        "ethicalads.html",
    ],
    "installation": [],  # removes Section Navigation sidebar
    "data_download": [],
}
