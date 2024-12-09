import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder.
project = 'CausalGraph'
copyright = '2024, J. Renero'
author = 'J. Renero'
release = '0.5.0'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'myst_parser',
    'numpydoc',
    'sphinx.ext.autosummary',
]

# Autosummary settings
autosummary_generate = True
add_module_names = False

# Add any paths that contain templates here
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# PyData theme options
html_theme_options = {
    "github_url": "https://github.com/renero/causalgraph",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/renero/causalgraph",
            "icon": "fab fa-github-square",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
}

# GitHub Pages specific settings
html_baseurl = 'https://renero.github.io/causalgraph/'
html_context = {
    'display_github': True,
    'github_user': 'renero',
    'github_repo': 'causalgraph',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
