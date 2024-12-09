import os
import sys


# Add the parent directory to the Python path
# This allows Sphinx to find and import the package for documentation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

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
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc.typehints',
    'myst_parser',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_design',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autodoc type hints settings
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_type_aliases = {}

# Autosummary settings
autosummary_generate = True
add_module_names = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Add any paths that contain templates here
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# PyData theme options
html_theme_options = {
    "github_url": "https://github.com/renero/causalgraph",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/renero/causalgraph",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
        "text": "CausalGraph",  # Fallback if no logo images
    },
    "announcement": "This is the development version of CausalGraph.",
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "show_prev_next": True,
}

# Theme-specific sidebars
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs", "sidebar-ethical-ads"]
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
