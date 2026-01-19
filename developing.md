# Design descriptions and details for the Python package template

> Author: Henry Webel

[packaging.python.org](https://packaging.python.org/en/latest/tutorials/packaging-projects/) 
has an excellent tutorial on how to package a Python project. I read and used insights from
that website to help create the template which is available on GitHub at
[https://github.com/RasmussenLab/python_package](https://github.com/RasmussenLab/python_package)
and I want to give here an overview specifically to some details regarding this template.
Some are overlapping with the 
[packaging.python.org](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
tutorial, but as always we decided for a certain set of tools, conventions and complexity 
which needs some explanation.

Here a brief overview of external resources you can also look at:

- [packaging.python.org](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [setuptools documentation](https://setuptools.pypa.io/en/latest/userguide/index.html)
- [learn.scientific-python.org](https://learn.scientific-python.org/development/)
- [Py-Pkgs](https://py-pkgs.org/)

## Project structure

First an overview of the main folder structure. See line comments for details on what 
is the purpose of each folder or file:

```bash
python_package
├── docs # Documentation using Sphinx
├── src # the source code of the package
├── tests # pytest tests
├── LICENSE # License file specifying usage terms
├── MANIFEST.in # non-python files to include into the build package
├── pyproject.toml # python package metadata, dependencies and configurations (incl. build tools)
├── pytest.ini # pytest configuration
├── README.md # README which is rendered on GitHub (or other hosting services)
└── setup.cfg # old python configuration file, empty
└── setup.py # artefact for backward compatibility, do not change
```

## Core packaging files

We will first look at [`pyproject.toml`](pyproject.toml) and its relation to the 
[`src`](src) directory. The 
[`pyproject.toml`](pyproject.toml) file is the main configuration file for the Python package
and is used to specify the package metadata, dependencies, build tools and configurations.
The [`src`](src) folder stores the actual source code of the package, where the package itself is
the subdirectories of the [`src`](src) directory. The  (e.g. `src/python_package`).

<details>
<summary>About <code>setup.py</code> and <code>setup.cfg</code> configuration files</summary>

The [`setup.py`](setup.py) file is an artefact for backward compatibility and should not 
be changed.  Everything that used to be in [`setup.py`](setup.py) or 
[`setup.cfg`](setup.cfg) is now largely in [`pyproject.toml`](pyproject.toml).
The notable exception would be the desired maximum line length in `setup.cfg` for 
the tool [`flake8`](https://flake8.pycqa.org/), which does not yet supported
[`pyproject.toml`](pyproject.toml) configuration. As we use `ruff` as linter,
we left it empty, but in case you want to use `flake8`, you can add:

```INI
; setup.cfg
[flake8]
exclude = docs
max-line-length = 88
aggressive = 2
```

</details>

### Changes required in `pyproject.toml`

You have to change entries under the `[project]` section to match your project name,
description, author, license, etc. Make sure to pick a license that works for you, e.g. 
using [choosealicense.com](https://choosealicense.com/). Also update the `LICENSE` file
accordingly.

The `dependencies` key can 
list the dependencies and is currently commented out. The dependencies could also be 
specified in via a `requirements.txt`, if you already have such a file.

```toml
# ref: https://setuptools.pypa.io/en/stable/userguide/pyproject_config.html
[project]
authors = [
  { name = "First Last", email = "first.last@gmail.com" },
]
description = "A small example package"
name = "python_package"
# This means: Load the version from the package itself.
# See the section below: [tools.setuptools.dynamic]
dynamic = ["version", # version is loaded from the package
#"dependencies", # add if using requirements.txt
]
readme = "README.md"
requires-python = ">=3.9"
# These are keywords
classifiers = [
  "Programming Language :: Python :: 3",
  "Operating System :: OS Independent",
]
license = "MIT" # https://choosealicense.com/
# # add dependencies here: (use one of the two)
# dependencies = ["numpy", "pandas", "scipy", "matplotlib", "seaborn"]
# use requirements.txt instead of pyproject.toml for dependencies
# https://stackoverflow.com/a/73600610/9684872
# [tool.setuptools.dynamic]
# dependencies = {file = ["requirements.txt"]}
```

The entry

```toml
dynamic = ["version"]
```

means that the version is loaded dynamically using the extension 
[setuptools_scm](https://setuptools-scm.readthedocs.io/)
we list under the `[build-system]` section in [`pyproject.toml`](pyproject.toml). 
This is done to avoid having to manually update the version and integrate with automatic 
versioning through releases on GitHub. It also
ensures that each commit has a unique version number, which is useful for attributing
errors to specific non-released versions. The dynamic version is picked up in the 
`__version__` variable in the `__init__.py` file of the package, which is located in the
[`src/python_package`](src/python_package) directory.

```toml
[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools>=64", "setuptools_scm>=8"]

[tool.setuptools_scm]
# https://setuptools-scm.readthedocs.io/ 
# used to pick up the version from the git tags or the latest commit.
```

Please also update the project URL to your project:

```toml
[project.urls]
"Bug Tracker" = "https://github.com/RasmussenLab/python_package/issues"
"Homepage" = "https://github.com/RasmussenLab/python_package"
```

The template also sets a command line script entry point, which allows to run
the function `main` in the `mockup` module of the package as a command line script,
wrapping the `hello_world` function from the `mockup` module.
The template uses the standard library [argparse](https://docs.python.org/3/library/argparse.html)
module to parse parameters from the command line and creates a basic interface.

```toml
# Script entry points, i.e. command line commands available after installing the package
# e.g. implemented using argparse
# Then you can type: `python-package-hello -h` in the terminal
[project.scripts]
python-package-hello = "python_package.cli:main"
```

You can therefore run the command line script using:

```bash
python-package-hello -h
# print hello world 3 times
python-package-hello -n 3
```

## Source directory layout of the package

The source code of the package is located in the `src` directory, to have a project 
independent folder to look for the source code recognized by most tools you would need
to build a package 
(read on [packagin namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/)). 
It also allows to have multiple subpackages or modules 
in the same project under the `python_package` package (see example 
[here](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout)).

```bash
├── src
│   └── python_package
│       ├── __init__.py # imported when the package is imported (import python_package)
│       └── mockup.py # a submodule of the package (import python_package.mockup)
```

So you will need to rename the `python_package` directory to your package name, 
e.g. `my_package` and specify the package name in the [`pyproject.toml`](pyproject.toml) file 
under the `[project]` section:

```toml
name = "my_package"
```

Strictly speaking you can give different names in both places, but this will only confuse
potential users. Think of `scikit-learn` for an example of a package that uses a different
name in the [`pyproject.toml`](pyproject.toml) file and the source code directory name, 
leading to the `sklearn` package name when imported.

## Documentation

The documentation is created using [Sphinx](https://www.sphinx-doc.org/en/master/),
which is common for Python documentation. It relies additionally on several extensions
enabling the use of `markdown` and `jupyter` notebooks. 

The documentation is located in the [`docs`](docs) directory. Sphinx is configured via
the [`conf.py`](docs/conf.py) file, where you can specify the extension you want:

```python
# in docs/conf.py

extensions = [
    "sphinx.ext.autodoc",  # Core extension for generating documentation from docstrings
    "sphinx.ext.autodoc.typehints",  # Automatically document type hints in function signatures
    "sphinx.ext.viewcode",  # Include links to the source code in the documentation
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.intersphinx",  # allows linking to other projects' documentation in API
    "sphinx_new_tab_link",  # each link opens in a new tab
    "myst_nb",  # Markdown and Jupyter Notebook support
    "sphinx_copybutton",  # add copy button to code blocks
]
```

These are added as dependencies through the 
`pyproject.toml` file under the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
# Optional dependencies to locally build the documentation, also used for 
# readthedocs.
docs = [
  "sphinx",
  "sphinx-book-theme",
  "myst-nb",
  "ipywidgets",
  "sphinx-new-tab-link!=0.2.2",
  "jupytext",
]
```

### Required changes in `conf.py`

The required changes in [`conf.py`](docs/conf.py) are at the following places:

```python
# in docs/conf.py

project = "python_package"
copyright = "2025, First Last"
author = "First Last"
PACKAGE_VERSION = metadata.version("python_package")

# ...

# and again links to your project repository
html_theme_options = {
    "github_url": "https://github.com/RasmussenLab/python_package",
    "repository_url": "https://github.com/RasmussenLab/python_package",
    # more...
}

# ...

# and one last line (the last below)
if os.environ.get("READTHEDOCS") == "True":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent
    PACKAGE_ROOT = PROJECT_ROOT / "src" / "python_package"
```

The last block is for Read The Docs to be able to generate the API documentation of your
package on the fly. See the Read The Docs section below for more details.

### Theme, autodoc and intersphinx

We build the documentation based on the template
[sphinx_book_theme](https://sphinx-book-theme.readthedocs.io), which is set in the
[`conf.py`](docs/conf.py) file and parts of our docs requirements in 
[`pyproject.toml`](pyproject.toml):

```python
html_theme = "sphinx_book_theme"
```

> If you use a different theme, some of the settings in `conf.py` might not be applicable
> and need to be changed. Explore other themes here:
> [sphinx-themes.org](https://sphinx-themes.org/)

The API of the Python package in the `src` directory is automatically included 
in the documentation using the
[`autodoc` extension](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).
We use per default the `numpydoc` style for docstrings, see the format
[here](https://numpydoc.readthedocs.io/en/stable/format.html).
The API documentation can be augmented with highlights from other types from projects 
using `intersphinx`:

```python
# Intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    # "scikit-learn": ("https://scikit-learn.org/stable/", None),
    # "matplotlib": ("https://matplotlib.org/stable/", None),
}
```

Here we only add the core Python documentation, but you can add more projects
like `pandas`, `scikit-learn`, or `matplotlib` to the mapping.

### Building the documentation locally (with integration tests)

To build the documentation locally, you can follow the instructions in the
[`docs/README.md`](docs/README.md), which you should also update with your name changes. 
In short, you can run the following commands in the [`docs`](docs ) directory:

```bash
# in root of the project
pip install ".[docs]"
cd docs # change to docs directory
sphinx-apidoc --force --implicit-namespaces --module-first -o reference ../src/python_package
sphinx-build -n -W --keep-going -b html ./ ./_build/
```

this will create a `reference` directory with the API documentation of the Python package
`python_package`, a `jupyter_execute` for the tutorial in [`docs/tutorial`](docs/tutorial)
 and a `_build` directory with an HTML version of the documentation. You can open the
`_build/index.html` file in your browser to view the documentation built locally.

The tutorial related configuration in `conf.py` is the following, specifying that 
errors stop the build process ensuring that examples are tested:

```python
#  https://myst-nb.readthedocs.io/en/latest/computation/execute.html
nb_execution_mode = "auto"

myst_enable_extensions = ["dollarmath", "amsmath"]

# Plotly support through require javascript library
# https://myst-nb.readthedocs.io/en/latest/render/interactive.html#plotly
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

# https://myst-nb.readthedocs.io/en/latest/configuration.html
# Execution
nb_execution_raise_on_error = True
# Rendering
nb_merge_streams = True
```

The tutorials are meant as a sort of integration test, where you make sure that the core
functionality your project wants to support is working as expected. For easier github
diffs, we use [`jupytext`](https://jupytext.readthedocs.io), which allows to 
have the tutorial in both a Jupyter Notebook format and a Python script format. 
You have to keep the files in sync using:

```bash
jupytext --sync docs/tutorial/*.ipynb
```

The [`docs/tutorial/.jupytext`](docs/tutorial/.jupytext) configuration sets the default 
format to `py:percent` and automatically allows syncing of new notebooks.

### Read The Docs

To build the documentation on Read The Docs, you need to create a file called
[`.readthedocs.yaml`](.readthedocs.yaml), which is located in the root of the project and 
specifies which dependencies are needed. The core is the following specifying where the 
[`conf.py`](docs/conf.py) file is and from where to install the required dependencies:

```yaml
# Build documentation in the "docs/" directory with Sphinx
sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

You will need to manually register your project repository on 
[Read The Docs](https://readthedocs.org/) in order that it can build the documentation
by the service. I recommend to activate builds for Pull Requests, so that 
the documentation is built for each PR and you can see if the documentation is gradually 
breaking, i.e. your integration test using the notebooks in
[`docs/tutorial`](docs/tutorial) fail. See their documentation
[on adding a project](https://docs.readthedocs.com/platform/stable/intro/add-project.html) 
for instructions.

## Running tests

The tests are located in the `tests` directory and can be run using `pytest`.
Pytest is specified as a dependency in the `pyproject.toml` file under the
`[project.optional-dependencies]` section along with the formatter `black` and the 
linter `ruff`:

```toml
[project.optional-dependencies]
# local development options
dev = ["black[jupyter]", "ruff", "pytest"]
```

Instead of running these tools manually, typing

```bash
black .
ruff check .
pytest tests
```

read the next section to see how this is automated using `GitHub Actions`.

## GitHub Actions

We run these checks also on GitHub using GitHub Actions. The configuration
for the actions is located in the [`.github/workflows`](.github/workflows) directory 
and is specified in the `cdci.yml` file. See the biosustain dsp tutorial on GitHub Actions
for more details (or any other resource you find):
[biosustain/dsp_actions_tutorial](https://github.com/biosustain/dsp_actions_tutorial)

```yaml
# This workflow will install Python dependencies, run tests and lint with a single version of Python
# For more information see: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python

name: Python application

on:
  push:
  pull_request:
    branches: ["main"]
  schedule:
    - cron: "0 2 * * 3"

permissions:
  contents: read

jobs:
  format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: psf/black@stable
  lint:
    name: Lint with ruff
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install ruff
        run: |
          pip install ruff
      - name: Lint with ruff
        run: |
          # stop the build if there are Python syntax errors or undefined names
          ruff check .
  test:
    name: Test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip" # caching pip dependencies
          cache-dependency-path: "**/pyproject.toml"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest
          pip install -e .
      - name: Run tests
        run: python -m pytest tests
```

This workflow also allows to create `PyPI` releases automatically if you register your 
project on `PyPI` (or `TestPyPI` for testing first) and create a GitHub release:

```yaml
  publish:
    name: Publish package
    if: startsWith(github.ref, 'refs/tags')
    needs:
      - format
      - lint
      - test
      - build_source_dist
      # - build_wheels
    runs-on: ubuntu-latest

    steps:
      - uses: actions/download-artifact@v4
        with:
          name: artifact
          path: ./dist

      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          # remove repository key to set the default to pypi (not test.pypi.org)
          repository-url: https://test.pypi.org/legacy/
```

To setup the [`gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish)
action, you need to register the repository
on [PyPI](https://pypi.org/) or [`TestPyPI`](https://test.pypi.org/), which allows PyPI 
and GitHub to communicate securely. See the instructions on
[packaging.python.org](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

You then trigger new releases to PyPI by creating a new GitHub release, which will
automatically trigger the `publish` job in the workflow as it needs you to set a tag.
Have a look at [VueGen Releases]( https://github.com/Multiomics-Analytics-Group/vuegen/releases)
for an example. The release notes are automatically generated using the PR titles,
see GitHub's
[docs](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes).

<details>
<summary>Wheels and testing builds</summary>
The wheels are not built by default, but you can be necessary for packages which need
to be partly compiled, e.g. if you use `Cython`, `numpy` C extensions or Rust extensions.

Also additionally you could use the artifact from the `build_source_dist` job
to test the build of the source distribution. This is useful to ensure that a package
with non-Python files (e.g. data files) is built correctly and that the package
can be installed correctly. You should probably best test this in as much isolation as 
you can, e.g. by not pulling the repository using `actions/checkout@v4`.

```yaml
  test_sdist:
    name: Install built source distribution
    needs: build_source_dist
    runs-on: ubuntu-latest
    steps:
      # - uses: actions/checkout@v4 
      - uses: actions/download-artifact@v4
        with:
          name: artifact
          path: ./dist
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install built sdist
        run: |
          pip install ./dist/*.tar.gz
      # ... some checks
```
</details>



## Full project structure

```bash
python_package
├── docs
│   ├── tutorial
│   │   ├── tutorial.ipynb # tutorial in Jupyter Notebook format
│   │   └── tutorial.py # tutorial in Python script format (created by jupytext)
│   ├── conf.py # configuration for Sphinx documentation
│   ├── index.md # defining the website structure
│   ├── Makefile # can be ignored
│   └── README.md # specifies how to build the documentation
├── src
│   └── python_package
│       ├── __init__.py # imported when the package is imported (import python_package)
│       └── mockup.py # a submodule of the package (import python_package.mockup)
├── tests
│   ├── __init__.py
│   └── test_mockup.py # files and test_function need to start with test_ to be recognized by pytest
├── LICENSE # License file specifying usage terms
├── MANIFEST.in # non-python files to include into the build package
├── pyproject.toml # python package metadata, dependencies and configurations (incl. build tools)
├── pytest.ini # pytest configuration
├── README.md # README which is rendered on GitHub (or other hosting services)
└── setup.cfg # old python configuration file, empty
└── setup.py # artefact for backward compatibility, do not change
```
