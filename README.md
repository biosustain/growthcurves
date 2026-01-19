# Example Python package

All design principles are explained in the [developing.md](developing.md) file.
The Python package template was created by Jakob Nybo Nissen and Henry Webel.

## How to use

Can be used as GitHub template repository,
see [GitHub documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).

You will need to find and replace occurences of

- `python_package` -> `your_package_name`
  - also the folder `src/python_package`
- `RasmussenLab` -> `GitHub_user_name` (or `organization`)
  with the name of your package and GitHub user name (or organization).

- look for `First Last` to see where to replace with your name
- choose a license, see [GitHub documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/licensing-a-repository)
  and [Creative Commons](https://creativecommons.org/chooser/).
  Replace [`LICENSE`](LICENSE) file with the license you choose.
- Update the `CITATION.cff` file with your information.

## Development environment

Install package so that new code is picked up in a restared python interpreter:

```
pip install -e ".[dev]"
```

## Basic usage

> works using this template

```python
from python_package import hello_world
print (python_package.__version__)
print(hello_world(4))
```

## Readthedocs

The documentation can be build using readthedocs automatically. See
[project on Readthedocs](https://readthedocs.org/projects/rasmussenlab-python-package/)
for the project based on this template. A new project needs
to [be registered on ReadTheDocs](https://docs.readthedocs.com/platform/stable/intro/add-project.html).

- make sure to enable build from PRs in the settings (advanded settings)
- checkout configuration file: [`.readthedocs.yaml`](.readthedocs.yaml)
