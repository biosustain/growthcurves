# Contributing code

Install the code with development dependencies:

```bash
pip install -e '.[dev]'
```

## Format code and sort imports

```bash
black .
isort .
```

## lint code

```bash
ruff check .
```

## Run tests

```bash
pytest
```

## Sync notebooks with jupytext

For easier diffs, you can use jupytext to sync notebooks in the `docs/tutorial` directory with the percent format.

```bash
jupytext --sync docs/tutorial/*.ipynb
```

This is configured in the [`.jupytext`](docs/tutorial/.jupytext) file in that directory.
