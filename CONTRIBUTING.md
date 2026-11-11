# Contributing

Install the package and development tools with
`python -m pip install -e '.[ase,dev]'`. Before submitting changes, run:

```sh
ruff format --check src examples/toy_live.py tests
ruff check src examples/toy_live.py tests
mypy src
pytest
```

Include focused tests for new behavior and preserve the ability to import the
library with NumPy alone. Do not commit generated caches or private data. For
releases, update the changelog and version, create a semver tag and GitHub
release, and configure the PyPI trusted publisher for the `publish.yml`
workflow and `pypi` environment before publishing.
