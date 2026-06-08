# Repository Guidelines

## Repo Layout

- `src/eht_inspection/` contains the Python package source.
- `src/eht_inspection/alist.py` handles ALIST-related inspection helpers.
- `src/eht_inspection/uvfits.py` handles UVFITS-related inspection helpers.
- `src/eht_inspection/coherence.py`, `closure.py`, and `fringe.py` contain core analysis helpers.
- `src/eht_inspection/filters.py`, `plotting.py`, and `utils.py` contain filtering, matplotlib plotting, and shared utilities.
- `tests/` contains pytest-based test coverage.
- `notebooks/examples/` contains example notebooks.
- `data/` contains sample data documentation and related assets.
- `scripts/` contains command-line helper scripts.

## Development Conventions

- Keep public APIs backward compatible unless a breaking change is explicitly requested.
- Preserve scientific and data-processing behavior unless the task explicitly asks to change it.
- Prefer small helper functions and focused changes over large rewrites.
- Avoid editing notebooks unless explicitly requested.
- Keep plotting functions matplotlib-based.
- Keep plotting refactors focused on readability, shared helpers, and style consistency.
- Do not change plotted quantities, filtering logic, averaging logic, baseline orientation, polarization handling, or time conversion during plotting-only refactors.
- Use clear docstrings for public functions.
- Private helper functions should usually start with `_`.

## Dependency Setup

Before running tests in a fresh environment, install the package in editable mode with development dependencies:

```sh
pip install -e ".[dev]"
```

If ALIST plotting functionality or optional seaborn-based helpers are involved, use:

```sh
pip install -e ".[dev,alist]"
```

If dependency installation is not possible in the current environment, clearly report which validation steps were skipped and why.

## Testing and Checks

Run the relevant checks before submitting changes:

```sh
python -m compileall src
pytest
python -c "import eht_inspection"
```

If `pytest`, `matplotlib`, or other required dependencies are not available in the environment, do not claim that tests passed. Instead, note the missing dependencies in the PR summary.

## Notebook Handling

- Avoid editing notebooks unless the task explicitly requires it.
- Before committing notebook changes, strip notebook outputs:

```sh
find . -name "*.ipynb" -not -path "./.git/*" -print0 | xargs -0 nbstripout
```

- Do not commit large notebook outputs or transient execution metadata.

## PR Summaries

PR summaries should include:

- Changed files and the purpose of each change.
- Backward compatibility impact, including whether public APIs changed.
- Scientific/data-processing behavior impact, especially for plotting, closure, fringe, ALIST, or UVFITS logic.
- Verification commands run and their results.
- Any validation commands that could not be run, with the reason.
