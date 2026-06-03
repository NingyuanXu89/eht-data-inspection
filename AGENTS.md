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
- Prefer small helper functions and focused changes over large rewrites.
- Avoid editing notebooks unless explicitly requested.
- Keep plotting functions matplotlib-based.
- Use clear docstrings for public functions.

## Testing and Checks

Run the relevant checks before submitting changes:

```sh
pytest
python -c "import eht_inspection"
```

If `pytest` is not available in the environment, note that in the PR summary.

## PR Summaries

PR summaries should include:

- Changed files and the purpose of each change.
- Backward compatibility impact, including whether public APIs changed.
- Verification commands run and their results.
