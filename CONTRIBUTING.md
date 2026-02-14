# Contributing to QUITS

Thanks for your interest in improving QUITS. This project is research-oriented software, but we aim for production-quality modularity and reproducibility.

## Ways to Contribute

- Report bugs (include a minimal reproducible example).
- Propose new code families / circuit construction strategies / decoders / noise models.
- Improve documentation and tutorials in `doc/`.
- Add tests and reduce maintenance burden.

## Development Setup

Requirements:

- Python >= 3.10

Install for development:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Optional (packaging sanity check):

```bash
python -m build
```

## Project Conventions

- Keep changes modular: avoid cross-layer coupling between code construction, circuit construction, noise modeling, and decoding.
- Prefer explicit configuration objects over hidden global state.
- If you add randomness, make results reproducible (seedable / injectable RNG).
- Add/adjust tests in `tests/` for any behavior change.
- Update `README.md`/`doc/` when you change user-facing behavior.
- If the change is user-visible, add a short note to `CHANGELOG.md`.

## Pull Requests

- Keep PRs small and focused.
- Describe the problem and acceptance criteria (what should change and how you verified it).
- Ensure CI passes (`.github/workflows/ci.yml` runs `pytest` on PRs).

