# Contributing to RustyStats

Thanks for your interest in contributing! This document covers development setup, coding standards, and the PR process.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/PricingFrontier/rustystats.git
cd rustystats

# Install pre-commit hooks
uv run pre-commit install

# Verify everything works
cargo test --workspace
uv run --extra dev pytest tests/python/ -v
```

## Project Architecture

```
rustystats/
├── crates/
│   ├── rustystats-core/   # Pure Rust: algorithms, solvers, statistics
│   └── rustystats/        # PyO3 bindings: thin bridge to Python
├── python/rustystats/     # Python: API, formula parsing, diagnostics
├── tests/python/          # Python test suite
└── pyproject.toml         # Python package config + tool settings
```

**Key principle**: Rust computes, Python orchestrates. Hot loops and linear algebra live in `rustystats-core`. Python owns the user-facing API and result presentation.

## Running Tests & Linters

```bash
# Rust
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings

# Python
uv run ruff check python/
uv run ruff format --check python/
uv run --extra dev pytest tests/python/ -v
```

## Commit Format

We use conventional-style commit messages:

```
<type>: <short description>

<optional body>

Co-Authored-By: <name> <email>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `release`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all linters and tests pass (CI runs automatically)
4. Open a PR against `main` with a clear description
5. Address review feedback
6. Squash-merge when approved

## Code Style

- **Rust**: Follow `rustfmt` defaults (edition 2021, max_width 100). Use `expect("reason")` instead of `unwrap()` in library code. Tests may use `unwrap()`.
- **Python**: Follow `ruff` configuration in `pyproject.toml`. Use modern type annotations (`X | None` not `Optional[X]`). All modules should have `from __future__ import annotations`.
- **Both**: Prefer clarity over cleverness. Add comments only where the logic isn't self-evident.
