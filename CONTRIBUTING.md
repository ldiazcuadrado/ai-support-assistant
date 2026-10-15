# Contributing

Bug reports and pull requests are welcome. For anything larger than a small fix, please open an
issue first so the approach can be agreed before you invest time.

## Development setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pre-commit install
```

## Checks

Every change must pass the same checks as CI:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The test suite never touches the network. Use `FakeLLM` and the fixtures in `tests/conftest.py`
rather than mocking the Anthropic SDK in new tests. Coverage must stay at or above 85%.

## Guidelines

- Keep modules small and typed; `mypy --strict` must pass.
- New behaviour needs tests. Bug fixes need a test that fails without the fix.
- Update `CHANGELOG.md` under `Unreleased` for user-visible changes.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages, for
  example `feat(retrieval): add cross-encoder reranker`.
- Do not commit secrets. Configuration belongs in environment variables; add new ones to
  `.env.example` and the README table.

## Reporting security issues

Please do not open a public issue for vulnerabilities. Contact the maintainers privately instead.
