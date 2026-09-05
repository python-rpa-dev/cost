PY = .venv-linux/bin/python

.PHONY: install test lint typecheck check

install:
	$(PY) -m pip install -e ".[dev]" vulture

test:
	$(PY) -m pytest

lint:
	.venv-linux/bin/ruff check .
	.venv-linux/bin/vulture

typecheck:
	.venv-linux/bin/pyright token_tracker.py tests

check: lint typecheck test
