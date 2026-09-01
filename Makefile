PY = .venv-linux/bin/python

.PHONY: install test lint check

install:
	$(PY) -m pip install -e ".[dev]" vulture

test:
	$(PY) -m pytest

lint:
	.venv-linux/bin/ruff check .
	.venv-linux/bin/vulture

check: lint test
