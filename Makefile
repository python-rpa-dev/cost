PY = .venv/bin/python

.PHONY: install test lint check

install:
	$(PY) -m pip install -e ".[dev]" vulture

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/vulture

check: lint test
