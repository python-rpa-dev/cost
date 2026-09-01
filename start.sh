#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=.venv-linux

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import token_tracker, filelock" >/dev/null 2>&1; then
    echo "Installing dependencies..."
    "$VENV/bin/python" -m pip install --quiet -e ".[dev]" vulture
fi

exec "$VENV/bin/token-tracker" "$@"
