#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x .venv/bin/token-tracker ]; then
    echo "No venv found. Run: python3 -m venv .venv && make install" >&2
    exit 1
fi

exec .venv/bin/token-tracker "$@"
