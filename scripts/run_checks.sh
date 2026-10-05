#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q src
python -m pytest -q
python scripts/check_public.py

