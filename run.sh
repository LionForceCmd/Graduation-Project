#!/usr/bin/env sh
# Linux/macOS:  ./run.sh            — без языковой модели
#               ./run.sh ollama     — с Ollama (модель: OLLAMA_MODEL, по умолчанию qwen3.5:latest)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
export LLM_PROVIDER="${1:-mock}"
exec .venv/bin/python -m uvicorn app.main:app --port 4000
