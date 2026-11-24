set shell := ["bash", "-euo", "pipefail", "-c"]

default: setup

# Install the dependencies on your machine (add your own with `uv add`).
setup:
    uv sync --locked

# Start the viewer at http://localhost:${APP_PORT:-8000}, and wait until it's up.
up:
    docker compose up -d --wait
    @echo "Viewer: http://localhost:${APP_PORT:-8000}"

# Stop the viewer.
down:
    docker compose down -v

# Check the viewer's code (app/) with Ruff and Pyrefly.
lint:
    uv run ruff check
    uv run ruff format --check
    uv run pyrefly check

# Write what the rules found for every company to runs/facts.jsonl.
facts:
    uv run python -m summariser facts

# Write summaries/<version>.jsonl. Versions: v0-template, v1-raw, v2-grounded. Add --labelled to run only the
# 22 hand-labelled companies, or --dry-run to save prompts without calling the model.
generate version *args:
    uv run python -m summariser generate {{version}} {{args}}

# Score every version in summaries/ and write eval/results.md.
eval:
    uv run python -m summariser eval

# Test the evaluator and the LLM client (no API key needed).
test:
    uv run pytest -q
