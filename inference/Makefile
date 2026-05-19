install:
	uv sync

run:
	uv run --no-sync python -u -m scripts.run

fmt:
	uv run --no-sync ruff format --no-cache . && uv run --no-sync ruff check --no-cache --fix .
