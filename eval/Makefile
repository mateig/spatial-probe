install:
	uv sync

fmt:
	uv run --no-sync ruff format --no-cache . && uv run --no-sync ruff check --no-cache --fix .

run:
	uv run --no-sync python -u -m scripts.run

inspect:
	uv run --no-sync python -u -m scripts.inspect
