.PHONY: check lint test

check: lint test

lint:
	python3 -m ruff check .
	python3 -m ruff format --check src tests

test:
	python3 -m pytest -q
