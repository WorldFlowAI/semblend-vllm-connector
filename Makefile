.PHONY: check lint test

check: lint test

lint:
	python3 -m ruff check .

test:
	python3 -m pytest -q
