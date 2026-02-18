.PHONY: lint typecheck test check format

lint:
	ruff check src/bigdance/
	ruff format --check src/bigdance/

typecheck:
	mypy src/bigdance/

test:
	pytest --cov=bigdance tests/

check: lint typecheck test

format:
	ruff check --fix src/bigdance/
	ruff format src/bigdance/
