.PHONY: install data run run-llm test lint

install:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

run:              ## template extractor + validation rules (no API key needed)
	python -m labcopilot.pipeline

run-llm:          ## adds Claude extraction; needs ANTHROPIC_API_KEY (responses cached in reports/llm_outputs)
	python -m labcopilot.pipeline --llm

test:
	pytest -q

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts
