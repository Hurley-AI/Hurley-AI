all: lint mypy test

lint:
	pylint hurley_agentic || true
	flake8 hurley_agentic tests || true
mypy:
	mypy hurley_agentic || true

test:
	python -m unittest discover -s tests

.PHONY: all lint mypy test
