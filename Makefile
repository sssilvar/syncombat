# A Makefile that runs the important commands for the project
# This should include running the tests, linting, and building the project
.PHONY: test lint install all
.DEFAULT_GOAL := help

# Run the tests, coverage and coverage badge for the syncombat package
test:
	poetry run pytest --cov=syncombat --cov-report=term-missing --mpl-generate-path=baseline tests/
	poetry run coverage-badge -o badges/coverage.svg -f

# Lint the project
lint:
	poetry run ruff check syncombat tests --line-length 120

# Install the project using poetry
install:
	poetry install

# Build the project
all: install lint test

clean-build:
	rm -rf dist/
	rm -rf build/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -rf {} +


help:
	@echo "test - run the tests, coverage and coverage badge for the syncombat package"
	@echo "lint - lint the project"
	@echo "install - install the project using poetry"
	@echo "all - build the project"
