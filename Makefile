SHELL := /bin/bash

MODEL_DIR ?= Trillim/BitNet-TRNQ
ADAPTER_DIR ?= Trillim/BitNet-GenZ-LoRA-TRNQ
TEST_PATTERN ?= *_test.py

ci: test

test:
	MODEL_DIR="$(MODEL_DIR)" ADAPTER_DIR="$(ADAPTER_DIR)" uv run python -m unittest discover -s tests -p "$(TEST_PATTERN)"

coverage:
	MODEL_DIR="$(MODEL_DIR)" ADAPTER_DIR="$(ADAPTER_DIR)" uv run --with coverage python -m coverage run --source=src/trillim -m unittest discover -s tests -p "$(TEST_PATTERN)"
	uv run --with coverage python -m coverage report -m

coverage-html:
	MODEL_DIR="$(MODEL_DIR)" ADAPTER_DIR="$(ADAPTER_DIR)" uv run --with coverage python -m coverage run --source=src/trillim -m unittest discover -s tests -p "$(TEST_PATTERN)"
	uv run --with coverage python -m coverage html
	uv run --with coverage python -m coverage report -m

coverage-xml:
	MODEL_DIR="$(MODEL_DIR)" ADAPTER_DIR="$(ADAPTER_DIR)" uv run --with coverage python -m coverage run --source=src/trillim -m unittest discover -s tests -p "$(TEST_PATTERN)"
	uv run --with coverage python -m coverage xml
	uv run --with coverage python -m coverage report -m

.PHONY: ci test coverage coverage-html coverage-xml
