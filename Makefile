SHELL := /bin/bash

MODEL_DIR ?= Trillim/BitNet-TRNQ
ADAPTER_DIR ?= Trillim/BitNet-GenZ-LoRA-TRNQ
TEST_PATTERN ?= *_test.py
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
ifeq ($(UNAME_M),x86_64)
LOCAL_PLATFORM := linux-x86_64
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
LOCAL_PLATFORM := linux-arm64
endif
else ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),x86_64)
LOCAL_PLATFORM := macos-x86_64
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
LOCAL_PLATFORM := macos-arm64
endif
endif

ci: test test-live

test test-live coverage coverage-html coverage-xml: bundle

bundle:
ifeq ($(LOCAL_PLATFORM),)
	@echo "Unsupported local platform: $(UNAME_S) $(UNAME_M)" >&2; exit 1
else
	uv run python -c "from scripts import build_wheels; build_wheels.clean_bin_dir(); build_wheels.copy_binaries('$(LOCAL_PLATFORM)')"
endif

test:
	MODEL_DIR="$(MODEL_DIR)" ADAPTER_DIR="$(ADAPTER_DIR)" uv run python -m unittest discover -s tests -p "$(TEST_PATTERN)"

test-live:
	@set -euo pipefail; \
	server_log=$$(mktemp); \
	uv run trillim serve "$(MODEL_DIR)" --voice >"$$server_log" 2>&1 & \
	server_pid=$$!; \
	cleanup() { \
		kill "$$server_pid" 2>/dev/null || true; \
		wait "$$server_pid" 2>/dev/null || true; \
		rm -f "$$server_log"; \
	}; \
	trap cleanup EXIT; \
	ready=0; \
	for _ in $$(seq 1 120); do \
		if uv run python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2).close()' >/dev/null 2>&1; then \
			ready=1; \
			break; \
		fi; \
		if ! kill -0 "$$server_pid" 2>/dev/null; then \
			cat "$$server_log"; \
			exit 1; \
		fi; \
		sleep 1; \
	done; \
	if [ "$$ready" -ne 1 ]; then \
		echo "Timed out waiting for live test server at http://127.0.0.1:8000" >&2; \
		cat "$$server_log"; \
		exit 1; \
	fi; \
	MODEL_DIR="$(MODEL_DIR)" ADAPTER_DIR="$(ADAPTER_DIR)" uv run python -m unittest tests.server_live_suite

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

.PHONY: ci bundle test test-live coverage coverage-html coverage-xml
