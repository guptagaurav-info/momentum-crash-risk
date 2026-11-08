PY ?= python
export PYTHONPATH := src

.PHONY: all ingest run test
all: ingest run
ingest:
	$(PY) -m mcr.ingest
run:
	$(PY) -m mcr.pipeline
test:
	$(PY) -m pytest -q tests
