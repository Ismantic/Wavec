# THUCNews acquisition and conversion. Raw and derived data are reproducible
# outputs and are ignored by git.

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(HERE)..
-include $(ROOT)/local.mk
PY ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3))

CORPUS := $(HERE)derived/THUCNews.sentences.txt

.PHONY: all download process status help

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)Makefile

all: $(CORPUS)

download:
	$(PY) $(HERE)download.py

process: $(CORPUS)

$(CORPUS):
	$(PY) $(HERE)download.py
	$(PY) $(HERE)process.py

status:
	@$(PY) $(HERE)download.py --status
