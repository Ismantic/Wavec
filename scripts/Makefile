ROOT      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
BUILD_DIR := $(ROOT)/build
PREPARE   := $(ROOT)/prepare
-include $(ROOT)/local.mk
NPROC     ?= 4
THREADS   ?= 8
CORPUS    ?= $(ROOT)/data/derived/THUCNews.sentences.txt
WAPIC_MODEL ?=
PY       ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3))
OUTPUT    ?= $(ROOT)/model.vec
DIM       ?= 100
WINDOW    ?= 8
MINCOUNT  ?= 5
ITER      ?= 5
SAMPLE    ?= 1e-3
K         ?= 100
MAX_ITER  ?= 50
TOPN      ?= 20
MINFREQ   ?= 10
MINLEN    ?= 2
FILTERED  := $(ROOT)/model.filter.vec
CLUSTERS  := $(ROOT)/clusters.txt

SEG_FILE  := $(PREPARE)/News.cut.txt

.PHONY: all data build cut fit filter kmeans clean

all: kmeans

data:
	$(MAKE) -C $(ROOT)/data all PY=$(PY)

build:
	cmake -S $(ROOT) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j$(NPROC)

$(CORPUS):
	$(MAKE) -C $(ROOT)/data all PY=$(PY)

cut: $(CORPUS)
	PYTHON=$(PY) bash $(ROOT)/scripts/cut.sh $(CORPUS) $(SEG_FILE) $(NPROC) $(WAPIC_MODEL)

fit: build
	@[ -f $(SEG_FILE) ] || $(MAKE) cut
	$(BUILD_DIR)/wavec \
		-dim $(DIM) -window $(WINDOW) -mincount $(MINCOUNT) \
		-threads $(THREADS) -iter $(ITER) -sample $(SAMPLE) \
		$(SEG_FILE) $(OUTPUT)

filter:
	@[ -f $(OUTPUT) ] || $(MAKE) fit
	$(PY) $(ROOT)/scripts/filter_vec.py $(SEG_FILE) $(OUTPUT) $(FILTERED) \
		--minfreq $(MINFREQ) --minlen $(MINLEN)

kmeans:
	@[ -f $(FILTERED) ] || $(MAKE) filter
	$(BUILD_DIR)/kmeans $(FILTERED) $(K) $(MAX_ITER) $(TOPN) --output $(CLUSTERS) --export $(ROOT)/clusters.map

clean:
	rm -f $(PREPARE)/News.cut.txt
