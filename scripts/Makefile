ROOT      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
BUILD_DIR := $(ROOT)/build
PREPARE   := $(ROOT)/prepare
NPROC     ?= 4
THREADS   ?= 8
OUTPUT    ?= $(ROOT)/model.vec
K         ?= 100
MAX_ITER  ?= 50
TOPN      ?= 20
MINFREQ   ?= 10
MINLEN    ?= 2
FILTERED  := $(ROOT)/model.filter.vec
CLUSTERS  := $(ROOT)/clusters.txt

SEG_FILE  := $(PREPARE)/News.cut.txt

.PHONY: cut fit filter kmeans clean

cut:
	bash $(ROOT)/scripts/cut.sh $(PREPARE)/News.documents.txt $(SEG_FILE) $(NPROC)

fit:
	@[ -f $(SEG_FILE) ] || $(MAKE) cut
	$(BUILD_DIR)/wavec \
		-dim 100 -window 8 -mincount 5 \
		-threads $(THREADS) -iter 5 -sample 1e-3 \
		$(SEG_FILE) $(OUTPUT)

filter:
	python3 $(ROOT)/scripts/filter_vec.py $(SEG_FILE) $(OUTPUT) $(FILTERED) \
		--minfreq $(MINFREQ) --minlen $(MINLEN)

kmeans:
	@[ -f $(FILTERED) ] || $(MAKE) filter
	$(BUILD_DIR)/kmeans $(FILTERED) $(K) $(MAX_ITER) $(TOPN) --output $(CLUSTERS) --export $(ROOT)/clusters.map

clean:
	rm -f $(PREPARE)/News.cut.txt
