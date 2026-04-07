ROOT      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
BUILD_DIR := $(ROOT)/build
PREPARE   := $(ROOT)/prepare
NPROC     ?= 4
THREADS   ?= 6
OUTPUT    ?= $(ROOT)/model.vec

SEG_FILE  := $(PREPARE)/News.cut.txt

.PHONY: cut fit clean

cut:
	bash $(ROOT)/scripts/cut.sh $(PREPARE)/News.sentences.txt $(SEG_FILE) $(NPROC)

fit:
	@[ -f $(SEG_FILE) ] || $(MAKE) cut
	$(BUILD_DIR)/wavec \
		-dim 100 -window 5 -mincount 5 \
		-threads $(THREADS) -iter 5 -sample 1e-3 \
		$(SEG_FILE) $(OUTPUT)

clean:
	rm -f $(PREPARE)/News.cut.txt
