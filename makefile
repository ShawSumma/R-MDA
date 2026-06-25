# RWKV-7 + diffusion, raw Vulkan. No Python, no GGML.
#   shader/*.comp -> build/*.spv   (glslc)
#   src/*.c        -> build/<name>  (cc, linked against vulkan)
# Incremental: a target rebuilds only when its sources (or vkc.h) change.

CC      := cc
CFLAGS  := -O2 -Isrc
LDFLAGS := -lvulkan -lm
GLSLC   := glslc

SHADERS := $(wildcard shader/*.comp)
SPV     := $(patsubst shader/%.comp,build/%.spv,$(SHADERS))

# host programs = every src/*.c with a main(). all src/*.h are shared deps.
SRCS    := $(wildcard src/*.c)
HDRS    := $(wildcard src/*.h)
BINS    := $(patsubst src/%.c,build/%,$(SRCS))

.PHONY: all shaders bins clean test
all: shaders bins

shaders: $(SPV)
bins:    $(BINS)

build/%.spv: shader/%.comp | build
	$(GLSLC) $< -o $@

build/%: src/%.c $(HDRS) | build
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

build:
	@mkdir -p build

# run all test_* binaries (depends on shaders being built)
test: all
	@for t in build/test_*; do echo "== $$t =="; $$t || exit 1; done

clean:
	rm -rf build
