.POSIX:
.PHONY: all clean test

CC = cc
CFLAGS = -std=gnu17 -O2 -Isrc -MMD -MP
LDLIBS = -lvulkan -lm
GLSLC = glslc --target-env=vulkan1.3

MODS = vkc util hash sort tokenizer vocab data model fwd bwd step ckpt
TESTS = test_ckpt test_grad test_matmul_f16 test_tokenizer
PROGS = bench generate probe_acts probe_coopmat probe_memory sample train train_tokenizer ${TESTS}
KERNELS != cd shader && echo *.comp

OBJS = ${MODS:%=build/obj/%.o}
BINS = ${PROGS:%=build/%}
SPV = ${KERNELS:%.comp=build/%.spv}

all: ${SPV} ${BINS}

build/%.spv: shader/%.comp
	@mkdir -p build
	${GLSLC} $< -o $@

build/obj/%.o: src/%.c
	@mkdir -p build/obj
	${CC} ${CFLAGS} -c $< -o $@

build/libmda.a: ${OBJS}
	ar -rcs $@ $^

build/%: src/%.c build/libmda.a
	${CC} ${CFLAGS} $< build/libmda.a ${LDLIBS} -o $@

-include ${MODS:%=build/obj/%.d} ${PROGS:%=build/%.d}

test: all
	@for t in ${TESTS}; do echo $$t; build/$$t || exit 1; done

clean:
	rm -rf build
