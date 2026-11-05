CC      ?= gcc
CFLAGS  ?= -O3 -g -std=c99 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?=
LIBS    ?=

UNAME_S := $(shell uname -s 2>/dev/null)
ifneq (,$(filter MINGW% MSYS% CYGWIN%,${UNAME_S}))
SOAK_LIBS := -lpsapi
endif
ifneq (,$(filter Linux Darwin FreeBSD GNU,${UNAME_S}))
LIBS += -lpthread
endif

SRC := $(wildcard src/*.c)
HDRS := $(wildcard src/*.h) h265dec.h
OBJ := $(patsubst src/%.c,build/%.o,$(SRC))
DEPS := $(OBJ:.o=.d)

all: h265dec_example

build:
	mkdir -p build

build/h265_cabac.o: src/h265_cabac.c $(HDRS) | build
	$(CC) $(CFLAGS) -fno-tree-vectorize -MMD -I. -c $< -o $@

build/%.o: src/%.c $(HDRS) | build
	$(CC) $(CFLAGS) -MMD -I. -c $< -o $@

libh265dec.a: $(OBJ)
	ar rcs $@ $^

h265dec_example: example_decoder.c libh265dec.a
	$(CC) $(CFLAGS) -I. $< libh265dec.a $(LDFLAGS) $(LIBS) -o $@

gen:
	sh tests/gen_streams.sh

# bit-exact against FFmpeg on the generated corpus at one, four and twelve
# slots.  Needs ffmpeg + ffprobe with libx265.
check: h265dec_example
	sh tests/gen_streams.sh
	THREADS=1 sh tests/compare.sh
	THREADS=4 sh tests/compare.sh
	THREADS=12 sh tests/compare.sh
	H265_WPP_THREADS=1 THREADS=12 sh tests/compare.sh
	sh tests/avcc_check.sh

fuzz: h265dec_example
	sh tests/fuzz.sh

bench: h265dec_example
	sh tests/bench.sh

clean:
	rm -rf build libh265dec.a h265dec_example h265dec_example.exe \
		testdata/tmp testdata/fuzz

-include $(DEPS)

.PHONY: all clean gen check fuzz bench
