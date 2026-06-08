CC ?= cc
CFLAGS_COMMON = -O3 -Wall -Wextra -std=c11 -fPIC -shared
INCLUDES = -Icsrc
SRCDIR = csrc
BUILDDIR = build
LIBNAME_BASE = libsimdkernels

# Source files (all backends compiled; preprocessor selects the right one)
SOURCES = $(SRCDIR)/binary_gemm.c \
          $(SRCDIR)/backend_scalar.c \
          $(SRCDIR)/backend_neon.c \
          $(SRCDIR)/backend_avx2.c

TEST_SRC = $(SRCDIR)/test_kernels.c

# ================================================================
# Auto-detect architecture
# ================================================================
UNAME_M := $(shell uname -m)

ifeq ($(BACKEND),avx2)
    # Forced AVX2
    CFLAGS_ARCH = -mavx2 -mfma -DFORCE_AVX2
    LIBNAME = $(LIBNAME_BASE).so
else ifeq ($(BACKEND),neon)
    # Forced NEON
    CFLAGS_ARCH = -DFORCE_NEON
    LIBNAME = $(LIBNAME_BASE).so
else ifeq ($(BACKEND),scalar)
    # Forced scalar
    CFLAGS_ARCH = -DFORCE_SCALAR
    LIBNAME = $(LIBNAME_BASE).so
else
    # Auto-detect
    ifneq (,$(filter $(UNAME_M),x86_64 amd64))
        CFLAGS_ARCH = -mavx2 -mfma -march=native
        LIBNAME = $(LIBNAME_BASE).so
    else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
        CFLAGS_ARCH = -march=native
        LIBNAME = $(LIBNAME_BASE).so
        ifeq ($(shell uname -s),Darwin)
            LIBNAME = $(LIBNAME_BASE).dylib
        endif
    else
        CFLAGS_ARCH = -DFORCE_SCALAR
        LIBNAME = $(LIBNAME_BASE).so
    endif
endif

CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_ARCH)

.PHONY: all clean test install info

all: info $(BUILDDIR)/$(LIBNAME)

info:
	@echo "Architecture: $(UNAME_M)"
	@echo "Backend flags: $(CFLAGS_ARCH)"
	@echo "Output: $(BUILDDIR)/$(LIBNAME)"

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/$(LIBNAME): $(SOURCES) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SOURCES) -lm

test: $(BUILDDIR)/test_kernels
	./$(BUILDDIR)/test_kernels

$(BUILDDIR)/test_kernels: $(TEST_SRC) $(SOURCES) | $(BUILDDIR)
	$(CC) -O3 -Wall -Wextra -std=c11 $(CFLAGS_ARCH) $(INCLUDES) \
		-o $@ $(TEST_SRC) \
		$(SRCDIR)/binary_gemm.c \
		$(SRCDIR)/backend_scalar.c \
		$(SRCDIR)/backend_neon.c \
		$(SRCDIR)/backend_avx2.c \
		-lm

clean:
	rm -rf $(BUILDDIR)

install: $(BUILDDIR)/$(LIBNAME)
	cp $(BUILDDIR)/$(LIBNAME) src/simd_kernels/
