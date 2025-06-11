# ------------------------------
# Optimised CUDA + CPU Makefile
# ------------------------------
# Usage:  make -j4           # parallel build
#         make run           # build + run
#         make clean         # remove artefacts
# -------------------------------------------------

# ─── User‑tunable section ─────────────────────────
ARCH        ?= sm_86            # target GPU architecture
NVCC        ?= nvcc             # CUDA compiler
HOSTCC      ?= gcc              # host‑only C sources (if any)
TARGET      ?= decrypt1          # final executable name
RUN_ARGS    ?= > ../error_position_gpu_single.txt

# ─── Flags ────────────────────────────────────────
COMMON_FLAGS := -O3 --use_fast_math -Xcompiler "-Wall,-Wextra,-g"
NVCCFLAGS    := -arch=$(ARCH) $(COMMON_FLAGS) -lineinfo \
               -Xptxas -v    # show register & shared‑mem usage
CFLAGS       := -std=c11 -O3 -Wall -Wextra 

LIBS         := -lcudart -lm

# ─── Source / object discovery ───────────────────
CU_SRCS := $(wildcard *.cu)
C_SRCS  := $(wildcard *.c)
OBJS    := $(CU_SRCS:.cu=.o) $(C_SRCS:.c=.o)

# ─── Default target ──────────────────────────────
.PHONY: all
all: $(TARGET)

# ─── Link step ───────────────────────────────────
$(TARGET): $(OBJS)
	$(NVCC) -arch=$(ARCH) $(COMMON_FLAGS) $^ -o $@ $(LIBS)

# ─── Compile rules ───────────────────────────────
#  CUDA → object
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

#  C (host) → object
%.o: %.c
	$(HOSTCC) $(CFLAGS) -c $< -o $@

# ─── Convenience targets ─────────────────────────
.PHONY: run
run: $(TARGET)
	./$(TARGET) $(RUN_ARGS)

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
