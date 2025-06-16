# # ------------------------------
# # Optimised CUDA + CPU Makefile
# # ------------------------------
# ARCH        ?= sm_86
# NVCC        ?= nvcc
# HOSTCC      ?= gcc
# TARGET      ?= batch_decrypt
# RUN_ARGS    ?= > Output/error_position_gpu_batch.txt

# COMMON_FLAGS := -O3 --use_fast_math -Xcompiler "-Wall,-Wextra,-g"
# NVCCFLAGS    := -arch=$(ARCH) $(COMMON_FLAGS) -lineinfo
# CFLAGS       := -std=c11 -O3 -Wall -Wextra
# LIBS         := -lcudart -lm

# INCLUDE_DIR  := include
# INC_FLAGS    := -I$(INCLUDE_DIR)

# OBJ_DIR      := Obj

# # Source files
# CU_SRCS := src/cuda/Decrypt.cu
# C_SRCS  := src/host/util.c src/host/gf.c src/host/root.c

# # Object files
# CU_OBJS := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))
# C_OBJS  := $(patsubst %.c,  $(OBJ_DIR)/%.o, $(C_SRCS))
# OBJS    := $(CU_OBJS) $(C_OBJS)

# # Default target
# .PHONY: all
# all: $(TARGET)

# # Link final executable
# $(TARGET): $(OBJS)
# 	$(NVCC) -arch=$(ARCH) $(INC_FLAGS) $(COMMON_FLAGS) $^ -o $@ $(LIBS)

# # CUDA compile rule
# $(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
# 	$(NVCC) $(NVCCFLAGS) $(INC_FLAGS) -c $< -o $@

# # Host C compile rule
# $(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
# 	$(HOSTCC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

# # Ensure Obj directory exists
# $(OBJ_DIR):
# 	mkdir -p $(OBJ_DIR)/src/cuda
# 	mkdir -p $(OBJ_DIR)/src/host

# # Run
# .PHONY: run
# run: $(TARGET)
# 	./$(TARGET) $(RUN_ARGS)

# # Clean
# .PHONY: clean
# clean:
# 	rm -f $(TARGET)
# 	rm -rf $(OBJ_DIR)
# --------------------------------------------
#  CUDA Reed-Solomon / BM pipeline – Makefile
#  (targets SM 8.6 / RTX 3050 Laptop GPU)
# --------------------------------------------

# ── Toolchain & target ───────────────────────────────────────────────────────
ARCH     ?= sm_86                    # override on the command line if needed
NVCC     ?= nvcc
HOSTCC   ?= gcc
TARGET   ?= batch_decrypt
RUN_ARGS ?= > Output/error_position_gpu_batch.txt

# ── Compilation flags ───────────────────────────────────────────────────────
COMMON_FLAGS := -O3 --use_fast_math -Xcompiler "-Wall,-Wextra,-g"
#  • --generate-line-info  → keeps Nsight source correlation
#  • -maxrregcount=$(MAXREG) lets you sweep 64 / 80 without editing the file
MAXREG       ?= 80
NVCCFLAGS    := -arch=$(ARCH) $(COMMON_FLAGS) --generate-line-info          \
                -Xptxas=-v,-dlcm=ca,-maxrregcount=$(MAXREG)
CFLAGS       := -std=c11 -O3 -Wall -Wextra
LIBS         := -lcudart -lm

# ── Include path ────────────────────────────────────────────────────────────
INCLUDE_DIR  := include
INC_FLAGS    := -I$(INCLUDE_DIR)

# ── Directory for intermediate objects ──────────────────────────────────────
OBJ_DIR := Obj

# ── Source files ────────────────────────────────────────────────────────────
#  (only Decrypt.cu changed between v6 → v7; paths stay identical)
CU_SRCS := src/cuda/Decrypt.cu
C_SRCS  := src/host/util.c src/host/gf.c src/host/root.c

# ── Derived object lists ────────────────────────────────────────────────────
CU_OBJS := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))
C_OBJS  := $(patsubst %.c,  $(OBJ_DIR)/%.o, $(C_SRCS))
OBJS    := $(CU_OBJS) $(C_OBJS)

# ────────────────────────────────────────────────────────────────────────────
#  Build rules
# ────────────────────────────────────────────────────────────────────────────

.PHONY: all
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJS)
	$(NVCC) -arch=$(ARCH) $(INC_FLAGS) $(COMMON_FLAGS) $^ -o $@ $(LIBS)

# CUDA compile rule
$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INC_FLAGS) -c $< -o $@

# Host-side C compile rule
$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(HOSTCC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

# Ensure the object directory hierarchy exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/src/cuda
	mkdir -p $(OBJ_DIR)/src/host

# ── Convenience targets ─────────────────────────────────────────────────────
.PHONY: run
run: $(TARGET)
	./$(TARGET) $(RUN_ARGS)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR)

# ---------------------------------------------------------------------------
#  Usage notes
#
#  • To profile different register ceilings (Opt-8) just do:
#         make MAXREG=64   # or 80
#
#  • For verbose PTXAS statistics add  “PTXAS=--verbose” on the command line.
#
#  • If you need to try an alternative block size or BM thread count at
#    compile-time, pass e.g.  “make NVCCFLAGS+=' -DBM_THREADS=128 ' ”.
#
# ---------------------------------------------------------------------------
