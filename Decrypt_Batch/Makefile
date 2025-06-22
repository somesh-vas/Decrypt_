# ------------------------------
# Optimised CUDA + CPU Makefile
# ------------------------------
ARCH        ?= sm_86
NVCC        ?= nvcc
HOSTCC      ?= gcc
TARGET      ?= batch_decrypt
RUN_ARGS    ?= > Output/error_position_gpu_Batch.txt

COMMON_FLAGS := -O3 --use_fast_math -Xcompiler "-Wall,-Wextra,-g"
NVCCFLAGS    := -arch=$(ARCH) $(COMMON_FLAGS) -lineinfo
CFLAGS       := -std=c11 -O3 -Wall -Wextra
LIBS         := -lcudart -lnvToolsExt -lm

INCLUDE_DIR  := include
INC_FLAGS    := -I$(INCLUDE_DIR)

OBJ_DIR      := Obj

# Source files
CU_SRCS := src/cuda/Decrypt.cu
C_SRCS  := src/host/util.c src/host/gf.c src/host/root.c

# Object files
CU_OBJS := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))
C_OBJS  := $(patsubst %.c,  $(OBJ_DIR)/%.o, $(C_SRCS))
OBJS    := $(CU_OBJS) $(C_OBJS)

# Default target
.PHONY: all
all: $(TARGET)

# Link final executable
$(TARGET): $(OBJS)
	$(NVCC) -arch=$(ARCH) $(INC_FLAGS) $(COMMON_FLAGS) $^ -o $@ $(LIBS)

# CUDA compile rule
$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INC_FLAGS) -c $< -o $@

# Host C compile rule
$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(HOSTCC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

# Ensure Obj directory exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/src/cuda
	mkdir -p $(OBJ_DIR)/src/host

# Run
.PHONY: run
run: $(TARGET)
	./$(TARGET) $(RUN_ARGS)

# Clean
.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR)
	rm -f Output/*
