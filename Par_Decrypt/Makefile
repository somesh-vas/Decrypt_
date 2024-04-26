# Define the compiler
CC=nvcc

# Define compiler flags
CFLAGS=-arch=sm_86
CUDAFLAGS=-Xcompiler -Wall,-Wextra

# Define libraries
LIBS=-lcudart -lm

# Define target executable name
TARGET=decrypt

# List CUDA and C source files
CU_SOURCES=$(wildcard *.cu)
C_SOURCES=$(wildcard *.c)

# Define object files from sources
CU_OBJECTS=$(CU_SOURCES:.cu=.o)
C_OBJECTS=$(C_SOURCES:.c=.o)

# Combine all objects
OBJECTS=$(CU_OBJECTS) $(C_OBJECTS)

# Default target
all: $(TARGET) run

# Link the program
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LIBS)

# Compile CUDA source files
%.o: %.cu
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

# Target to run the program
run: $(TARGET)
	./$(TARGET) > ../error_position_gpu.txt

# Clean target to remove generated files
clean:
	rm -f $(OBJECTS) $(TARGET) decrypt *.o ../error_position_gpu.txt

# Phony targets
.PHONY: all clean run
