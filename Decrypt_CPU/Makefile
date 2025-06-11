# Define compiler and flags
CC=gcc
CFLAGS=-O3 -march=native -mtune=native -Wall -Wextra -Wno-unused-function -Wno-unused-parameter -Wno-sign-compare -Wno-memset-elt-size -I. -I../include
LIBS= -ldl -lm

# Define target executable name
TARGET=decrypt

# Source files to compile
SOURCES=Decrypt.c gf.c bm.c root.c util.c

# Object files to generate
OBJECTS=$(SOURCES:.c=.o)

# Default target
all: $(TARGET) run

# Rule to link the program
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LIBS)

# Rule to compile the source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Target to run the program
run: $(TARGET)
	./$(TARGET) > ../error_position_cpu.txt

# Clean target to remove generated files
clean:
	rm -f $(OBJECTS) $(TARGET) *.o ../error_position_cpu.txt

# Phony targets
.PHONY: all clean run
