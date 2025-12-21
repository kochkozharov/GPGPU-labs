# Makefile for lab4 benchmarks

NVCC = nvcc
CXX = g++

# Compiler flags
# Автоматическое определение compute capability (можно переопределить)
COMPUTE_CAP ?= sm_75
NVCC_FLAGS = -O3 -arch=$(COMPUTE_CAP) -std=c++14
CXX_FLAGS = -O3 -std=c++14

# Directories
SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = target

# Source files
CUDA_SOURCES = lab4_benchmark.cu
CPU_SOURCES = lab4_cpu.cpp matrix_generator.cpp

# Object files
CUDA_OBJECTS = $(OBJ_DIR)/lab4_benchmark.o
CPU_OBJECTS = $(OBJ_DIR)/lab4_cpu.o $(OBJ_DIR)/matrix_generator.o

# Executables
CUDA_BIN = $(BIN_DIR)/lab4_benchmark
CPU_BIN = $(BIN_DIR)/lab4_cpu

# Default target
all: $(CUDA_BIN) $(CPU_BIN)

# Create directories
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# CUDA benchmark
$(CUDA_BIN): lab4_benchmark.cu matrix_generator.cpp matrix_generator.h | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(CUDA_BIN) lab4_benchmark.cu matrix_generator.cpp

# CPU benchmark
$(CPU_BIN): lab4_cpu.cpp matrix_generator.cpp matrix_generator.h | $(BIN_DIR)
	$(CXX) $(CXX_FLAGS) -o $(CPU_BIN) lab4_cpu.cpp matrix_generator.cpp

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) benchmark_results

# Run benchmarks (requires nvprof)
benchmark: all
	chmod +x run_benchmark.sh
	./run_benchmark.sh

# Quick test
test: all
	@echo "Testing CPU version..."
	$(CPU_BIN) 10 42
	@echo ""
	@echo "Testing CUDA version..."
	$(CUDA_BIN) 10 16 32 32 32 42

.PHONY: all clean benchmark test

