NVCC        = nvcc
CXX         = g++

NVCC_FLAGS  = -Werror cross-execution-space-call -lm
CXX_FLAGS   = -std=c++17

INCLUDES    = -I. -I../utils

TARGET_DIR  = target
SRC_DIR     = .

UTIL_HEADERS := $(wildcard ../utils/*.h)

TARGET_CUDA     = $(TARGET_DIR)/lab1_cuda
TARGET_CUDA_MAIN    = $(TARGET_DIR)/lab1_cuda_main
TARGET_BENCH    = $(TARGET_DIR)/lab1_bench
TARGET_NO_CUDA  = $(TARGET_DIR)/lab1_no_cuda

SRC_CUDA    = $(SRC_DIR)/lab1.cu
SRC_CUDA_MAIN    = $(SRC_DIR)/lab1_main.cu
SRC_BENCH   = $(SRC_DIR)/lab1_bench.cu
SRC_NO_CUDA = $(SRC_DIR)/lab1_no_cuda.cpp

OBJ_CUDA      = $(TARGET_DIR)/lab1.o
OBJ_CUDA_MAIN      = $(TARGET_DIR)/lab1_main.o
OBJ_BENCH     = $(TARGET_DIR)/lab1_bench.o
OBJ_NO_CUDA   = $(TARGET_DIR)/lab1_no_cuda.o

all: $(TARGET_CUDA) $(TARGET_BENCH) $(TARGET_NO_CUDA)

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(OBJ_CUDA): $(SRC_CUDA) lab1.h $(UTIL_HEADERS) | $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(SRC_CUDA) -o $@

$(OBJ_CUDA_MAIN): $(SRC_CUDA_MAIN) lab1.h $(UTIL_HEADERS) | $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(SRC_CUDA_MAIN) -o $@

$(OBJ_BENCH): $(SRC_BENCH) lab1.h $(UTIL_HEADERS) | $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(SRC_BENCH) -o $@

$(OBJ_NO_CUDA): $(SRC_NO_CUDA) lab1.h $(UTIL_HEADERS) | $(TARGET_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $(SRC_NO_CUDA) -o $@

$(TARGET_CUDA): $(OBJ_CUDA)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

$(TARGET_CUDA_MAIN): $(OBJ_CUDA)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

$(TARGET_BENCH): $(OBJ_BENCH) $(OBJ_CUDA)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

$(TARGET_NO_CUDA): $(OBJ_NO_CUDA)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -rf $(TARGET_DIR)

.PHONY: all clean lab1_cuda lab1_bench lab1_no_cuda
lab1_cuda: $(TARGET_CUDA)
lab1_bench: $(TARGET_BENCH)
lab1_no_cuda: $(TARGET_NO_CUDA)
