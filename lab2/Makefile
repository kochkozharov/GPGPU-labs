NVCC        = nvcc
CXX         = g++

NVCC_FLAGS  = -Werror cross-execution-space-call -lm -std=c++17 -O3
CXX_FLAGS   = -std=c++17 -O3

INCLUDES    = -I. -I../utils

TARGET_DIR  = target
SRC_DIR     = .

TARGET_BENCH    = $(TARGET_DIR)/benchmark
TARGET_GEN      = $(TARGET_DIR)/generate_test_images
TARGET_LAB2     = $(TARGET_DIR)/lab2

SRC_BENCH   = $(SRC_DIR)/benchmark.cu
SRC_GEN     = $(SRC_DIR)/generate_test_images.cpp
SRC_LAB2    = $(SRC_DIR)/lab2.cu
SRC_KERNEL  = $(SRC_DIR)/prewitt_kernel.cu
SRC_UTILS   = $(SRC_DIR)/image_utils.cpp
SRC_CPU     = $(SRC_DIR)/prewitt_cpu.cpp

OBJ_BENCH     = $(TARGET_DIR)/benchmark.o
OBJ_GEN       = $(TARGET_DIR)/generate_test_images.o
OBJ_LAB2      = $(TARGET_DIR)/lab2.o
OBJ_KERNEL    = $(TARGET_DIR)/prewitt_kernel.o
OBJ_UTILS     = $(TARGET_DIR)/image_utils.o
OBJ_CPU       = $(TARGET_DIR)/prewitt_cpu.o

all: $(TARGET_BENCH) $(TARGET_GEN) $(TARGET_LAB2)

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(OBJ_KERNEL): $(SRC_KERNEL) prewitt_kernel.h | $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(SRC_KERNEL) -o $@

$(OBJ_UTILS): $(SRC_UTILS) image_utils.h | $(TARGET_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $(SRC_UTILS) -o $@

$(OBJ_CPU): $(SRC_CPU) prewitt_cpu.h image_utils.h | $(TARGET_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $(SRC_CPU) -o $@

$(OBJ_BENCH): $(SRC_BENCH) prewitt_kernel.h prewitt_cpu.h image_utils.h | $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(SRC_BENCH) -o $@

$(OBJ_GEN): $(SRC_GEN) image_utils.h | $(TARGET_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $(SRC_GEN) -o $@

$(OBJ_LAB2): $(SRC_LAB2) | $(TARGET_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(SRC_LAB2) -o $@

$(TARGET_BENCH): $(OBJ_BENCH) $(OBJ_KERNEL) $(OBJ_UTILS) $(OBJ_CPU)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

$(TARGET_GEN): $(OBJ_GEN) $(OBJ_UTILS)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $@ $^

$(TARGET_LAB2): $(OBJ_LAB2) $(OBJ_KERNEL) $(OBJ_UTILS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -rf $(TARGET_DIR)

.PHONY: all clean benchmark generate lab2
benchmark: $(TARGET_BENCH)
generate: $(TARGET_GEN)
lab2: $(TARGET_LAB2)

