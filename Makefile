# Compiler & Tools
VERILATOR = verilator
CXX = g++

# Directories
SRC_DIR = src
BUILD_DIR = obj_dir

# Find ALL .v files in SRC_DIR and all of its subdirectories
VERILOG_SRCS = $(shell find $(SRC_DIR) -name "*.v")

# Your C++ testbench / driver
CPP_SRC = $(SRC_DIR)/main.cpp

# Name of the top-level Verilog module
TOP_MODULE = chunk

# Target executable name
TARGET = run_sim
CUDA_ARCH ?= sm_75
CUDA_HOME ?= /usr/local/cuda-13.3
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOST_COMPILER ?= gcc
CUDA_FLAGS ?= -allow-unsupported-compiler
CUDA_XS ?= 4
CUDA_YS ?= 4
CUDA_DIMENSIONS = -DTRAIN_XS=$(CUDA_XS) -DTRAIN_YS=$(CUDA_YS)

# Verilator Flags
# -I$(SRC_DIR) tells Verilator where to search for included files
VERILATOR_FLAGS = -Wall --cc \
                  $(VERILOG_SRCS) \
                  --top-module $(TOP_MODULE) \
                  -GXS=64 -GYS=32 \
                  --exe $(CPP_SRC) \
                  -I$(SRC_DIR) \
                  --build \
                  -o $(TARGET)

.PHONY: all clean cuda cuda-train

all:
	$(VERILATOR) $(VERILATOR_FLAGS)

run: all
	./$(BUILD_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

cuda:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_DIMENSIONS) -ccbin=$(CUDA_HOST_COMPILER) -std=c++17 -O3 -arch=$(CUDA_ARCH) -Isrc -c src/cuda_fitness.cu -o $(BUILD_DIR)/cuda_fitness.o

cuda-train:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_DIMENSIONS) -ccbin=$(CUDA_HOST_COMPILER) -std=c++17 -O3 -arch=$(CUDA_ARCH) -Isrc \
		src/cuda_fitness.cu src/cuda_train.cu -o $(BUILD_DIR)/cuda_train