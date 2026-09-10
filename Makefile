# Compiler & Tools
VERILATOR = verilator
CXX = g++

# Directories
SRC_DIR = src
BUILD_DIR = obj_dir
SIM_BUILD_DIR = obj_sim
TRAIN_BUILD_DIR = obj_train

# Find ALL .v files in SRC_DIR and all of its subdirectories
VERILOG_SRCS = $(shell find $(SRC_DIR) -name "*.v")

# Your C++ testbench / driver
CPP_SRC = $(SRC_DIR)/main.cpp
SIM_CPP_SRC = $(SRC_DIR)/sim_server.cpp

# Name of the top-level Verilog module
TOP_MODULE = chunk

# Target executable name
TARGET = run_sim
MAIN_XS ?= 8
MAIN_YS ?= 8
SIM_TARGET = sim_server
TRAIN_TARGET = trainer_server
SIM_XS ?= 4
SIM_YS ?= 8
TRAIN_XS ?= 4
TRAIN_YS ?= 8
CUDA_ARCH ?= sm_89
CUDA_HOME ?= /usr/local/cuda-13.3
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOST_COMPILER ?= gcc
CUDA_FLAGS ?= -allow-unsupported-compiler
CUDA_XS ?= 16
CUDA_YS ?= 16
CUDA_POPULATION_SIZE ?= 65536
CUDA_MUTATION_RATE ?= 0.04
CUDA_SIMULATION_CYCLES ?= 32
CUDA_ELITE_PERCENT ?= 20
CUDA_DIMENSIONS = -DTRAIN_XS=$(CUDA_XS) -DTRAIN_YS=$(CUDA_YS) \
	-DCUDA_POPULATION_SIZE=$(CUDA_POPULATION_SIZE) \
	-DCUDA_MUTATION_RATE=$(CUDA_MUTATION_RATE) \
	-DCUDA_SIMULATION_CYCLES=$(CUDA_SIMULATION_CYCLES) \
	-DCUDA_ELITE_PERCENT=$(CUDA_ELITE_PERCENT)

# Verilator Flags
# -I$(SRC_DIR) tells Verilator where to search for included files
VERILATOR_FLAGS = -Wall --cc \
                  $(VERILOG_SRCS) \
                  --top-module $(TOP_MODULE) \
				  -GXS=$(MAIN_XS) -GYS=$(MAIN_YS) \
                  --exe $(CPP_SRC) \
                  -I$(SRC_DIR) \
                  --build \
				  -CFLAGS "-DMAIN_XS=$(MAIN_XS) -DMAIN_YS=$(MAIN_YS)" \
                  -o $(TARGET)

SIM_VERILATOR_FLAGS = -Wall --cc \
				  $(VERILOG_SRCS) \
				  --top-module $(TOP_MODULE) \
				  -GXS=$(SIM_XS) -GYS=$(SIM_YS) \
				  --exe $(SIM_CPP_SRC) \
				  -I$(SRC_DIR) \
				  --Mdir $(SIM_BUILD_DIR) \
				  --build \
				  -CFLAGS "-DSIM_XS=$(SIM_XS) -DSIM_YS=$(SIM_YS)" \
				  -o $(SIM_TARGET)

TRAIN_VERILATOR_FLAGS = -Wall --cc \
				  $(VERILOG_SRCS) \
				  --top-module $(TOP_MODULE) \
				  -GXS=$(TRAIN_XS) -GYS=$(TRAIN_YS) \
				  --exe $(SRC_DIR)/trainer_server.cpp \
				  -I$(SRC_DIR) \
				  --Mdir $(TRAIN_BUILD_DIR) \
				  --build \
				  -CFLAGS "-DSIM_XS=$(TRAIN_XS) -DSIM_YS=$(TRAIN_YS)" \
				  -o $(TRAIN_TARGET)

.PHONY: all run sim sim-build train train-build clean cuda cuda-train cuda-train-server fpga-lint fpga-synth

all:
	$(VERILATOR) $(VERILATOR_FLAGS)

run: all
	./$(BUILD_DIR)/$(TARGET)

sim-build:
	$(VERILATOR) $(SIM_VERILATOR_FLAGS)

sim: sim-build
	./$(SIM_BUILD_DIR)/$(SIM_TARGET)

train-build:
	$(VERILATOR) $(TRAIN_VERILATOR_FLAGS)

train:
	$(MAKE) train-build
	./$(TRAIN_BUILD_DIR)/$(TRAIN_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(SIM_BUILD_DIR) $(TRAIN_BUILD_DIR)

cuda:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_DIMENSIONS) -ccbin=$(CUDA_HOST_COMPILER) -std=c++17 -O3 -arch=$(CUDA_ARCH) -Isrc -c src/cuda_fitness.cu -o $(BUILD_DIR)/cuda_fitness.o

cuda-train:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_DIMENSIONS) -ccbin=$(CUDA_HOST_COMPILER) -std=c++17 -O3 -arch=$(CUDA_ARCH) -Isrc \
		src/cuda_fitness.cu src/cuda_train.cu -o $(BUILD_DIR)/cuda_train

cuda-train-server:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_DIMENSIONS) -ccbin=$(CUDA_HOST_COMPILER) -std=c++17 -O3 -arch=$(CUDA_ARCH) -Isrc \
		-DUSE_CUDA -x cu src/cuda_fitness.cu src/trainer_server.cpp -o $(BUILD_DIR)/cuda_trainer_server

fpga-lint:
	verilator --lint-only -Wall $(VERILOG_SRCS) --top-module chunk \
		-GXS=$(MAIN_XS) -GYS=$(MAIN_YS) -I$(SRC_DIR)

fpga-synth:
	yosys -p "read_verilog -sv $(VERILOG_SRCS); synth -top chunk"