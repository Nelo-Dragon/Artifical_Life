# Compiler & Tools
VERILATOR = verilator
CXX = g++

# Directories
SRC_DIR = src
BUILD_DIR = obj_dir
SIM_BUILD_DIR = obj_sim
TRAIN_BUILD_DIR = obj_train
EQUIV_BUILD_DIR = obj_equivalence

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
SIM_YS ?= 4
TRAIN_XS ?= 4
TRAIN_YS ?= 8
CUDA_ARCH ?= sm_89
CUDA_HOME ?= /usr/local/cuda-13.3
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOST_COMPILER ?= gcc
CUDA_FLAGS ?= -allow-unsupported-compiler
CUDA_XS ?= 16
CUDA_YS ?= 16

VERILATOR_ROOT := $(shell $(VERILATOR) --getenv VERILATOR_ROOT)
CURR_XS ?= 4
CURR_YS ?= 8
CURRICULUM_BUILD_DIR = obj_curriculum
CURRICULUM_TARGET = curriculum_train
CURRICULUM_CPP_SRC = $(SRC_DIR)/curriculum_train.cpp
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

PIPE_XS ?= 4
PIPE_YS ?= 8
PIPELINE_BUILD_DIR = obj_pipeline
PIPELINE_TARGET = pipeline_server
PIPELINE_CPP_SRC = $(SRC_DIR)/pipeline_server.cpp

.PHONY: all run sim sim-build train train-build native-equivalence clean cuda cuda-train cuda-train-server curriculum curriculum-build fpga-lint fpga-synth pipeline-build pipeline-server

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

native-equivalence:
	mkdir -p $(EQUIV_BUILD_DIR)
	$(VERILATOR) -Wall --cc $(VERILOG_SRCS) --top-module $(TOP_MODULE) \
		-GXS=$(MAIN_XS) -GYS=$(MAIN_YS) \
		--exe src/native_equivalence.cpp -I$(SRC_DIR) \
		--Mdir $(EQUIV_BUILD_DIR) --build \
		-CFLAGS "-DEQUIV_XS=$(MAIN_XS) -DEQUIV_YS=$(MAIN_YS)" \
		-o native_equivalence

clean:
	rm -rf $(BUILD_DIR) $(SIM_BUILD_DIR) $(TRAIN_BUILD_DIR) $(CURRICULUM_BUILD_DIR) $(PIPELINE_BUILD_DIR) $(EQUIV_BUILD_DIR)

pipeline-build:
	mkdir -p $(PIPELINE_BUILD_DIR)
	$(VERILATOR) -Wall --cc $(VERILOG_SRCS) --top-module pipeline \
		-GXS=$(PIPE_XS) -GYS=$(PIPE_YS) -I$(SRC_DIR) \
		--Mdir $(PIPELINE_BUILD_DIR)
	$(MAKE) -C $(PIPELINE_BUILD_DIR) -f Vpipeline.mk Vpipeline__ALL.a verilated.o verilated_threads.o
	$(CXX) -std=c++17 -O2 -Wall \
		-DPIPE_XS=$(PIPE_XS) -DPIPE_YS=$(PIPE_YS) \
		-I$(VERILATOR_ROOT)/include -I$(VERILATOR_ROOT)/include/vltstd \
		-I$(PIPELINE_BUILD_DIR) \
		$(PIPELINE_CPP_SRC) \
		$(PIPELINE_BUILD_DIR)/Vpipeline__ALL.a \
		$(PIPELINE_BUILD_DIR)/verilated.o \
		$(PIPELINE_BUILD_DIR)/verilated_threads.o \
		-lpthread -latomic \
		-o $(PIPELINE_BUILD_DIR)/$(PIPELINE_TARGET)

pipeline-server: pipeline-build
	./$(PIPELINE_BUILD_DIR)/$(PIPELINE_TARGET)

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

curriculum-build:
	mkdir -p $(CURRICULUM_BUILD_DIR)/chunk $(CURRICULUM_BUILD_DIR)/pipeline
	$(VERILATOR) -Wall --cc $(VERILOG_SRCS) --top-module chunk \
		-GXS=$(CURR_XS) -GYS=$(CURR_YS) -I$(SRC_DIR) \
		--Mdir $(CURRICULUM_BUILD_DIR)/chunk
	$(MAKE) -C $(CURRICULUM_BUILD_DIR)/chunk -f Vchunk.mk Vchunk__ALL.a verilated.o verilated_threads.o
	$(VERILATOR) -Wall --cc $(VERILOG_SRCS) --top-module pipeline \
		-GXS=$(CURR_XS) -GYS=$(CURR_YS) -I$(SRC_DIR) \
		--Mdir $(CURRICULUM_BUILD_DIR)/pipeline
	$(MAKE) -C $(CURRICULUM_BUILD_DIR)/pipeline -f Vpipeline.mk Vpipeline__ALL.a
	$(CXX) -std=c++17 -O2 -Wall \
		-DCURR_XS=$(CURR_XS) -DCURR_YS=$(CURR_YS) \
		-I$(VERILATOR_ROOT)/include -I$(VERILATOR_ROOT)/include/vltstd \
		-I$(CURRICULUM_BUILD_DIR)/chunk -I$(CURRICULUM_BUILD_DIR)/pipeline \
		$(CURRICULUM_CPP_SRC) \
		$(CURRICULUM_BUILD_DIR)/chunk/Vchunk__ALL.a \
		$(CURRICULUM_BUILD_DIR)/pipeline/Vpipeline__ALL.a \
		$(CURRICULUM_BUILD_DIR)/chunk/verilated.o \
		$(CURRICULUM_BUILD_DIR)/chunk/verilated_threads.o \
		-lpthread -latomic \
		-o $(CURRICULUM_BUILD_DIR)/$(CURRICULUM_TARGET)

curriculum: curriculum-build
	./$(CURRICULUM_BUILD_DIR)/$(CURRICULUM_TARGET)

fpga-lint:
	verilator --lint-only -Wall $(VERILOG_SRCS) --top-module chunk \
		-GXS=$(MAIN_XS) -GYS=$(MAIN_YS) -I$(SRC_DIR)

fpga-synth:
	yosys -p "read_verilog -sv $(VERILOG_SRCS); synth -top chunk"