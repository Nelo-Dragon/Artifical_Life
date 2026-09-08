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

.PHONY: all clean

all:
	$(VERILATOR) $(VERILATOR_FLAGS)

run: all
	./$(BUILD_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)