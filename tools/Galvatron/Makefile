CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC
PYTHON_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)
SOURCE_DIR = csrc
SOURCE_FILE = dp_core.cpp
BUILD_DIR = galvatron/build
LIB_DIR = $(BUILD_DIR)/lib
OUTPUT_FILE = $(LIB_DIR)/galvatron_dp_core$(PYTHON_EXTENSION_SUFFIX)
CURRENT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SOURCE_DIR)/$(SOURCE_FILE)
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDES) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: clean