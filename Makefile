
SRC_DIR := src
CXX = g++

CXXFLAGS = \
	-std=c++1y \
	-Wall \
	-O3

CPP_SRC = src/hpyplm.cc
EXE = main_exe
PYTHON3 := $(if $(PYTHON3),$(PYTHON3),python3)

.PHONY: all hpyplm run clean distclean

all: generate_pybind hpyplm

hpyplm: src/hpyplm.cc
	$(CXX) $(CXXFLAGS) -I.. src/hpyplm.cc -o $(EXE)

generate_pybind:  
	$(CXX) -shared $(CXXFLAGS) -fPIC `$(PYTHON3) -m pybind11 --includes` \
		python_exports.cpp -o bnplm.so`$(PYTHON3)-config --extension-suffix` 


clean:
	$(RM) *.o
	$(RM) $(SRC_DIR)/*.o

distclean: clean
	$(RM) $(EXE)
	$(RM) *~
