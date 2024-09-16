UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	FLAGS = -fPIC -shared -O3 -march=native
	TRG = fast_median.so
	COMPILER = g++
endif
ifeq ($(UNAME_S),Darwin)
        FLAGS = -dynamiclib -current_version 1.0 -compatibility_version 1.0 -O3 -march=native
	TRG = fast_median.dylib
	COMPILER = clang++
endif

all: fast_median.cpp fast_median.hpp
	$(COMPILER) $(FLAGS) -o $(TRG) fast_median.cpp
