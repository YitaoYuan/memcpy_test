CXX = g++
CXX_flags = -O3
link_flags = -lpthread -lnuma

all: memcpy_test

memcpy_test: memcpy_test.cpp
	$(CXX) $(CXX_flags) -o $@ $^ $(link_flags)

