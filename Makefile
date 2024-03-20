CXX = g++
CXX_flags = -O3 # this does not affect performance
GCC_flags = -march=native -mavx512f -mavx512bw # this does not affect the performance, since we use memcpy which is in a precompiled library
link_flags = -lpthread -lnuma
cuda_link_flags = -lcudart
exe_list = memcpy_test cuda_memcpy_test
all: $(exe_list)

memcpy_test: memcpy_test.cpp
	$(CXX) $(CXX_flags) $(GCC_flags) -o $@ $^ $(link_flags)

cuda_memcpy_test: cuda_memcpy_test.cpp
	nvcc $(CXX_flags) -o $@ $^ $(link_flags) $(cuda_link_flags)