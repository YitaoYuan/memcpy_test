#include <bits/stdc++.h>
#include <numa.h>
#include <cuda.h>
#include <cuda_runtime.h>
//using namespace std;
char *malloc_touched(size_t size, int node)
{
    // void *buf;
    // cudaHostAlloc(&buf, size, cudaHostAllocDefault | cudaHostAllocWriteCombined);
    void *buf = numa_alloc_onnode(size, node);//
    // void *buf = numa_alloc_interleaved(size);//numa_alloc_local(size);//malloc(size);
    cudaHostRegister(buf, size, cudaHostRegisterDefault);
    memset(buf, 0xff, size);
    // I found that if we don't register it, the MemcpyAsync becomes sync
    // And, registering memory let the throughput +80 Gbps (120->200)
    return (char*)buf;
}
uint64_t gettime_ns()
{
    struct timespec cur_time;
    clock_gettime(CLOCK_MONOTONIC, &cur_time);
    return cur_time.tv_sec * (uint64_t)1000000000 + cur_time.tv_nsec;
}
// no_inline: force the compiler not remove any call of this function (in optimization)
// volatile: used for the same reason as no_inline
__attribute__((noinline)) void volatile_memcpy(void *rbuf, void *sbuf, size_t size, cudaStream_t s)
{
    cudaMemcpyAsync(rbuf, sbuf, size, cudaMemcpyDeviceToHost, s);
}

int main(int argc, char **argv)
{
    size_t size = std::stoll(argv[1]);
    int nthread = std::stoi(argv[2]);// It seems that nthread does not improve performance
    int niter = std::stoi(argv[3]);
    // numa_alloc_onnode
    // numa_alloc_interleaved
    // numa_alloc_local
    int nnode = numa_num_configured_nodes();
    int ngpu;
    auto ret = cudaGetDeviceCount(&ngpu);
    assert(ret == cudaSuccess);

    printf("%d numa nodes, %d gpus\n", nnode, ngpu);
    std::vector<std::pair<char *, char *>> mem_pool;
    std::vector<cudaStream_t> thread_pool;
    for(int i = 0; i < nthread; i++) {
        int gpuid = i % ngpu;
        ret = cudaSetDevice(gpuid);
        assert(ret == cudaSuccess);
    

        int node = i % nnode;
        void *gpu_buf;
        cudaMalloc(&gpu_buf, size);
        mem_pool.push_back(std::make_pair(malloc_touched(size, node), (char*)gpu_buf));

        cudaStream_t s;
        cudaStreamCreate(&s);
        thread_pool.push_back(s);
    }
    uint64_t start = gettime_ns();
    for(int cnt = 0; cnt < niter; cnt ++) {
        for(int i = 0; i < nthread; i++) {
            volatile_memcpy(mem_pool[i].first, mem_pool[i].second, size, thread_pool[i]); 
        }
    }
    for(int i = 0; i < nthread; i++) {
	    cudaStreamSynchronize(thread_pool[i]);
    }
    //volatile_memcpy(rbuf, sbuf, size);
    // std::copy(sbuf, sbuf+size, rbuf);
    uint64_t nbyte = size * nthread * niter;
    uint64_t end = gettime_ns();
    uint64_t dt_ns = end - start;
    double tpt_MBps = 1.0*nbyte/dt_ns*1e3;
    double tpt_Gbps = 1.0*nbyte*8/dt_ns;
    printf("%llu: %.2lf s, %.2lf Gbps, %.2lf MBps\n", nbyte, dt_ns*1e-9, tpt_Gbps, tpt_MBps);
    return 0;
}
