#include <bits/stdc++.h>
#include <numa.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

using std::vector, std::thread;
//using namespace std;

#define CHECK_NVML(call)                                                   \
    do {                                                                    \
        nvmlReturn_t _res = call;                                          \
        if (_res != NVML_SUCCESS) {                                        \
            std::cerr << "NVML error " << _res                               \
                      << " at " << __FILE__ << ":" << __LINE__             \
                      << " -> " << nvmlErrorString(_res) << "\n";          \
            std::exit(1);                                                  \
        }                                                                   \
    } while (0)

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
__attribute__((noinline)) void memcpy_test(void *rbuf, void *sbuf, size_t size, cudaStream_t s)
{
    cudaMemcpyAsync(rbuf, sbuf, size, cudaMemcpyDeviceToHost, s);
}

void repeat_memcpy_test(size_t n, void *rbuf, void *sbuf, size_t size, size_t size_alloc, cudaStream_t s)
{
    size_t segs = size_alloc / size;
    for(size_t i = 0; i < n; i++) {
        size_t seg = i % segs;
        memcpy_test((char*)rbuf + size * seg, (char*)sbuf + size * seg, size, s);
    }
    cudaStreamSynchronize(s);
}

int readNumaNode(const std::string &pciBusId)
{
    std::string path = "/sys/bus/pci/devices/" + pciBusId + "/numa_node";
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        // -1 表示读取失败或该设备不属于任何 node
        return -1;
    }
    int node = -1;
    ifs >> node;
    return node;
}

int main(int argc, char **argv)
{
    size_t size = std::stoll(argv[1]);
    size_t size_alloc = std::stoll(argv[2]);
    int nthread = std::stoi(argv[3]);
    size_t niter = std::stoll(argv[4]);
    // numa_alloc_onnode
    // numa_alloc_interleaved
    // numa_alloc_local

    CHECK_NVML( nvmlInit() );


    int nnode = numa_num_configured_nodes();
    int ngpu;
    auto ret = cudaGetDeviceCount(&ngpu);
    assert(ret == cudaSuccess);
    size_t segs = size_alloc / size;

    printf("%d numa nodes, %d gpus\n", nnode, ngpu);
    std::vector<std::pair<char *, char *>> mem_pool;
    std::vector<cudaStream_t> stream_pool;
    for(int i = 0; i < nthread; i++) {
        int gpuid = i % ngpu;
        ret = cudaSetDevice(gpuid);
        assert(ret == cudaSuccess);
    
        nvmlDevice_t dev;
        CHECK_NVML( nvmlDeviceGetHandleByIndex(i, &dev) );
        nvmlPciInfo_t pci;
        CHECK_NVML( nvmlDeviceGetPciInfo(dev, &pci) );
        std::string busId = pci.busId;
        busId = busId.substr(4, 12);
        for (auto &c : busId) c = std::tolower(c);
        std::cout << "busId: " << busId << std::endl;
        int node = readNumaNode(busId);
        void *gpu_buf;
        cudaMalloc(&gpu_buf, size_alloc);
        mem_pool.push_back(std::make_pair(malloc_touched(size_alloc, node), (char*)gpu_buf));

        cudaStream_t s;
        ret = cudaStreamCreate(&s);
        assert(ret == cudaSuccess);
        stream_pool.push_back(s);

        printf("stream %d on gpu %d, write to memory on node %d\n", i, gpuid, node);
    }
    uint64_t start = gettime_ns();
    // for(int cnt = 0; cnt < niter; cnt ++) {
    //     int seg = cnt % segs;
    //     for(int i = 0; i < nthread; i++) {
    //         volatile_memcpy((char*)mem_pool[i].first + size * seg, (char*)mem_pool[i].second + size * seg, size, stream_pool[i]); 
    //     }
    // }
    std::vector<std::thread> thread_pool;
    for(int i = 0; i < nthread; i++) {
        thread t = std::thread(repeat_memcpy_test, niter, mem_pool[i].first, mem_pool[i].second, size, size_alloc, stream_pool[i]);
        thread_pool.push_back(std::move(t));
    }
    for(int i = 0; i < nthread; i++) {
	    thread_pool[i].join();
    }
    uint64_t nbyte = size * nthread * niter;
    uint64_t end = gettime_ns();
    uint64_t dt_ns = end - start;
    double tpt_MBps = 1.0*nbyte/dt_ns*1e3;
    double tpt_Gbps = 1.0*nbyte*8/dt_ns;
    printf("%llu bytes, %.2lf s, %.2lf Gbps, %.2lf MBps\n", nbyte, dt_ns*1e-9, tpt_Gbps, tpt_MBps);

    CHECK_NVML( nvmlShutdown() );
    return 0;
}
