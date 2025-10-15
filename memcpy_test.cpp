#include <bits/stdc++.h>
#include <numa.h>
#include <pthread.h>
#include <vector>
using std::vector, std::thread;
//using namespace std;
char *malloc_touched(size_t size, int node)
{
    void *buf = numa_alloc_onnode(size, node);//numa_alloc_interleaved(size);//numa_alloc_local(size);//malloc(size);
    memset(buf, 0xff, size);
    return (char*)buf;
}
uint64_t gettime_ns()
{
    struct timespec cur_time;
    clock_gettime(CLOCK_MONOTONIC, &cur_time);
    return cur_time.tv_sec * (uint64_t)1000000000 + cur_time.tv_nsec;
}
// no_inline: force the compiler not remove any call of this function (in optimization)
__attribute__((noinline)) void memcpy_test(void *rbuf, void *sbuf, size_t size)
{
    memcpy(rbuf, sbuf, size);
}
void repeate_memcpy_test(size_t n, void *rbuf, void *sbuf, size_t size, size_t size_alloc)
{
    size_t segs = size_alloc / size;
    for(size_t i = 0; i < n; i++) {
        size_t seg = i % segs;
        memcpy_test((char*)rbuf + size * seg, (char*)sbuf + size * seg, size);
    }
}
vector<int> get_socket_list_with_cpu_index()
{
    vector<int>ret;
    int ncpu = numa_num_configured_cpus();
    for (int i = 0; i < ncpu; i++)
        ret.push_back(numa_node_of_cpu(i));
    return ret;
}
struct memcpy_info{
    void *sbuf, *rbuf;
    int cpuid, socketid;
};

int main(int argc, char **argv)
{
    size_t size = std::stoll(argv[1]);
    size_t size_alloc = std::stoll(argv[2]);
    int nthread = std::stoi(argv[3]);
    size_t niter = std::stoll(argv[4]);
    // numa_alloc_onnode
    // numa_alloc_interleaved
    // numa_alloc_local
    if(size_alloc < 1000000000) {
        printf("If you want to test DRAM speed, use GB level memory.\n");
    }
    int nnode = numa_num_configured_nodes();
    vector<int>socket_of_cpu = get_socket_list_with_cpu_index();

    printf("%d numa nodes\n", nnode);
    std::vector<memcpy_info> mem_pool;
    for(int i = 0; i < nthread; i++) {
        int node = i % nnode;
        int cpuid = std::find(socket_of_cpu.begin(), socket_of_cpu.end(), node) - socket_of_cpu.begin();
        printf("thread %d on cpu %d of numa node %d\n", i, cpuid, node);
        if(cpuid == socket_of_cpu.size()) {
            perror("No enough cores\n");
            exit(1);
        }
        socket_of_cpu[cpuid] = -1;
        mem_pool.push_back((memcpy_info){malloc_touched(size_alloc, node), malloc_touched(size_alloc, node), cpuid, node});
    }
    uint64_t start = gettime_ns();
    std::vector<std::thread> thread_pool;
    for(int i = 0; i < nthread; i++) {
        thread t = std::thread(repeate_memcpy_test, niter, mem_pool[i].rbuf, mem_pool[i].sbuf, size, size_alloc);

        pthread_t thread_native_handle = t.native_handle();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(mem_pool[i].cpuid, &cpuset);
        pthread_setaffinity_np(thread_native_handle, sizeof(cpu_set_t), &cpuset);

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
    return 0;
}
