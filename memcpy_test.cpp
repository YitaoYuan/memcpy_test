#include <bits/stdc++.h>
#include <numa.h>
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
__attribute__((noinline)) void volatile_memcpy(int node, void *rbuf, void *sbuf, size_t size)
{
    memcpy(rbuf, sbuf, size);
    //memcpy((void*)sbuf, (void*)rbuf, size);
}
void repeate_volatile_memcpy(int n, int node, void *rbuf, void *sbuf, size_t size)
{
    //struct bitmask mask;
    //numa_node_to_cpus(node, &mask);
    pthread_t thread = pthread_self();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int ncpu = numa_num_configured_cpus();
    for (int i = 0; i < ncpu; i++)
        if(numa_node_of_cpu(i) == node)
            CPU_SET(i, &cpuset);
    int ret = pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);//numa api numa_sched_setaffinity() only set process affinity, not thread affinity
    assert(ret == 0);
    for(int i = 0; i < n; i++) 
        volatile_memcpy(node, rbuf, sbuf, size);
}
int main(int argc, char **argv)
{
    size_t size = std::stoll(argv[1]);
    int nthread = std::stoi(argv[2]);
    int niter = std::stoi(argv[3]);
    // numa_alloc_onnode
    // numa_alloc_interleaved
    // numa_alloc_local
    if(size < 1000000000) {
        printf("If you want to test DRAM speed, use GB level memory.\n");
    }
    int nnode = numa_num_configured_nodes();
    printf("%d numa nodes\n", nnode);
    std::vector<std::pair<char *, char *>> mem_pool;
    for(int i = 0; i < nthread; i++) {
        int node = i % nnode;
        mem_pool.push_back(std::make_pair(malloc_touched(size, node), malloc_touched(size, node)));
    }
    uint64_t start = gettime_ns();
    std::vector<std::thread> thread_pool;
    for(int i = 0; i < nthread; i++) {
        thread_pool.push_back(std::thread(repeate_volatile_memcpy, niter, i % nnode, mem_pool[i].first, mem_pool[i].second, size));
    }
    for(int i = 0; i < nthread; i++) {
	    thread_pool[i].join();
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
