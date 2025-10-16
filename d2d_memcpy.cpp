// P2P Test by Greg Gutmann
 
#include "stdio.h"
#include "stdint.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <random>
#include <vector>

#define CUDA_CALL(call)                                                      \
  do {                                                                       \
    cudaError_t e = call;                                                    \
    if (e != cudaSuccess) {                                                  \
      printf("[CUDA ERR] %s:%d '%s'\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(e));                                        \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

int main(int argc, char **argv)
{
    // GPUs
    int gpuid_0 = 0;
    int gpuid_1 = 1;
 
    
    // Memory Copy Size
    size_t size = std::atoi(argv[1]);
    size_t repeat = std::atoi(argv[2]);
    bool enable_peer = (bool)std::atoi(argv[3]);

    // Allocate Memory
    char* dev_0;
    cudaSetDevice(gpuid_0);
    CUDA_CALL(cudaMalloc((void**)&dev_0, size * repeat));
 
    char* dev_1;
    cudaSetDevice(gpuid_1);
    CUDA_CALL(cudaMalloc((void**)&dev_1, size * repeat));

    std::random_device rd;
    std::mt19937       g(rd());

    std::vector<void*> send_addr;
    for(size_t i = 0; i < repeat; i++) send_addr.push_back(dev_1 + i * size);
    std::shuffle(send_addr.begin(), send_addr.end(), g);
    std::vector<void*> recv_addr;
    for(size_t i = 0; i < repeat; i++) recv_addr.push_back(dev_0 + i * size);
    std::shuffle(recv_addr.begin(), recv_addr.end(), g);

    // std::vector<size_t> sizes;
    // for(int i = 0; i < repeat; i++) {
    //     sizes.push_back(size);
    // }

 
    //Check for peer access between participating GPUs: 
 
    if (enable_peer) {
        int can_access_peer_0_1;
        int can_access_peer_1_0;
        cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);
        cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);
        printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_0, gpuid_1, can_access_peer_0_1);
        printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_1, gpuid_0, can_access_peer_1_0);
        if(!(can_access_peer_0_1 && can_access_peer_1_0)) {
            printf("Peer access not allowed\n");
            exit(1);
        }
        // Enable P2P Access
        cudaSetDevice(gpuid_0);
        cudaDeviceEnablePeerAccess(gpuid_1, 0);
        cudaSetDevice(gpuid_1);
        cudaDeviceEnablePeerAccess(gpuid_0, 0);
    }
 
    // Init Timing Data
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Init Stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
 
    // std::vector<cudaMemcpyAttributes> attrs(1);
    // // attrs[0].kind            = cudaMemcpyHostToDevice;
    // attrs[0].srcAccessOrder  = cudaMemcpySrcAccessOrderAny;
    // attrs[0].srcLocHint      = 0;       // 无管理内存 hint
    // attrs[0].dstLocHint      = 0;
    // attrs[0].flags           = cudaMemcpyFlagPreferOverlapWithCompute;  

    // ~~ Start Test ~~
    cudaEventRecord(start, stream);
 
    //Do a P2P memcpy
    size_t failIdx;
    // cudaMemcpyBatchAsync (recv_addr.data(), send_addr.data(), sizes.data(), recv_addr.size(), attrs.data, &(size_t)0, 1, &failIdx, stream);
    for (int i = 0; i < repeat; ++i) {
        cudaMemcpyAsync((char*)recv_addr[i] - dev_0 + dev_1, send_addr[i], size, cudaMemcpyDeviceToDevice, stream);
        // cudaMemcpyPeerAsync(recv_addr[i], gpuid_0, send_addr[i], gpuid_1, size, stream);
    }
 
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    // ~~ End of Test ~~
 
    // Check Timing & Performance
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    double time_s = time_ms / 1e3;
 
    double gb = size * repeat / (double)1e9;
    double bandwidth = gb / time_s;
 
    printf("Seconds: %f\n", time_s);
    printf("Unidirectional Bandwidth: %f (GB/s)\n", bandwidth);
 
    if (enable_peer) {
        // Shutdown P2P Settings
        cudaSetDevice(gpuid_0);
        cudaDeviceDisablePeerAccess(gpuid_1);
        cudaSetDevice(gpuid_1);
        cudaDeviceDisablePeerAccess(gpuid_0);
    }
 
    // Clean Up
    cudaFree(dev_0);
    cudaFree(dev_1);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}