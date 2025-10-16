// nccl_p2p_bw.cpp
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <nccl.h>

// CUDA 错误检查
#define CUDA_CALL(call)                                                      \
  do {                                                                       \
    cudaError_t e = call;                                                    \
    if (e != cudaSuccess) {                                                  \
      printf("[CUDA ERR] %s:%d '%s'\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(e));                                        \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// NCCL 错误检查
#define NCCL_CALL(call)                                                      \
  do {                                                                       \
    ncclResult_t r = call;                                                   \
    if (r != ncclSuccess) {                                                  \
      printf("[NCCL ERR] %s:%d '%s'\n", __FILE__, __LINE__,                  \
             ncclGetErrorString(r));                                        \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage: %s <bytes> <repeat>\n", argv[0]);
    return 0;
  }

  size_t bytes  = atoll(argv[1]);
  int    repeat = atoi(argv[2]);
  const int NDEVS = 2;
  int devs[NDEVS]  = {0, 1};
  ncclComm_t comms[NDEVS];
  cudaStream_t streams[NDEVS];
  cudaEvent_t startEvt, stopEvt;
  uint8_t* sendbuff[NDEVS];
  uint8_t* recvbuff[NDEVS];
  void* sendhandle[NDEVS];
  void* recvhandle[NDEVS];

  // 1) Allocate buffers & streams
  for (int i = 0; i < NDEVS; ++i) {
    CUDA_CALL(cudaSetDevice(devs[i]));
    CUDA_CALL(cudaStreamCreate(&streams[i]));
    CUDA_CALL(cudaMalloc(&sendbuff[i], bytes));
    CUDA_CALL(cudaMalloc(&recvbuff[i], bytes));
  }

  // 2) Init NCCL
  NCCL_CALL(ncclCommInitAll(comms, NDEVS, devs));

  for (int i = 0; i < NDEVS; ++i) {
    NCCL_CALL(ncclCommRegister(comms[i], sendbuff[i], bytes, &sendhandle[i]));
    NCCL_CALL(ncclCommRegister(comms[i], recvbuff[i], bytes, &recvhandle[i]));
  }

  // 3) Create timing events on device 0
  CUDA_CALL(cudaSetDevice(devs[0]));
  CUDA_CALL(cudaEventCreate(&startEvt));
  CUDA_CALL(cudaEventCreate(&stopEvt));

  

  // 4) P2P send/recv loop
  const size_t count = bytes;  // 每个元素 1 byte

  ncclGroupStart();// warm up
  for (int i = 0; i < 1; ++i) {
    NCCL_CALL(ncclSend(sendbuff[0], count, ncclUint8, 1, comms[0], streams[0]));
    NCCL_CALL(ncclRecv(recvbuff[1], count, ncclUint8, 0, comms[1], streams[1]));
  }
  ncclGroupEnd();


  CUDA_CALL(cudaEventRecord(startEvt, streams[0]));
  ncclGroupStart();
  for (int i = 0; i < repeat; ++i) {
    NCCL_CALL(ncclSend(sendbuff[0], count, ncclUint8, 1, comms[0], streams[0]));
    NCCL_CALL(ncclRecv(recvbuff[1], count, ncclUint8, 0, comms[1], streams[1]));
  }
  ncclGroupEnd();
  CUDA_CALL(cudaEventRecord(stopEvt, streams[0]));

  // 5) Synchronize & compute bandwidth
  CUDA_CALL(cudaStreamSynchronize(streams[0]));
  float ms = 0.0f;
  CUDA_CALL(cudaEventElapsedTime(&ms, startEvt, stopEvt));
  double seconds   = ms / 1e3;
  double totalGB   = (double)bytes * repeat / 1e9;
  double bandwidth = totalGB / seconds;
  printf("bytes=%zu repeat=%d time=%.3f s BW=%.3f GB/s\n",
         bytes, repeat, seconds, bandwidth);

  // 6) Cleanup
  for (int i = 0; i < NDEVS; ++i) {
    CUDA_CALL(cudaSetDevice(devs[i]));
    NCCL_CALL(ncclCommDestroy(comms[i]));
    CUDA_CALL(cudaFree(sendbuff[i]));
    CUDA_CALL(cudaFree(recvbuff[i]));
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  }
  CUDA_CALL(cudaEventDestroy(startEvt));
  CUDA_CALL(cudaEventDestroy(stopEvt));
  return 0;
}
