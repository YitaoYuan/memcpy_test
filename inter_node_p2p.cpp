// p2p.cpp: nvcc -lnccl -lcudart inter_node_p2p.cpp -o inter_node_p2p
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include <cuda_runtime.h>
#include <nccl.h>

#define PORT 5000
#define CHECK_CUDA(cmd) do {                              \
  cudaError_t e = cmd;                                    \
  if (e != cudaSuccess) {                                 \
    std::cerr << "CUDA:" << cudaGetErrorString(e)         \
              << " at " << __FILE__ << ":" << __LINE__    \
              << std::endl; exit(EXIT_FAILURE);           \
  }                                                       \
} while(0)

#define CHECK_NCCL(cmd) do {                              \
  ncclResult_t r = cmd;                                   \
  if (r != ncclSuccess) {                                 \
    std::cerr << "NCCL " << ncclGetErrorString(r)         \
              << " at " << __FILE__ << ":" << __LINE__    \
              << std::endl; exit(EXIT_FAILURE);           \
  }                                                       \
} while(0)

// send all bytes
ssize_t sendAll(int sock, const void* buf, size_t len) {
  size_t sent = 0; const char* p = (const char*)buf;
  while (sent < len) {
    ssize_t s = send(sock, p + sent, len - sent, 0);
    if (s <= 0) return s;
    sent += s;
  }
  return sent;
}
// recv all bytes
ssize_t recvAll(int sock, void* buf, size_t len) {
  size_t recvd = 0; char* p = (char*)buf;
  while (recvd < len) {
    ssize_t r = recv(sock, p + recvd, len - recvd, 0);
    if (r <= 0) return r;
    recvd += r;
  }
  return recvd;
}

int main(int argc, char* argv[]) {
  bool isServer = false;
  std::string peerIP;
  int gpuid = 0;

  if (argc==3 && strcmp(argv[1],"-s")==0) {
    isServer = true;
    gpuid = atoi(argv[2]);
  } else if (argc==4 && strcmp(argv[1],"-c")==0) {
    isServer = false;
    peerIP = argv[2];
    gpuid = atoi(argv[3]);
  } else {
    std::cerr << "Usage:\n"
              << "  Server: " << argv[0] << " -s X\n"
              << "  Client: " << argv[0] << " -c <server_ip> Y\n";
    return 1;
  }

  // 1) 准备 socket 通信，交换 NCCL uniqueId
  int sock = -1, conn = -1;
  if (isServer) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd<0) { perror("socket"); exit(1); }
    #ifdef SO_REUSEPORT
    int opt = 1;
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
    perror("setsockopt SO_REUSEPORT"); /* 不必致命 */ 
    }
    #endif
    sockaddr_in addr;
    std::memset(&addr,0,sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);
    if (bind(listen_fd, (sockaddr*)&addr, sizeof(addr))<0) {
      perror("bind"); exit(1);
    }
    if (listen(listen_fd,1)<0) { perror("listen"); exit(1); }
    std::cout<<"[Server] listening on port "<<PORT<<std::endl;
    conn = accept(listen_fd, NULL, NULL);
    if (conn<0) { perror("accept"); exit(1); }
    close(listen_fd);
  } else {
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock<0) { perror("socket"); exit(1); }
    sockaddr_in addr;
    std::memset(&addr,0,sizeof(addr));
    addr.sin_family = AF_INET;
    inet_pton(AF_INET, peerIP.c_str(), &addr.sin_addr);
    addr.sin_port = htons(PORT);
    std::cout<<"[Client] connecting to "<<peerIP<<":"<<PORT<<std::endl;
    if (connect(sock, (sockaddr*)&addr, sizeof(addr))<0) {
      perror("connect"); exit(1);
    }
    conn = sock;
  }

  // 2) 在本地 GPU 上准备 NCCL 环境
  CHECK_CUDA( cudaSetDevice(gpuid) );
  ncclUniqueId id;
  if (isServer) {
    // 服务端生成 uniqueId 并发给客户端
    CHECK_NCCL( ncclGetUniqueId(&id) );
    ssize_t s = sendAll(conn, &id, sizeof(id));
    if (s!=sizeof(id)) { std::cerr<<"sendAll error\n"; exit(1); }
  } else {
    // 客户端接收 uniqueId
    ssize_t r = recvAll(conn, &id, sizeof(id));
    if (r!=sizeof(id)) { std::cerr<<"recvAll error\n"; exit(1); }
  }
  close(conn);

  // 3) 创建 NCCL communicator
  ncclComm_t comm;
  int nranks = 2;
  int rank = isServer ? 0 : 1;
  CHECK_NCCL( ncclCommInitRank(&comm, nranks, id, rank) );

  // 4) 准备通信缓冲区
  const size_t count = 1024;
  float* d_buf = nullptr;
  CHECK_CUDA( cudaMalloc(&d_buf, count*sizeof(float)) );
  cudaStream_t stream;
  CHECK_CUDA( cudaStreamCreate(&stream) );

  if (!isServer) {
    // 客户端在 buf 中写数据并发送
    std::vector<float> h_send(count);
    for (size_t i=0;i<count;i++) h_send[i] = float(i);
    CHECK_CUDA( cudaMemcpy(d_buf, h_send.data(),
                          count*sizeof(float),
                          cudaMemcpyHostToDevice) );
    std::cout<<"[Client] sending "<<count<<" floats\n";
    CHECK_NCCL( ncclSend(d_buf, count, ncclFloat, 0, comm, stream) );
    CHECK_CUDA( cudaStreamSynchronize(stream) );
    std::cout<<"[Client] send done\n";
  } else {
    // 服务端接收到 buf 中，然后拷贝回 host 验证
    CHECK_NCCL( ncclRecv(d_buf, count, ncclFloat, 1, comm, stream) );
    CHECK_CUDA( cudaStreamSynchronize(stream) );
    std::vector<float> h_recv(count);
    CHECK_CUDA( cudaMemcpy(h_recv.data(), d_buf,
                          count*sizeof(float),
                          cudaMemcpyDeviceToHost) );
    std::cout<<"[Server] received, sample:";
    for (int i=0;i<10;i++) std::cout<<" "<<h_recv[i];
    std::cout<<"\n";
  }

  // 5) 清理
  CHECK_CUDA( cudaStreamDestroy(stream) );
  CHECK_CUDA( cudaFree(d_buf) );
  CHECK_NCCL( ncclCommDestroy(comm) );

  std::cout<<(isServer?"[Server] done\n":"[Client] done\n");
  return 0;
}
