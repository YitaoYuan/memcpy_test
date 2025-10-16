// nvcc -std=c++11 gds_stream_example.cu -o gds_stream_example -lcufile
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "cufile.h"

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(e) << std::endl;
        exit(1);
    }
}
static void checkCufile(CUfileError_t st, const char* msg) {
    if (st.err != CU_FILE_SUCCESS) {
        std::cerr << msg << ": cuFileError " << st.err << std::endl;
        exit(1);
    }
}

int main() {
    const char* testfn = std::getenv("TESTFILE");
    if (!testfn) {
        std::cerr << "请通过 TESTFILE 环境变量指定测试文件路径\n";
        return 1;
    }

    // 1. 打开驱动 & 文件
    checkCufile(cuFileDriverOpen(), "cuFileDriverOpen failed");
    int fd = open(testfn, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    CUfileDescr_t descr = {};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    CUfileHandle_t fh;
    checkCufile(cuFileHandleRegister(&fh, &descr),
                "cuFileHandleRegister failed");

    // 2. 分配 GPU 缓冲区并注册
    const size_t IO_SIZE  = 1024 * 1024 * 1024;        // 16 MiB
    const size_t BUF_SIZE = IO_SIZE + 4096; // 多留一点头部空间对齐
    void* d_buf = nullptr;
    checkCuda(cudaMalloc(&d_buf, BUF_SIZE), "cudaMalloc failed");
    d_buf = (void*)(((size_t)d_buf + 4096 - 1) & ~((size_t)4096 - 1));
    checkCufile(cuFileBufRegister(d_buf, BUF_SIZE, 0),
                "cuFileBufRegister failed");

    // 3. 创建 CUDA 流并注册给 cuFile
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");
    // 标记：buffer/file/size 在提交时已知且 4K 对齐
    checkCufile(cuFileStreamRegister(stream, /*flags=*/0xF),
                "cuFileStreamRegister failed");

    checkCuda(cudaStreamSynchronize(stream),
                "cudaStreamSynchronize failed");

    // Use chrono to measure the time
    auto start = std::chrono::high_resolution_clock::now();
    
    // 4. 异步发起读操作
    size_t   size_p       = IO_SIZE;
    off_t    file_off_p   = 0;    // 从文件头开始读
    off_t    buf_off_p    = 0;    // 从 d_buf 开始写
    ssize_t  bytes_read_p = 0;    // 必须预置为 0

    CUfileError_t st = cuFileReadAsync(
        fh,
        d_buf,
        &size_p,
        &file_off_p,
        &buf_off_p,
        &bytes_read_p,
        stream
    );
    checkCufile(st, "cuFileReadAsync failed");

    // 5. 在流上同步，等待 IO 完成
    checkCuda(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize failed");

    auto end = std::chrono::high_resolution_clock::now();
    // output time in seconds (double), and compute the throughput in GB/s
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    std::cout << "Throughput: " << (double)IO_SIZE * 1e-9 / duration.count() << " GB/s" << std::endl;

    std::cout << "Async read completed, bytes_read = "
              << bytes_read_p << std::endl;

    // 6. 清理
    cuFileStreamDeregister(stream);
    cudaStreamDestroy(stream);

    cuFileBufDeregister(d_buf);
    cudaFree(d_buf);

    cuFileHandleDeregister(fh);
    close(fd);

    cuFileDriverClose();
    return 0;
}
