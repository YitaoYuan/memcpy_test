// nvcc -std=c++11 gds_stream_example.cu -o gds_stream_example -lcufile
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <boost/asio.hpp>
#include <cufile.h>

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

size_t test_stream(size_t num_iters, size_t num_streams, size_t alloc_size, CUfileHandle_t fh, void* d_buf, cudaStream_t *stream) {
    ssize_t total_bytes_read = 0;
    ssize_t bytes_read[num_iters][num_streams];
    size_t size_per_stream = alloc_size / num_iters / num_streams;
    size_t size_per_iter = alloc_size / num_iters;
    for(int iter = 0; iter < num_iters; iter++) {
        for (int s = 0; s < num_streams; s++) {
            off_t    file_off_p   = size_per_iter * iter + s * size_per_stream;    // 从文件头开始读
            off_t    buf_off_p    = size_per_iter * iter + s * size_per_stream;    // 从 d_buf 开始写
            checkCufile(cuFileReadAsync(fh, d_buf, &size_per_stream, &file_off_p, &buf_off_p, &bytes_read[iter][s], stream[s]), "cuFileReadAsync failed");
        }
    }
    for(int s = 0; s < num_streams; s++) {
        checkCuda(cudaStreamSynchronize(stream[s]), "cudaStreamSynchronize failed");
    }
    for(int iter = 0; iter < num_iters; iter++) {
        for (int s = 0; s < num_streams; s++) {
            if (bytes_read[iter][s] != size_per_stream) {
                std::cerr << "bytes_read[iter][s] != size_per_stream" << std::endl;
                exit(1);
            }
            total_bytes_read += bytes_read[iter][s];
        }
    }
    return total_bytes_read;
}

size_t test_thread(size_t num_iters, size_t num_threads, size_t alloc_size, CUfileHandle_t fh, void* d_buf) {
    ssize_t total_bytes_read = 0;
    ssize_t bytes_read[num_iters][num_threads];
    ssize_t *bytes_read_addr = &bytes_read[0][0];
    size_t size_per_thread = alloc_size / num_iters / num_threads;
    size_t size_per_iter = alloc_size / num_iters;
    boost::asio::thread_pool pool(num_threads);
    for(int iter = 0; iter < num_iters; iter++) {
        for (int t = 0; t < num_threads; t++) {
            off_t    file_off_p   = size_per_iter * iter + t * size_per_thread;    // 从文件头开始读
            off_t    buf_off_p    = size_per_iter * iter + t * size_per_thread;    // 从 d_buf 开始写
            auto func = [=]() {
                ssize_t (*bytes_read_p)[num_threads] = (ssize_t (*) [num_threads]) bytes_read_addr;
                bytes_read_p[iter][t] = cuFileRead(fh, d_buf, size_per_thread, file_off_p, buf_off_p);
            };
            boost::asio::post(pool, func);
        }
    }
    pool.join();
    for(int iter = 0; iter < num_iters; iter++) {
        for (int t = 0; t < num_threads; t++) {
            if (bytes_read[iter][t] != size_per_thread) {
                std::cerr << "bytes_read[iter][t] != size_per_thread" << std::endl;
                exit(1);
            }
            total_bytes_read += bytes_read[iter][t];
        }
    }
    return total_bytes_read;
}

size_t test_batch(size_t num_iters, size_t batch_size, size_t alloc_size, CUfileHandle_t fh, void* d_buf) {
    int max_concurrency = 32;
    CUfileBatchHandle_t batch_id;
    checkCufile(cuFileBatchIOSetUp(&batch_id, max_concurrency), "cuFileBatchIOSetUp failed");

    size_t size_per_io = alloc_size / num_iters / batch_size;
    size_t size_per_iter = alloc_size / num_iters;
    int posted_nr = 0;
    int completed_nr = 0;
    CUfileIOParams_t io_batch_params[batch_size];
    for(int iter = 0; iter < num_iters; iter++) {
        while (posted_nr + batch_size - completed_nr > max_concurrency) {
            unsigned min_nr = batch_size;
            unsigned nr = batch_size; // nr is both input (max_nr) and output (completed_nr)
            CUfileIOEvents_t io_batch_events[batch_size];
            struct timespec* timeout = NULL;
            checkCufile(cuFileBatchIOGetStatus(batch_id, min_nr, &nr, io_batch_events, timeout), "cuFileBatchIOGetStatus failed");	
            completed_nr += nr;
        }
        for(int i = 0; i < batch_size; i++) {
            io_batch_params[i].mode = CUFILE_BATCH;
            io_batch_params[i].fh = fh;
            io_batch_params[i].u.batch.devPtr_base = d_buf;
            io_batch_params[i].u.batch.file_offset = size_per_iter * iter + i * size_per_io;
            io_batch_params[i].u.batch.devPtr_offset = size_per_iter * iter + i * size_per_io;
            io_batch_params[i].u.batch.size = size_per_io;
            io_batch_params[i].opcode = CUFILE_READ;
        }
        unsigned int flags = 0;
        checkCufile(cuFileBatchIOSubmit(batch_id, batch_size, io_batch_params, flags), "cuFileBatchIO failed");

        posted_nr += batch_size;
    }
    while (posted_nr > completed_nr) {
        unsigned min_nr = batch_size;
        unsigned nr = batch_size; // nr is both input (max_nr) and output (completed_nr)
        CUfileIOEvents_t io_batch_events[batch_size];
        struct timespec* timeout = NULL;
        checkCufile(cuFileBatchIOGetStatus(batch_id, min_nr, &nr, io_batch_events, timeout), "cuFileBatchIOGetStatus failed");	
        completed_nr += nr;
    }
    cuFileBatchIODestroy(batch_id);
    return 0;
}

int main(int argc, char** argv) {// (4threads, 4MB) is good enough, (8threads, 8MB) for best performance
    std::string file = argv[1];
    size_t num_streams = std::stoll(argv[2]);
    size_t stream_io_size = std::stoll(argv[3]);
    size_t total_size = std::stoll(argv[4]);
    std::string mode = argv[5];
    size_t num_iters = total_size / (num_streams * stream_io_size);
    size_t alloc_size = num_iters * num_streams * stream_io_size;

    // 1. 打开驱动 & 文件
    checkCufile(cuFileDriverOpen(), "cuFileDriverOpen failed");
    int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // checkCufile(cuFileDriverSetMaxDirectIOSize(16 * 1024), "cuFileDriverSetMaxDirectIOSize failed");


    CUfileDescr_t descr = {};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    CUfileHandle_t fh;
    checkCufile(cuFileHandleRegister(&fh, &descr),
                "cuFileHandleRegister failed");

    // 2. 分配 GPU 缓冲区并注册
    void* d_buf = nullptr;
    checkCuda(cudaMalloc(&d_buf, alloc_size), "cudaMalloc failed");
    checkCufile(cuFileBufRegister(d_buf, alloc_size, 0),
                "cuFileBufRegister failed");

    // 3. 创建 CUDA 流并注册给 cuFile
    cudaStream_t stream[num_streams];
    for (int i = 0; i < num_streams; i++) {
        checkCuda(cudaStreamCreate(&stream[i]), "cudaStreamCreate failed");
        // flags=0xF 表示 file_off、buf_off、size 在提交时都是 4K 对齐已知，设置这个flag可以提高性能
        checkCufile(cuFileStreamRegister(stream[i], /*flags=*/0xF),
                    "cuFileStreamRegister failed"); 
        checkCuda(cudaStreamSynchronize(stream[i]),
                  "cudaStreamSynchronize failed");
    }
    // Use chrono to measure the time
    auto start = std::chrono::high_resolution_clock::now();
    
    ssize_t total_bytes_read = 0;
    if (mode == "stream") {
        total_bytes_read = test_stream(num_iters, num_streams, alloc_size, fh, d_buf, stream);
    }
    else if (mode == "thread") {
        total_bytes_read = test_thread(num_iters, num_streams, alloc_size, fh, d_buf);
    }
    else if (mode == "batch") {
        total_bytes_read = test_batch(num_iters, num_streams, alloc_size, fh, d_buf);
    }
    else {
        std::cerr << "Invalid test mode" << std::endl;
        exit(1);
    }
    // for(int iter = 0; iter < num_iters; iter++) {
    //     for (int s = 0; s < num_streams; s++) {
    //         // 4. 异步发起读操作
    //         size_t   size_p       = alloc_size / num_iters / num_streams;
    //         off_t    file_off_p   = (alloc_size / num_iters) * iter + s * size_p;    // 从文件头开始读
    //         off_t    buf_off_p    = (alloc_size / num_iters) * iter + s * size_p;    // 从 d_buf 开始写
    //         // printf("file_off_p = %ld, buf_off_p = %ld, size_p = %ld\n", file_off_p, buf_off_p, size_p);
    //         // CUfileError_t st = cuFileReadAsync(
    //         //     fh,
    //         //     d_buf,
    //         //     &size_p,
    //         //     &file_off_p,
    //         //     &buf_off_p,
    //         //     &bytes_read[iter][s],
    //         //     stream[s]
    //         // );
    //         // checkCufile(st, "cuFileReadAsync failed");
    //         bytes_read[iter][s] = cuFileRead(
    //             fh,
    //             d_buf,
    //             size_p,
    //             file_off_p,
    //             buf_off_p
    //         );
            
    //     }
    // }

    printf("total_bytes_read = %ld\n", total_bytes_read);

    auto end = std::chrono::high_resolution_clock::now();
    // output time in seconds (double), and compute the throughput in GB/s
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    std::cout << "Throughput: " << (double)alloc_size * 1e-9 / duration.count() << " GB/s" << std::endl;

    std::cout << "Async read completed, bytes_read = "
              << total_bytes_read << std::endl;

    // 6. 清理
    for (int i = 0; i < num_streams; i++) {
        cuFileStreamDeregister(stream[i]);
        cudaStreamDestroy(stream[i]);
    }
    cuFileBufDeregister(d_buf);
    cudaFree(d_buf);
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileDriverClose();
    return 0;
}
