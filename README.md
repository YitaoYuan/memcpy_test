# memcpy_test

带宽和线程数强相关，但很快饱和(双插槽4线程240Gbps，再往上收效甚微)

numa架构也会影响，当cpu与内存位于相同numa socket时性能最好

# cuda_memcpy_test

对于同一块GPU，带宽和线程（stream）无关，和内存的numa socket也几乎无关

cudaHostRegister能大大提升性能

对于多GPU，由于一个GPU就可以喂满一个socket的内存，所以GPU为2时带宽就饱和了

# PS

咱们的平台GPU到内存是内存瓶颈

咱们一个socket上的内存带宽大概是210Gbps，而PCIE 4.0x16的带宽是256Gbps

所以两个GPU就能打满大约420Gbps的内存带宽，interleaved内存也是这个带宽