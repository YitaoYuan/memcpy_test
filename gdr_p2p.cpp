// nvcc ./gdr_p2p.cpp -o gdr_p2p -O2 -lnccl -lgdrapi -lnvidia-ml $(pkg-config --cflags --libs libibverbs)
/*
 * gdr_rc_pingpong_fixed2.c
 *
 * 用法: ./gdr_rc_pingpong_fixed2 <size> <repeat>
 *
 * fork 出 client/server 两进程，在 GPU 显存上做 RDMA RC
 * ping‑pong，测平均 RTT。
 * 
 * 特点：
 *   1) 通过 /sys/class/infiniband/<hca>/device 下的 domain/bus/dev 读取 PCI 信息，
 *      避免 ibv_device_attr_ex 版本差异。
 *   2) 避免 WR 设计器初始化在 C++ 下出错，全部采用赋值方式。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <infiniband/verbs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <gdrapi.h>

#define ASSERT(exp)                                                   \
  do {                                                                 \
    if (!(exp)) {                                                      \
      fprintf(stderr,                                             \
                   "Assertion failed: (%s)\n"                          \
                   "  file: %s\n"                                      \
                   "  line: %d\n",                                     \
                   #exp, __FILE__, __LINE__);                          \
      std::abort();                                                    \
    }                                                                  \
  } while (0)

#define TCP_PORT 18515

struct exch {
    uint64_t addr;
    uint32_t rkey;
    uint32_t qp_num;
    uint16_t lid;
    ibv_gid  gid;
    uint16_t pad;
};

static void getPortPci(struct ibv_context *ctx, uint8_t port,
                       uint32_t *pd, uint32_t *pb, uint32_t *pr)
{
    char syspath[PATH_MAX];
    char realpathbuf[PATH_MAX];
    char *basename_pci;
    unsigned d, b, slot, func;
    /* 1) Build the path to “…/infiniband/<hca>/device” */
    snprintf(syspath, sizeof(syspath),
             "/sys/class/infiniband/%s/device",
             ibv_get_device_name(ctx->device));
    /* 2) Resolve the symlink to get the absolute path, e.g.
         /sys/devices/pci0000:40/0000:40:05.0 */
    if (!realpath(syspath, realpathbuf)) {
        perror("realpath");
        return;
    }
    /* 3) Extract the last component “0000:40:05.0” */
    basename_pci = strrchr(realpathbuf, '/');
    if (!basename_pci) {
        fprintf(stderr, "unexpected pci path: %s\n", realpathbuf);
        return;
    }
    basename_pci++;  /* now points to “0000:40:05.0” */
    /* 4) Parse domain:bus:slot.func */
    if (sscanf(basename_pci, "%x:%x:%x.%u",
               &d, &b, &slot, &func) != 4) {
        fprintf(stderr, "cannot sscanf pci id: %s\n", basename_pci);
        return;
    }
    *pd = d;        /* PCI domain */
    *pb = b;        /* PCI bus    */
    /* Here we combine slot and func into a single u32.
       You could also return just the slot, depending on usage. */
    *pr = (slot << 3) | func;
}

/* Retrieve the GPU’s PCI triplet (domain, bus, device) via NVML */
static void getGpuPci(int gpu, uint32_t *pd, uint32_t *pb, uint32_t *pr)
{
    nvmlReturn_t nvret = nvmlInit();
    if (nvret != NVML_SUCCESS) {
        fprintf(stderr, "NVML init failed: %s\n",
                nvmlErrorString(nvret));
        exit(1);
    }
    nvmlDevice_t h;
    nvret = nvmlDeviceGetHandleByIndex(gpu, &h);
    if (nvret != NVML_SUCCESS) {
        fprintf(stderr, "NVML handle %d failed: %s\n",
                gpu, nvmlErrorString(nvret));
        exit(1);
    }
    nvmlPciInfo_t pci;
    nvret = nvmlDeviceGetPciInfo(h, &pci);
    if (nvret != NVML_SUCCESS) {
        fprintf(stderr, "NVML getPciInfo failed: %s\n",
                nvmlErrorString(nvret));
        exit(1);
    }
    *pd = pci.domain;
    *pb = pci.bus;
    *pr = pci.device;
}

static uint64_t pciDistance(uint32_t d1, uint32_t b1, uint32_t r1,
                            uint32_t d2, uint32_t b2, uint32_t r2)
{
    if (d1 == d2 && b1 == b2 && r1 == r2) return 0;
    if (d1 == d2 && b1 == b2)         return 1;
    return 2;
}

static int tcp_listen()
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); exit(1); }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a;
    memset(&a,0,sizeof a);
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port        = htons(TCP_PORT);
    if (bind(s,(struct sockaddr*)&a,sizeof a)) { perror("bind"); exit(1); }
    if (listen(s,1)) { perror("listen"); exit(1); }
    int c = accept(s,NULL,NULL);
    if (c<0){ perror("accept"); exit(1); }
    close(s);
    return c;
}

static int tcp_connect()
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s<0){ perror("socket"); return -1;}
    struct sockaddr_in a;
    memset(&a,0,sizeof a);
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port        = htons(TCP_PORT);
    for(int i=0;i<200;i++){
        if (connect(s,(struct sockaddr*)&a,sizeof a)==0) return s;
        usleep(10000);
    }
    perror("connect");
    close(s);
    return -1;
}

static void modify_qp_init(struct ibv_qp *qp, int port)
{
    struct ibv_qp_attr attr;
    memset(&attr,0,sizeof attr);
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.port_num        = port;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ
                         | IBV_ACCESS_REMOTE_WRITE
                         | IBV_ACCESS_LOCAL_WRITE;
    int ret = ibv_modify_qp(qp,&attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX |
        IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS);
    ASSERT(ret == 0);
}

static void modify_qp_rtr(struct ibv_qp *qp,
                          uint32_t remote_qpn,
                          uint16_t dlid,
                          ibv_gid dgid,
                          int port,
                          int local_gid_index)
{
    struct ibv_qp_attr attr;
    struct ibv_ah_attr ah;
    memset(&attr,0,sizeof attr);
    memset(&ah,0,sizeof ah);
    ah.is_global     = 1; 
    
    ah.grh.dgid = dgid;
    ah.grh.flow_label = 0;
    ah.grh.sgid_index = local_gid_index; // 本地 GID index，需与 query_gid 一致
    ah.grh.hop_limit  = 1;
    ah.grh.traffic_class = 0;

    ah.dlid          = dlid;
    ah.sl            = 0;
    ah.src_path_bits = 0;
    ah.port_num      = port;
    attr.ah_attr            = ah;
    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = IBV_MTU_1024;
    attr.dest_qp_num        = remote_qpn;
    attr.rq_psn             = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer      = 12;
    int ret = ibv_modify_qp(qp,&attr,
        IBV_QP_STATE   | IBV_QP_AV       |
        IBV_QP_PATH_MTU| IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN  | IBV_QP_MAX_DEST_RD_ATOMIC|
        IBV_QP_MIN_RNR_TIMER);
    ASSERT(ret == 0);
}

static void modify_qp_rts(struct ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    memset(&attr,0,sizeof attr);
    attr.qp_state      = IBV_QPS_RTS;
    attr.timeout       = 14;
    attr.retry_cnt     = 7;
    attr.rnr_retry     = 7;
    attr.sq_psn        = 0;
    attr.max_rd_atomic = 1;
    int ret = ibv_modify_qp(qp,&attr,
        IBV_QP_STATE    | IBV_QP_TIMEOUT     |
        IBV_QP_RETRY_CNT| IBV_QP_RNR_RETRY  |
        IBV_QP_SQ_PSN   | IBV_QP_MAX_QP_RD_ATOMIC);
    ASSERT(ret == 0);
}

bool find_gid_index(struct ibv_context *ctx, int port, int *out_gid_index, ibv_gid *out_gid, int gid_type=IBV_GID_TYPE_ROCE_V2)
{
    struct ibv_port_attr pattr;
    if (ibv_query_port(ctx, port, &pattr)) {
        perror("ibv_query_port");
        return false;
    }

    // 遍历 GID 表
    for (int i = 0; i < pattr.gid_tbl_len; i++) {
        struct ibv_gid_entry entry;
        
        // 使用 ibv_query_gid_ex 获取详细信息 (flags 传 0 即可)
        // 注意：这是 verbs.h 中的扩展 API
        if (ibv_query_gid_ex(ctx, port, i, &entry, 0) == 0) {
            
            // 检查类型是否为 RoCE v2
            if (entry.gid_type == gid_type) {
                // 再次检查 GID 是否非零 (通常 query_gid_ex 成功的条目都是有效的，但为了保险)
                uint64_t *raw = (uint64_t *)entry.gid.raw;
                if (raw[0] == 0 && raw[1] == 0) continue;

                if (raw[0] != 0 || (raw[1] & 0xffffffff) != 0xffff0000) continue; // not a IP mapped GID

                if (out_gid) {
                    *out_gid = entry.gid;
                }
                // printf("Found RoCE v2 via API at index %d\n", i);
                if (out_gid_index) {
                    *out_gid_index = i;
                }
                return true;
            }
        }
    }

    fprintf(stderr, "Warning: No RoCE v2 GID found on port %d\n", port);
    return false;
}

int main(int argc, char **argv)
{
    if (argc!=3){
        fprintf(stderr,"Usage:%s <size> <repeat>\n",argv[0]);
        return 1;
    }
    size_t SIZE   = atol(argv[1]);
    int    REPEAT = atoi(argv[2]);

    int ret;

    /* 1) 枚举 HCA，分别为 GPU0/GPU1 选最小 PCI 距离 */
    struct ibv_device **devs = ibv_get_device_list(NULL);
    ASSERT(devs != NULL);
    if (!devs){ perror("ibv_get_device_list"); return 1; }

    struct ibv_context *best_ctx[2] = {NULL,NULL};
    int                best_dev[2] = {-1,-1};
    uint8_t            best_port[2]= {0,0};
    uint64_t           best_dist[2]= {UINT64_MAX,UINT64_MAX};

    for(int g=0;g<2;g++){
        uint32_t gd,gb,gr;
        getGpuPci(g,&gd,&gb,&gr);
        for(int i=0;devs[i];i++){
            // printf("1\n");
            struct ibv_context *ctx = ibv_open_device(devs[i]);
            ASSERT(ctx != NULL);
            if (!ctx) continue;
            struct ibv_device_attr dattr;
            ret = ibv_query_device(ctx,&dattr);
            ASSERT(ret == 0);
            for(int p=1;p<=dattr.phys_port_cnt;p++){
                // printf("2\n");
                struct ibv_port_attr pattr;
                if (ibv_query_port(ctx, p, &pattr)) continue;
                // printf("3\n");
                // printf("%d %d\n", pattr.phys_state, IBV_PORT_PHY_LINK_UP);
                if (pattr.state!=IBV_PORT_ACTIVE) continue;
                // printf("4\n");
                uint32_t pd,pb,pr;
                getPortPci(ctx,p,&pd,&pb,&pr);
                uint64_t d = pciDistance(gd,gb,gr,pd,pb,pr);
                if (d<best_dist[g]){
                    if (best_ctx[g]) ibv_close_device(best_ctx[g]);
                    best_ctx[g]=ctx;
                    best_dev[g]=i;
                    best_port[g]=p;
                    best_dist[g]=d;
                } else {
                    ibv_close_device(ctx);
                }
            }
        }
        assert(best_ctx[g]);
        printf("GPU %d -> HCA \"%s\", port %u, dist %lu\n",
               g, ibv_get_device_name(devs[best_dev[g]]),
               best_port[g], best_dist[g]);
    }
    ibv_free_device_list(devs);

    /* fork client/server */
    pid_t pid       = fork();
    int   is_client = (pid==0);
    int   peer_fd   = is_client? tcp_connect() : tcp_listen();
    if (peer_fd<0) exit(1);

    int my_gpu = is_client?0:1;
    struct ibv_context *ctx = best_ctx[my_gpu];
    int my_port = best_port[my_gpu];

    int my_gid_index;
    ibv_gid my_gid;
    if(find_gid_index(ctx, my_port, &my_gid_index, &my_gid) == false) {
        fprintf(stderr, "ibv_query_gid failed\n");
        exit(1);
    }
    printf("gid_index: %d\n", my_gid_index);

    /* 2) CUDA malloc + GDR pin/map */
    cudaSetDevice(my_gpu);
    void *d_buf = NULL;
    cudaMalloc(&d_buf,SIZE);
    if (is_client) cudaMemset(d_buf,0xAB,SIZE);
    // void *d_buf = malloc(SIZE);

    // gdr_t g = gdr_open();
    // if (!g){ fprintf(stderr,"gdr_open failed\n"); return 1; }
    // gdr_mh_t mh;
    // if (gdr_pin_buffer(g,(unsigned long)d_buf,SIZE,0,0,&mh)){
    //     fprintf(stderr,"gdr_pin_buffer failed\n"); return 1;
    // }
    // void *bar_ptr = NULL;
    // if (gdr_map(g,mh,&bar_ptr,SIZE)){
    //     fprintf(stderr,"gdr_map failed\n"); return 1;
    // }

    /* 3) IB 资源 */
    int qsize = 128;

    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    ASSERT(pd != NULL);
    struct ibv_cq *cq = ibv_create_cq(ctx,qsize,NULL,NULL,0);
    ASSERT(cq != NULL);

    struct ibv_qp_init_attr qpia;
    memset(&qpia,0,sizeof qpia);
    qpia.send_cq        = cq;
    qpia.recv_cq        = cq;
    qpia.cap.max_send_wr  = qsize;
    qpia.cap.max_recv_wr  = qsize;
    qpia.cap.max_send_sge = 8;
    qpia.cap.max_recv_sge = 8;
    qpia.qp_type        = IBV_QPT_RC;
    struct ibv_qp *qp = ibv_create_qp(pd,&qpia);
    ASSERT(qp != NULL);

    struct ibv_mr *mr = ibv_reg_mr(pd,d_buf /*bar_ptr*/,SIZE,
        IBV_ACCESS_LOCAL_WRITE|
        IBV_ACCESS_REMOTE_READ|
        IBV_ACCESS_REMOTE_WRITE);
    ASSERT(mr != NULL);

    modify_qp_init(qp,my_port);

    struct ibv_port_attr pattr;
    ibv_query_port(ctx,my_port,&pattr);
    uint16_t my_lid = pattr.lid;

    /* 4) 交换信息 */
    struct exch local, remote;
    local.addr   = (uint64_t)(uintptr_t)d_buf /*bar_ptr*/;
    local.rkey   = mr->rkey;
    local.qp_num = qp->qp_num;
    local.lid    = my_lid;
    memcpy(&local.gid, &my_gid, sizeof(ibv_gid));
    local.pad    = 0;

    if (is_client) {
        write(peer_fd,&local,sizeof local);
        read (peer_fd,&remote,sizeof remote);
    } else {
        read (peer_fd,&remote,sizeof remote);
        write(peer_fd,&local,sizeof local);
    }

    modify_qp_rtr(qp, remote.qp_num, remote.lid, remote.gid, my_port, my_gid_index);
    modify_qp_rts(qp);

    struct ibv_sge sg = {};
    sg.addr   = local.addr;
    sg.length = (uint32_t)SIZE;
    sg.lkey   = mr->lkey;

    struct ibv_send_wr  s_wr = {}, *bad_s=NULL;
    struct ibv_recv_wr  r_wr = {}, *bad_r=NULL;
    struct ibv_wc wc = {};

    memset(&s_wr,0,sizeof s_wr);
    s_wr.wr_id      = 1;
    s_wr.sg_list    = &sg;
    s_wr.num_sge    = 1;
    s_wr.opcode     = IBV_WR_SEND;
    s_wr.send_flags = IBV_SEND_SIGNALED;

    memset(&r_wr,0,sizeof r_wr);
    r_wr.wr_id   = 2;
    r_wr.sg_list = &sg;
    r_wr.num_sge = 1;

    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);

    int num_wr = 0, num_wc = 0;
    for(int i=0;i<REPEAT;i++){
        if (is_client) {
            ret = ibv_post_send(qp,&s_wr,&bad_s);
            ASSERT(ret == 0);
        } else {
            ret = ibv_post_recv(qp,&r_wr,&bad_r);
            ASSERT(ret == 0);
        }
        num_wr ++;
        while(1) {
            ret = ibv_poll_cq(cq,1,&wc);
            if(ret == 0) {
                if((num_wr - num_wc) < qsize) break;
                else continue;
            }
            num_wc ++;
            if (wc.status!=IBV_WC_SUCCESS) {
                fprintf(stderr, "failed op\n");
            }
        }
    }
    while(num_wr - num_wc > 0) {
        ret = ibv_poll_cq(cq,1,&wc);
        if(ret == 0) continue;
        num_wc ++;
        if (wc.status!=IBV_WC_SUCCESS) {
            fprintf(stderr, "failed op\n");
        }
    }

    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt  = t1.tv_sec - t0.tv_sec + (t1.tv_nsec - t0.tv_nsec)/1e9;
    double tpt = SIZE * REPEAT / dt;
    printf("%.2lfGB/s\n", tpt*1e-9);


    /* 6) cleanup */
    close(peer_fd);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);

    // gdr_unmap(g,mh,bar_ptr,SIZE);
    // gdr_unpin_buffer(g,mh);
    // gdr_close(g);

    ibv_close_device(best_ctx[0]);
    ibv_close_device(best_ctx[1]);

    return 0;
}
