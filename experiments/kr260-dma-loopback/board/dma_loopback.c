/* dma_loopback.c - full KR260 XRT BO + MMIO DMA data-path test.
 *
 * Combines the two penzai planes:
 *   memory  -> XRT BO (src/dst buffers, physical addr, cache sync)
 *   control -> /dev/mem MMIO programming of the AXI DMA (PG021 direct mode)
 * then verifies dst == src after MM2S -> FIFO -> S2MM loopback.
 *
 * Uses the 40-bit SA/DA MSB registers (BOs land in high DDR > 4 GB).
 * Needs root for /dev/mem:
 *   gcc dma_loopback.c -o dma_loopback -lxrt_coreutil && sudo ./dma_loopback
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

typedef void *xrtDeviceHandle;
typedef void *xrtBufferHandle;
extern xrtDeviceHandle xrtDeviceOpen(unsigned int);
extern int             xrtDeviceClose(xrtDeviceHandle);
extern xrtBufferHandle xrtBOAlloc(xrtDeviceHandle, size_t, uint64_t, uint32_t);
extern int             xrtBOFree(xrtBufferHandle);
extern void           *xrtBOMap(xrtBufferHandle);
extern uint64_t        xrtBOAddress(xrtBufferHandle);
extern int             xrtBOSync(xrtBufferHandle, int, size_t, size_t);
#define TO_DEVICE   0
#define FROM_DEVICE 1

#define DMA_BASE 0xA0000000UL
#define DMA_SPAN 0x10000

/* PG021 direct register mode, word (uint32) indices */
enum { MM2S_DMACR=0x00/4, MM2S_DMASR=0x04/4, MM2S_SA=0x18/4, MM2S_SA_MSB=0x1C/4,
       MM2S_LENGTH=0x28/4, S2MM_DMACR=0x30/4, S2MM_DMASR=0x34/4, S2MM_DA=0x48/4,
       S2MM_DA_MSB=0x4C/4, S2MM_LENGTH=0x58/4 };
#define RS   (1u<<0)
#define IDLE (1u<<1)
#define ERRM (0x70u)

static volatile uint32_t *dma;

static int wait_idle(int sr, const char *who) {
    for (long i = 0; i < 100000000L; i++) {
        uint32_t s = dma[sr];
        if (s & ERRM) { printf("DMA %s ERROR SR=0x%08x\n", who, s); return -1; }
        if (s & IDLE) return 0;
    }
    printf("DMA %s TIMEOUT SR=0x%08x\n", who, dma[sr]); return -1;
}

int main(void) {
    const size_t n = 4096;
    xrtDeviceHandle d = xrtDeviceOpen(0);
    if (!d) { printf("FAIL: device open\n"); return 1; }
    xrtBufferHandle src = xrtBOAlloc(d, n, 0, 0), dst = xrtBOAlloc(d, n, 0, 0);
    if (!src || !dst) { printf("FAIL: BO alloc\n"); return 2; }

    uint64_t sp = xrtBOAddress(src), dp = xrtBOAddress(dst);
    unsigned char *sm = xrtBOMap(src), *dm = xrtBOMap(dst);
    for (size_t i = 0; i < n; i++) { sm[i] = (unsigned char)(i * 7 + 1); dm[i] = 0; }
    xrtBOSync(src, TO_DEVICE, n, 0);
    xrtBOSync(dst, TO_DEVICE, n, 0);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 3; }
    dma = mmap(NULL, DMA_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DMA_BASE);
    if (dma == MAP_FAILED) { perror("mmap"); return 3; }

    printf("src_phys=0x%llx  dst_phys=0x%llx  n=%zu\n",
           (unsigned long long)sp, (unsigned long long)dp, n);

    dma[MM2S_DMACR] |= RS;
    dma[S2MM_DMACR] |= RS;
    /* arm S2MM (receiver) before triggering MM2S (sender) */
    dma[S2MM_DA]     = (uint32_t)(dp & 0xFFFFFFFF);
    dma[S2MM_DA_MSB] = (uint32_t)(dp >> 32);
    dma[S2MM_LENGTH] = (uint32_t)n;
    dma[MM2S_SA]     = (uint32_t)(sp & 0xFFFFFFFF);
    dma[MM2S_SA_MSB] = (uint32_t)(sp >> 32);
    dma[MM2S_LENGTH] = (uint32_t)n;   /* writing LENGTH triggers the transfer */

    if (wait_idle(MM2S_DMASR, "MM2S")) return 4;
    if (wait_idle(S2MM_DMASR, "S2MM")) return 4;

    xrtBOSync(dst, FROM_DEVICE, n, 0);
    if (memcmp(sm, dm, n) == 0) {
        printf("PASS: DMA loopback src==dst (%zu bytes through MM2S->FIFO->S2MM)\n", n);
        return 0;
    }
    printf("FAIL: mismatch. first bytes src/dst:");
    for (int i = 0; i < 8; i++) printf(" %02x/%02x", sm[i], dm[i]);
    printf("\n");
    return 5;
}
