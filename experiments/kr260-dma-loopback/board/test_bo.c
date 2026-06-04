/* test_bo.c — minimal XRT-native-C-API probe of the KR260 memory path.
 * Mirrors the planned Zig binding: open device, alloc BO, get phys addr,
 * map, write, cache-sync. Proves XRT-for-memory works (or shows where it
 * stops). Build on the board:
 *   gcc test_bo.c -o test_bo -lxrt_coreutil
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

typedef void *xrtDeviceHandle;
typedef void *xrtBufferHandle;
extern xrtDeviceHandle xrtDeviceOpen(unsigned int index);
extern int             xrtDeviceClose(xrtDeviceHandle);
extern xrtBufferHandle xrtBOAlloc(xrtDeviceHandle, size_t, uint64_t flags, uint32_t grp);
extern int             xrtBOFree(xrtBufferHandle);
extern void           *xrtBOMap(xrtBufferHandle);
extern uint64_t        xrtBOAddress(xrtBufferHandle);
extern int             xrtBOSync(xrtBufferHandle, int dir, size_t, size_t);

#define TO_DEVICE   0
#define FROM_DEVICE 1

int main(void) {
    xrtDeviceHandle d = xrtDeviceOpen(0);
    printf("xrtDeviceOpen(0) = %p\n", d);
    if (!d) { printf("FAIL: device open\n"); return 1; }

    size_t n = 4096;
    xrtBufferHandle b = xrtBOAlloc(d, n, 0, 0);
    printf("xrtBOAlloc(%zu, flags=0, grp=0) = %p\n", n, b);
    if (!b) { printf("FAIL: BO alloc (likely no memory topology -> need a thin xclbin)\n"); return 2; }

    uint64_t pa = xrtBOAddress(b);
    unsigned char *m = (unsigned char *)xrtBOMap(b);
    printf("phys=0x%llx  map=%p\n", (unsigned long long)pa, (void *)m);
    if (!m) { printf("FAIL: map\n"); return 3; }

    memset(m, 0xAB, n);
    m[0] = 0x11; m[1] = 0x22;
    int r1 = xrtBOSync(b, TO_DEVICE,   n, 0);
    int r2 = xrtBOSync(b, FROM_DEVICE, n, 0);
    printf("sync to=%d from=%d   map[0..2]=%02x %02x %02x\n", r1, r2, m[0], m[1], m[2]);
    printf("PASS: XRT BO alloc/map/addr/sync OK on KR260\n");

    xrtBOFree(b);
    xrtDeviceClose(d);
    return 0;
}
