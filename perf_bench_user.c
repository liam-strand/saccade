#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

static long
perf_event_open(
    struct perf_event_attr *hw_event,
    pid_t pid,
    int cpu,
    int group_fd,
    unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int
main(int argc, char **argv) {
    struct perf_event_attr pe[5];
    long fds[5];
    unsigned long ids[5] = { 0xc0, 145, 142, 138, 139 };
    int iterations = 10000000;

    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    for (unsigned i = 0; i < 5; i++) {
        struct perf_event_attr *this_pe = &(pe[i]);
        memset(this_pe, 0, sizeof(struct perf_event_attr));
        this_pe->type = PERF_TYPE_RAW;
        this_pe->size = sizeof(struct perf_event_attr);
        this_pe->config = ids[i];
        this_pe->disabled = 1;
        this_pe->exclude_kernel = 1;
        this_pe->exclude_hv = 1;
        // Open for this process, any CPU
        fds[i] = perf_event_open(this_pe, 0, -1, -1, 0);
        if (fds[i] == -1) {
            fprintf(stderr, "Error opening event\n");
            perror("perf_event_open");
            exit(EXIT_FAILURE);
        }
    }
    

    for (unsigned i = 0; i < 4; i++) {
        ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
    }

    printf("Benchmarking %d iterations of ENABLE/DISABLE pairs...\n", iterations);

/* 
    123f:	e8 9c fe ff ff       	call   10e0 <clock_gettime@plt>
    1244:	45 85 e4             	test   %r12d,%r12d
    1247:	7e 2e                	jle    1277 <main+0xf7>
    1249:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
--->1250:	31 d2                	xor    %edx,%edx
    1252:	be 00 24 00 00       	mov    $0x2400,%esi
    1257:	89 ef                	mov    %ebp,%edi
    1259:	31 c0                	xor    %eax,%eax
    125b:	ff c3                	inc    %ebx
    125d:	e8 9e fe ff ff       	call   1100 <ioctl@plt>
    1262:	31 d2                	xor    %edx,%edx
    1264:	be 01 24 00 00       	mov    $0x2401,%esi
    1269:	89 ef                	mov    %ebp,%edi
    126b:	31 c0                	xor    %eax,%eax
    126d:	e8 8e fe ff ff       	call   1100 <ioctl@plt>
    1272:	41 39 dc             	cmp    %ebx,%r12d
<---1275:	75 d9                	jne    1250 <main+0xd0>
    1277:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
    127c:	bf 01 00 00 00       	mov    $0x1,%edi
    1281:	45 01 e4             	add    %r12d,%r12d
    1284:	e8 57 fe ff ff       	call   10e0 <clock_gettime@plt>
*/

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 4; i < iterations + 4; i++) {
        int ret1 = ioctl(fds[(i - 4) % 5], PERF_EVENT_IOC_DISABLE, 0);
        int ret2 = ioctl(fds[i % 5], PERF_EVENT_IOC_ENABLE, 0);
        
        // if (ret1 != 0) {
        //     printf("IOCTL(DISABLE) FAILED\n");
        // }
        // if (ret2 != 0) {
        //     printf("IOCTL(ENABLE) FAILED\n");
        // }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Each iteration has 2 ioctls
    double avg_ns = (elapsed * 1e9) / (iterations * 2);

    printf("Total time: %.6f s\n", elapsed);
    printf("Average time per ioctl: %.2f ns\n", avg_ns);

    for (unsigned i = 0; i < 5; i++) {
        close(fds[i]);
    }

    return 0;
}
