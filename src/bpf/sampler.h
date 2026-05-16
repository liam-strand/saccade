#pragma once

#ifdef __BPF__
#include "vmlinux.h"
#else
#include <stdint.h>
typedef uint8_t __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;
#endif

/* Maximum length of a task command name, matching TASK_COMM_LEN in the kernel. */
#define TASK_COMM_LEN 16
/* Number of hardware performance counter slots rotated simultaneously. */
#define MAX_COUNTERS 4
/* Maximum number of CPUs supported; bounds per-CPU arrays and maps. */
#define MAX_CPUS 256

/* Discriminator stored in saccade_sample.type indicating why a sample was emitted. */
enum SampleType {
    /* Periodic in-flight sample taken while the task is still on-CPU. */
    SAMPLE_TYPE_INTERMEDIATE = 0,
    /* Final sample emitted when a tracked task is switched off-CPU. */
    SAMPLE_TYPE_FLUSH = 1,
    /* Emitted when a CPU resumes from stopped state. Userspace uses
     * the counter values to reset its per-(cpu,slot) baselines and
     * does not emit a RawSample for this record. */
    SAMPLE_TYPE_RESUME = 2,
};

/* One performance sample delivered from kernel to userspace via the ringbuffer. */
struct saccade_sample {
    __u64 timestamp_ns;           // Kernel monotonic time (bpf_ktime_get_ns) at sample emission.
    __u64 duration_ns;            // Time elapsed since the previous sample or switch-in (0 for RESUME).
    __u32 pid;                    // TGID (process ID visible to userspace, i.e. task_struct->tgid).
    __u32 cpu_id;                 // Logical CPU on which this sample was taken.
    __u32 type;                   // Sample discriminator; cast to enum SampleType.
    __u32 tid;                    // Kernel thread ID (task_struct->pid); pid field holds TGID.
    __u64 counters[MAX_COUNTERS]; // Absolute perf counter readings (not deltas); delta computed in userspace.
    __u64 events[MAX_COUNTERS];   // active_counter_ids slot values at sample time, identifying each counter.
    __u8 task[TASK_COMM_LEN];     // Null-terminated task command name from bpf_get_current_comm.
};
