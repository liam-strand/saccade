#pragma once

#include "vmlinux.h"

#define TASK_COMM_LEN 16
#define MAX_COUNTERS 4
#define MAX_CPUS 256
#define TOTAL_COUNTERS (MAX_COUNTERS * MAX_CPUS)

enum SampleType {
    SAMPLE_TYPE_INTERMEDIATE = 0,
    SAMPLE_TYPE_FLUSH = 1,
};

struct saccade_sample {
    __u64 timestamp_ns;
    __u64 duration_ns;
    __u32 pid;
    __u32 cpu_id;
    __u32 type;
    __u32 pad; // Explicit padding for alignment if needed, though u64 alignment is usually fine.
    __u64 values[MAX_COUNTERS];
    char task[TASK_COMM_LEN];
};
