#pragma once

#include "vmlinux.h"

#define TASK_COMM_LEN 16
#define MAX_COUNTERS 4
#define MAX_CPUS 256

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
    __u32 pad0; // Explicit padding for alignment
    __u64 values[MAX_COUNTERS];
    __u64 events[MAX_COUNTERS];
    char task[TASK_COMM_LEN];
};
