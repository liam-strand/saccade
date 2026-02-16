// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include "sampler.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>

#define TASK_RUNNING 0

// Global variables for control (placed in .bss by default which is map-based in libbpf-rs)
volatile __u64 min_sample_interval_ns = 1000000; // 1ms default
volatile __u32 target_pid = 0;
volatile __u32 active_counter_ids[MAX_COUNTERS] = {0, 0, 0, 0};

// Ring Buffer for samples
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024); // 256 KB
} ringbuf SEC(".maps");

// Perf Event Array for reading hardware counters
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, TOTAL_COUNTERS);
    __type(key, u32);
    __type(value, u32);
} counters SEC(".maps");

// Map to track the start time and last sample time of target tasks
// Key: PID, Value: Timestamp (ns) of when the task was scheduled in (or last sampled)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, u32);
    __type(value, u64);
} start_map SEC(".maps");

// Helper to determine task state for older kernels
struct task_struct___pre_5_14 {
    long int state;
};

static inline long get_task_state(struct task_struct *t)
{
    if (bpf_core_field_exists(t->__state))
        return t->__state;
    return ((struct task_struct___pre_5_14 *)t)->state;
}

static __always_inline void record_sample(__u32 pid, __u32 tgid, __u64 now, __u64 delta, __u32 type) {
    struct saccade_sample *s;

    // Reserve space in ring buffer
    s = bpf_ringbuf_reserve(&ringbuf, sizeof(*s), 0);
    if (!s)
        {return;}

    s->timestamp_ns = now;
    s->duration_ns = delta;
    s->pid = pid;
    s->cpu_id = bpf_get_smp_processor_id();
    s->type = type;
    bpf_get_current_comm(&s->task, sizeof(s->task));

    // Read hardware counters
    // Iterate 0..MAX_COUNTERS-1. Loop is compatible with verifier limits.
    #pragma unroll
    for (int i = 0; i < MAX_COUNTERS; i++) {
         u32 idx = (s->cpu_id * MAX_COUNTERS) + i;
         
         struct bpf_perf_event_value buf;
         long err = bpf_perf_event_read_value(&counters, idx, &buf, sizeof(buf));
         if (err) {
             s->values[i] = err;
         } else {
             s->values[i] = buf.counter;
         }
    }

    bpf_ringbuf_submit(s, 0);
}

// Hook: Context Switch
SEC("tp_btf/sched_switch")
int handle__sched_switch(u64 *ctx)
{
    struct task_struct *prev = (struct task_struct *)ctx[1];
    struct task_struct *next = (struct task_struct *)ctx[2];
    u32 prev_pid = prev->pid;
    u32 next_pid = next->pid;
    u64 now = bpf_ktime_get_ns();
    u64 *tsp;

    // Handle Switch-OUT (prev)
    tsp = bpf_map_lookup_elem(&start_map, &prev_pid);
    if (tsp) {
        u64 delta = (now - *tsp);
        record_sample(prev_pid, prev->tgid, now, delta, SAMPLE_TYPE_FLUSH);
        bpf_map_delete_elem(&start_map, &prev_pid);
    }

    // Handle Switch-IN (next)
    if (target_pid != 0 && next_pid != target_pid) {
        return 0;
    }
    
    bpf_map_update_elem(&start_map, &next_pid, &now, BPF_ANY);

    return 0;
}

// Hook: Timer (perf_event) for intermediate sampling
SEC("perf_event")
int handle_timer(struct bpf_perf_event_data *ctx)
{
    // This hook fires periodically (e.g. 100Hz) on each CPU.
    // Check if the CURRENT task is being tracked.
    u64 now = bpf_ktime_get_ns();
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 tgid = bpf_get_current_pid_tgid();

    u64 *tsp = bpf_map_lookup_elem(&start_map, &pid);
    if (!tsp) {
        // Not tracking this task (or not switched in via hook)
        return 0;
    }

    u64 last_time = *tsp;
    u64 delta = now - last_time;

    if (delta < min_sample_interval_ns) {
        return 0;
    }

    // Record Intermediate Sample based on delta since last time
    // Capture delta since last sample.
    record_sample(pid, tgid, now, delta, SAMPLE_TYPE_INTERMEDIATE);

    // Update timestamp so next delta is relative to now
    bpf_map_update_elem(&start_map, &pid, &now, BPF_EXIST);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
