// SPDX-License-Identifier: GPL-2.0
#include "sampler.h"
#include "vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>

// Global variables for control (placed in .bss by default which is map-based in libbpf-rs)
volatile __u64 min_sample_interval_ns = 1000000; // 1ms default
volatile __u32 target_pid = 0;
volatile __u32 active_counter_ids[MAX_COUNTERS] = {0};
volatile __u64 prev_counter_values[MAX_CPUS][MAX_COUNTERS] = {0};
volatile bool tracking = false;
volatile bool stopped[MAX_CPUS] = {[0 ... MAX_CPUS - 1] = true};

// Ring Buffer for samples
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024); // 256 KB
} ringbuf SEC(".maps");

// Perf Event Arrays for reading hardware counters
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter0 SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter1 SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter2 SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter3 SEC(".maps");

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

static __always_inline long get_task_state(struct task_struct *t) {
    if (bpf_core_field_exists(t->__state))
        return t->__state;
    return ((struct task_struct___pre_5_14 *)t)->state;
}

static __always_inline void *get_counter(int i) {
    switch (i) {
        case 0:
            return &counter0;
        case 1:
            return &counter1;
        case 2:
            return &counter2;
        case 3:
            return &counter3;
        default:
            return NULL;
    }
}

static __always_inline void set_stopped(u64 idx, bool v) {
    if (idx < MAX_CPUS) {
        stopped[idx] = v;
    }
}

static __always_inline bool handle_resume(__u64 cpu_id, __u32 pid, __u64 now) {
    if (cpu_id >= MAX_CPUS || !stopped[cpu_id]) {
        return false;
    }
    set_stopped(cpu_id, false);

// Snapshot counter baselines so next sample excludes dead-time drift
#pragma unroll
    for (int i = 0; i < MAX_COUNTERS; i++) {
        struct bpf_perf_event_value buf;
        long err = bpf_perf_event_read_value(get_counter(i), cpu_id, &buf, sizeof(buf));
        if (!err) {
            prev_counter_values[cpu_id][i] = buf.counter;
        } else {
            prev_counter_values[cpu_id][i] = 0;
        }
    }

    // Reset start_map timestamp to exclude dead time
    u64 *tsp = bpf_map_lookup_elem(&start_map, &pid);
    if (tsp) {
        bpf_map_update_elem(&start_map, &pid, &now, BPF_EXIST);
    }
    return true;
}

static __always_inline void
record_sample(__u32 pid, __u32 tgid, __u64 now, __u64 delta, __u32 type) {
    struct saccade_sample *s;

    // Reserve space in ring buffer
    s = bpf_ringbuf_reserve(&ringbuf, sizeof(*s), 0);
    if (!s) {
        return;
    }

    s->timestamp_ns = now;
    s->duration_ns = delta;
    s->pid = pid;
    s->cpu_id = bpf_get_smp_processor_id();
    s->type = type;
    bpf_get_current_comm(&s->task, sizeof(s->task));

// Read hardware counters
#pragma unroll
    for (int i = 0; i < MAX_COUNTERS; i++) {
        u32 idx = s->cpu_id;
        if (idx >= MAX_CPUS || i >= MAX_COUNTERS) {
            continue;
        }

        struct bpf_perf_event_value buf;
        long err = bpf_perf_event_read_value(get_counter(i), idx, &buf, sizeof(buf));
        if (err) {
            s->values[i] = 18000000000000000000 - err;
            prev_counter_values[idx][i] = 0;
        } else {
            s->values[i] = buf.counter - prev_counter_values[idx][i];
            prev_counter_values[idx][i] = buf.counter;
        }
        s->events[i] = active_counter_ids[i];
    }

    bpf_ringbuf_submit(s, 0);
}

// Hook: Context Switch
SEC("tp_btf/sched_switch")
int handle__sched_switch(u64 *ctx) {
    struct task_struct *prev = (struct task_struct *)ctx[1];
    struct task_struct *next = (struct task_struct *)ctx[2];
    u32 prev_pid = prev->pid;
    u32 next_pid = next->pid;
    u64 now = bpf_ktime_get_ns();
    u64 cpu_id = bpf_get_smp_processor_id();
    u64 *tsp;

    if (!tracking) {
        set_stopped(cpu_id, true);
        return 0;
    }

    if (handle_resume(cpu_id, prev_pid, now)) {
        // Resumed from stopped — baselines reset, skip flush.
        // prev is being switched out: remove its start_map entry
        // (handle_resume may have reset its timestamp, but prev is
        // no longer on-CPU so leaving it would produce a bogus sample).
        bpf_map_delete_elem(&start_map, &prev_pid);
        // Still handle switch-in for next.
        if (target_pid != 0 && next_pid != target_pid)
            return 0;
        bpf_map_update_elem(&start_map, &next_pid, &now, BPF_ANY);
        return 0;
    }
    set_stopped(cpu_id, false);

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
int handle_timer(struct bpf_perf_event_data *ctx) {
    u64 cpu_id = bpf_get_smp_processor_id();
    u64 now = bpf_ktime_get_ns();
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 tgid = bpf_get_current_pid_tgid();

    if (!tracking) {
        set_stopped(cpu_id, true);
        return 0;
    }

    if (handle_resume(cpu_id, pid, now)) {
        return 0; // Resumed — baseline reset, no sample
    }
    set_stopped(cpu_id, false);

    // This hook fires periodically (e.g. 100Hz) on each CPU.
    // Check if the CURRENT task is being tracked.

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
