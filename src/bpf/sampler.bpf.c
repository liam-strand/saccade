// SPDX-License-Identifier: GPL-2.0
#include "sampler.h"
#include "vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>

/* Control knobs written by userspace via the libbpf-rs .bss map interface. */
volatile __u64 min_sample_interval_ns = 1000000; // Minimum ns between INTERMEDIATE samples (default 1 ms).
volatile __u32 target_tgid = 0;                  // TGID to trace; 0 means trace all processes.
volatile __u32 active_counter_ids[MAX_COUNTERS] = {0}; // Logical event IDs for the currently-loaded counter slots.
volatile bool tracking = false;                  // Master enable; handlers mark CPUs stopped and return early while false.
/* Per-CPU stopped flags; initialized true so every CPU emits a RESUME on first
 * observation, establishing counter baselines before any real samples are recorded. */
volatile bool stopped[MAX_CPUS] = {[0 ... MAX_CPUS - 1] = true};

/* Shared ringbuffer through which samples are delivered to userspace. */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024); // 256 KB
} ringbuf SEC(".maps");

/* Four perf_event_array maps, one per counter slot (counter0..counter3).
 * Each map holds one perf_event fd per CPU, opened by userspace for the
 * hardware event identified by active_counter_ids[i].  Four separate maps
 * are required because BPF cannot index perf_event_array references by a
 * runtime variable; get_counter() dispatches to the right map via switch. */
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter0 SEC(".maps"); // Slot 0: active_counter_ids[0]
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter1 SEC(".maps"); // Slot 1: active_counter_ids[1]
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter2 SEC(".maps"); // Slot 2: active_counter_ids[2]
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, MAX_CPUS);
    __type(key, u32);
    __type(value, u32);
} counter3 SEC(".maps"); // Slot 3: active_counter_ids[3]

/* Tracks the reference timestamp for each tracked kernel thread (key: tid).
 * Set to switch-in time on context switch; updated to now after each INTERMEDIATE sample. */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, u32);
    __type(value, u64);
} start_map SEC(".maps");

/* CO-RE shadow struct for kernels before 5.14, which used task_struct::state
 * instead of the renamed task_struct::__state field. */
struct task_struct___pre_5_14 {
    long int state;
};

/* Returns the scheduler state of t, handling the state→__state rename in 5.14. */
static __always_inline long get_task_state(struct task_struct *t) {
    if (bpf_core_field_exists(t->__state))
        return t->__state;
    return ((struct task_struct___pre_5_14 *)t)->state;
}

/* Returns a pointer to the perf_event_array map for counter slot i (0–3), or
 * NULL for out-of-range indices.  A switch is required because BPF cannot
 * dereference a runtime-computed pointer to a map reference. */
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

/* Writes v to stopped[idx], bounds-checking idx against MAX_CPUS. */
static __always_inline void set_stopped(u64 idx, bool v) {
    if (idx < MAX_CPUS) {
        stopped[idx] = v;
    }
}

/* Reserves a slot in the ringbuffer, fills all saccade_sample fields, and submits it.
 * The `pid` parameter is the kernel thread ID (task_struct->pid, written to s->tid);
 * `tgid` is the process ID (task_struct->tgid, written to s->pid).
 * Counter readings are absolute; delta computation is left to userspace. */
static __always_inline void
record_sample(__u32 pid, __u32 tgid, __u64 now, __u64 delta, __u32 type) {
    struct saccade_sample *s;

    s = bpf_ringbuf_reserve(&ringbuf, sizeof(*s), 0);
    if (!s) {
        return;
    }

    s->timestamp_ns = now;
    s->duration_ns = delta;
    s->pid = tgid;
    s->tid = pid;
    s->cpu_id = bpf_get_smp_processor_id();
    s->type = type;
    bpf_get_current_comm(&s->task, sizeof(s->task));

    // Read absolute hardware counter values into the sample; delta computation happens in userspace.
#pragma unroll
    for (int i = 0; i < MAX_COUNTERS; i++) {
        u32 idx = s->cpu_id;
        if (idx >= MAX_CPUS || i >= MAX_COUNTERS) {
            continue;
        }

        struct bpf_perf_event_value buf;
        long err = bpf_perf_event_read_value(get_counter(i), idx, &buf, sizeof(buf));
        s->counters[i] = err ? 0 : buf.counter;
        s->events[i] = active_counter_ids[i];
    }

    bpf_ringbuf_submit(s, 0);
}

/* Called on the first event seen on a stopped CPU.  Clears the stopped flag,
 * emits a RESUME sample (duration_ns=0) so userspace can reset its per-(cpu,slot)
 * counter baselines, and seeds start_map for the current thread — including threads
 * that were already on-CPU when tracking was enabled and never passed through a
 * switch-in (which would otherwise leave them absent from start_map).
 * Returns true if the CPU was stopped and has now been resumed, false otherwise. */
static __always_inline bool handle_resume(__u64 cpu_id, __u32 pid, __u32 tgid, __u64 now) {
    if (cpu_id >= MAX_CPUS || !stopped[cpu_id]) {
        return false;
    }
    set_stopped(cpu_id, false);

    record_sample(pid, tgid, now, 0, SAMPLE_TYPE_RESUME);

    // Register this thread so timer-based sampling can begin immediately; BPF_ANY
    // creates the entry if absent or overwrites a stale timestamp if already present.
    if (target_tgid == 0 || tgid == target_tgid) {
        bpf_map_update_elem(&start_map, &pid, &now, BPF_ANY);
    }
    return true;
}

/* Fires on every context switch.  Flushes accumulated counter data for the
 * outgoing task (SAMPLE_TYPE_FLUSH) and registers the incoming task in
 * start_map so subsequent timer samples and the next flush can measure it. */
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

    if (handle_resume(cpu_id, prev_pid, prev->tgid, now)) {
        // Baselines reset; prev is leaving CPU so remove its start_map entry to
        // avoid a bogus flush on the next switch involving this tid.
        bpf_map_delete_elem(&start_map, &prev_pid);
        // Still register next for future sampling.
        if (target_tgid != 0 && next->tgid != target_tgid)
            return 0;
        bpf_map_update_elem(&start_map, &next_pid, &now, BPF_ANY);
        return 0;
    }
    set_stopped(cpu_id, false);

    // Switch-OUT: flush counters for prev and remove its start_map entry.
    tsp = bpf_map_lookup_elem(&start_map, &prev_pid);
    if (tsp) {
        u64 delta = (now - *tsp);
        record_sample(prev_pid, prev->tgid, now, delta, SAMPLE_TYPE_FLUSH);
        bpf_map_delete_elem(&start_map, &prev_pid);
    }

    // Switch-IN: register next so timer samples and the eventual flush can measure it.
    if (target_tgid != 0 && next->tgid != target_tgid) {
        return 0;
    }

    bpf_map_update_elem(&start_map, &next_pid, &now, BPF_ANY);

    return 0;
}

/* Fires periodically (e.g. 100 Hz) on each CPU via a perf_event.  If the
 * current task is tracked and at least min_sample_interval_ns has elapsed
 * since the last sample, emits a SAMPLE_TYPE_INTERMEDIATE record and
 * advances the start_map timestamp so the next delta is relative to now. */
SEC("perf_event")
int handle_timer(struct bpf_perf_event_data *ctx) {
    u64 cpu_id = bpf_get_smp_processor_id();
    u64 now = bpf_ktime_get_ns();
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    u32 tid = (u32)pid_tgid;

    if (!tracking) {
        set_stopped(cpu_id, true);
        return 0;
    }

    if (handle_resume(cpu_id, tid, tgid, now)) {
        return 0; // Resumed — baseline reset, no sample
    }
    set_stopped(cpu_id, false);

    u64 *tsp = bpf_map_lookup_elem(&start_map, &tid);
    if (!tsp) {
        // Task is not being tracked (not yet switched in, or belongs to a different process).
        return 0;
    }

    u64 last_time = *tsp;
    u64 delta = now - last_time;

    if (delta < min_sample_interval_ns) {
        return 0;
    }

    record_sample(tid, tgid, now, delta, SAMPLE_TYPE_INTERMEDIATE);

    bpf_map_update_elem(&start_map, &tid, &now, BPF_EXIST);

    return 0;
}

/* BPF program license; must be GPL to access GPL-only kernel helpers. */
char LICENSE[] SEC("license") = "GPL";
