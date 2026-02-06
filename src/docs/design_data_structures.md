# eBPF Data Structures Design

## 1. Communication Channels

We need two primary channels between Kernel (eBPF) and Userspace (Rust):

1.  **Data Channel (Kernel -> User)**: High-throughput stream of samples.
2.  **Control Channel (User -> Kernel)**: Low-frequency configuration updates (e.g., sample rate, thresholds).

## 2. Data Channel: Ring Buffer

We propose using `BPF_MAP_TYPE_RINGBUF` (available since Linux 5.8) instead of the older `BPF_MAP_TYPE_PERF_EVENT_ARRAY`.

*   **Why**:
    *   **Memory Efficiency**: Shared memory region, less copying.
    *   **Ordering**: Strict ordering of events across all CPUs (optional but often useful, though per-CPU buffers are also fine).
    *   **Performance**: continuous polling is more efficient.

### BPF Definition

```c
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024); // 256KB buffer
} samples SEC(".maps");
```

### Sample Structure

The `Sample` struct must be ABI-compatible between C and Rust.

```c
// In sampler.h

enum SampleType {
    SAMPLE_TYPE_INTERMEDIATE = 0,
    SAMPLE_TYPE_FLUSH = 1,
};

struct saccade_sample {
    __u64 timestamp_ns;
    __u64 duration_ns;   // For flush: task execution time. For intermediate: window size.
    __u32 pid;
    __u32 cpu_id;
    __u32 type;          // enum SampleType
    __u64 values[4];     // Fixed size array for counter deltas (e.g. 4 metrics)
                         // OR variable length if using ringbuf_reserve flexibility
};
```

*Note: If the number of counters is dynamic, we can use the `ringbuf_reserve` feature to allocate variably sized records, where the header indicates the payload size.*

## 3. Control Channel: Configuration Map

To control the "minimum sample rate" (and other parameters), we use a single-entry Array Map or Global Variables (BSS). Global variables are easiest to use with `libbpf-rs` skeletons.

### BPF Definition (Global Variables)

```c
// In sampler.bpf.c

// Volatile to ensure BPF doesn't optimize it out as constant, 
// allowing userspace to update it via memory mapping or bpf_map_update_elem.
volatile const __u64 min_sample_interval_ns = 1000000; // Default 1ms
volatile const __u32 active_counter_ids[4] = {0, 0, 0, 0}; // IDs of hardware counters to read
```

### Logic Implementation

```c
SEC("perf_event")
int handle_timer(struct bpf_perf_event_data *ctx) {
    __u64 now = bpf_ktime_get_ns();
    
    // Rate Limiting Logic
    if (now - last_sample_time < min_sample_interval_ns) {
        return 0;
    }
    
    // Reserve space in ringbuf
    struct saccade_sample *s = bpf_ringbuf_reserve(&samples, sizeof(*s), 0);
    if (!s) return 0;
    
    s->timestamp_ns = now;
    s->type = SAMPLE_TYPE_INTERMEDIATE;
    // ... fill other fields ...
    
    bpf_ringbuf_submit(s, 0);
    last_sample_time = now;
    return 0;
}
```

## 4. Rust Integration (libbpf-rs)

The Rust side will use the generated skeleton to:

1.  **Consume**: Poll `ringbuf` for `saccade_sample` events.
2.  **Control**: Update `min_sample_interval_ns` by writing to the skeleton's data or map.

```rust
// Logical specificiation for discussion
struct Oculomotor {
    // ...
    fn set_min_sample_rate(&mut self, rate_ns: u64) {
        self.skel.maps_mut().data().min_sample_interval_ns = rate_ns;
    }
}
```
