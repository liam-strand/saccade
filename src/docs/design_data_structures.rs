//! # eBPF Data Structures Design
//!
//! ## 1. Communication Channels
//!
//! We need two primary channels between Kernel (eBPF) and Userspace (Rust):
//!
//! 1.  **Data Channel (Kernel -> User)**: High-throughput stream of samples.
//! 2.  **Control Channel (User -> Kernel)**: Low-frequency configuration updates (e.g., sample rate, thresholds).
//!
//! ## 2. Data Channel: Ring Buffer
//!
//! We propose using `BPF_MAP_TYPE_RINGBUF` (available since Linux 5.8) instead of the older `BPF_MAP_TYPE_PERF_EVENT_ARRAY`.
//!
//! *   **Why**:
//!     *   **Memory Efficiency**: Shared memory region, less copying.
//!     *   **Ordering**: Strict ordering of events across all CPUs.
//!     *   **Performance**: continuous polling is more efficient.
//!
//! ### BPF Definition
//!
//! ```c
//! struct {
//!     __uint(type, BPF_MAP_TYPE_RINGBUF);
//!     __uint(max_entries, 256 * 1024); // 256KB buffer
//! } ringbuf SEC(".maps");
//! ```
//!
//! ### Sample Structure
//!
//! The `Sample` struct must be ABI-compatible between C and Rust.
//!
//! ```c
//! // In sampler.h
//!
//! enum SampleType {
//!     SAMPLE_TYPE_INTERMEDIATE = 0,
//!     SAMPLE_TYPE_FLUSH = 1,
//! };
//!
//! struct saccade_sample {
//!     __u64 timestamp_ns;
//!     __u64 duration_ns;
//!     __u32 pid;
//!     __u32 cpu_id;
//!     __u32 type;          // enum SampleType
//!     __u32 pad;
//!     __u64 values[MAX_COUNTERS]; // 4 metrics
//!     char task[TASK_COMM_LEN];   // Task name
//! };
//! ```
//!
//! ## 3. Control Channel: Configuration Map
//!
//! To control the "minimum sample rate" and other parameters, we use Global Variables (BSS).
//!
//! ### BPF Definition (Global Variables)
//!
//! ```c
//! // In sampler.bpf.c
//!
//! volatile __u64 min_sample_interval_ns = 1000000; // Default 1ms
//! volatile __u32 target_pid = 0;
//! volatile __u32 active_counter_ids[MAX_COUNTERS] = {0, 0, 0, 0};
//! ```
//!
//! ## 4. Hardware Counters Map
//!
//! We use `BPF_MAP_TYPE_PERF_EVENT_ARRAY` to read hardware counters.
//!
//! ### BPF Definition
//!
//! ```c
//! struct {
//!     __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
//!     __uint(max_entries, TOTAL_COUNTERS); // MAX_COUNTERS * MAX_CPUS
//!     __type(key, u32);
//!     __type(value, u32);
//! } counters SEC(".maps");
//! ```
//!
//! Userspace populates this map where key is `(cpu_id * MAX_COUNTERS) + slot_idx`.
//!
