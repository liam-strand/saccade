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
//! We use `BPF_MAP_TYPE_RINGBUF` (available since Linux 5.8) instead of the older `BPF_MAP_TYPE_PERF_EVENT_ARRAY`.
//!
//! *   **Why**:
//!     *   **Memory Efficiency**: Shared memory region, less copying.
//!     *   **Performance**: Continuous polling is more efficient than per-event wakeups.
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
//! The `saccade_sample` struct is ABI-compatible between C (`sampler.h`) and
//! Rust (`WireSample` in `src/sample.rs`). In fact, `WireSample` is generated
//! from `saccade_sample` at compile-time! Counters hold **absolute** perf
//! counter readings; delta computation happens in userspace.
//!
//! ```c
//! // In sampler.h
//!
//! enum SampleType {
//!     SAMPLE_TYPE_INTERMEDIATE = 0, // Periodic in-flight sample
//!     SAMPLE_TYPE_FLUSH = 1,        // Context-switch-out sample
//!     SAMPLE_TYPE_RESUME = 2,       // Counter baseline reset; produces no RawSample
//! };
//!
//! struct saccade_sample {
//!     __u64 timestamp_ns;             // Kernel monotonic time at sample emission
//!     __u64 duration_ns;              // Interval duration (0 for RESUME)
//!     __u32 pid;                      // TGID (userspace process ID)
//!     __u32 cpu_id;                   // Logical CPU index
//!     __u32 type;                     // enum SampleType discriminant
//!     __u32 tid;                      // Kernel thread ID (task_struct->pid)
//!     __u64 counters[MAX_COUNTERS];   // Absolute perf counter readings (not deltas)
//!     __u64 events[MAX_COUNTERS];     // active_counter_ids at sample time
//!     __u8  task[TASK_COMM_LEN];       // Null-terminated task comm string
//! };
//! ```
//!
//! ## 3. Control Channel: Global Variables
//!
//! Userspace controls sampling behavior by writing to BPF global variables
//! (exposed via the libbpf-rs `.bss` and `.data` map interfaces).
//!
//! ```c
//! // In sampler.bpf.c
//!
//! volatile __u64 min_sample_interval_ns = 1000000; // Minimum ns between INTERMEDIATE samples (default 1 ms)
//! volatile __u32 target_tgid = 0;                  // TGID to trace; 0 = trace all
//! volatile __u32 active_counter_ids[MAX_COUNTERS];  // Logical event IDs for each counter slot
//! volatile bool  tracking = false;                  // Master enable; BPF hooks are no-ops while false
//! volatile bool  stopped[MAX_CPUS];                 // Per-CPU stopped flags; set when tracking=false
//! ```
//!
//! `tracking` and `stopped` underpin the world-stop mechanism: userspace sets
//! `tracking = false`, spins until all `stopped[cpu]` flags are true, reconfigures
//! the counter slots, then sets `tracking = true` to resume.
//!
//! ## 4. Hardware Counters Map
//!
//! We use multiple `BPF_MAP_TYPE_PERF_EVENT_ARRAY`s to read hardware counters.
//!
//! ### BPF Definition
//!
//! ```c
//! struct {
//!     __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
//!     __uint(max_entries, MAX_CPUS);
//!     __type(key, u32);
//!     __type(value, u32);
//! } counter0 SEC(".maps");
//! // ... counter1, counter2, counter3
//! ```
//!
//! Userspace populates these maps where the key is simply `cpu_id`.
//!
