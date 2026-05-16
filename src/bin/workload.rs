//! Standalone synthetic benchmark binary used to exercise distinct hardware performance counter profiles for testing the profiler.

use serde::Deserialize;
use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};
use std::{env, fs, process};

/// Top-level configuration deserialized from the JSON file passed on the command line.
#[derive(Deserialize)]
struct WorkloadConfig {
    /// Ordered list of benchmark phases to execute sequentially.
    phases: Vec<Phase>,
}

/// Configuration for a single benchmark phase.
#[derive(Deserialize)]
struct Phase {
    /// How long to run this phase, in seconds.
    duration_secs: u64,
    /// Number of worker threads to spawn concurrently for this phase.
    threads: usize,
    /// The type of workload to run, along with its parameters.
    #[serde(flatten)]
    kind: PhaseKind,
}

/// Selects the micro-benchmark kernel to run and carries its configuration.
#[derive(Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PhaseKind {
    /// Random-access reads/writes into an array larger than L1/L2 cache to stress the cache hierarchy.
    CacheThrash {
        /// Working-set size in KiB; should exceed the target cache level to be stressed.
        array_size_kb: usize,
    },
    /// Dense FMA-style floating-point work on two vectors to saturate the FP execution units.
    FpHeavy {
        /// Number of `f64` elements in each vector.
        vector_size: usize,
    },
    /// Unpredictable conditional branches dispatched through non-inlined call sites to saturate the branch predictor.
    BranchMispredict {
        /// Number of random branches issued per inner loop iteration.
        array_size: usize,
    },
    /// Random page-order touches across a large allocation to evict TLB entries.
    TlbThrash {
        /// Number of 4 KiB pages to allocate and walk randomly.
        num_pages: usize,
    },
    /// Sequential read-modify-write sweep over a buffer to saturate memory bandwidth.
    MemStream {
        /// Buffer size in MiB; should exceed the last-level cache to measure DRAM bandwidth.
        buffer_size_mb: usize,
    },
    /// Integer division with a randomised divisor pool to stress the integer division unit.
    IntDiv {
        /// Upper bound (exclusive) on the randomly generated divisors; must be > 0.
        divisor_range: usize,
    },
}

/// 64-bit xorshift PRNG; updates `state` in place and returns the next pseudo-random value.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Runs the cache-thrash workload for `duration`, doing random reads/writes across an `array_size_kb` KiB buffer.
#[inline(never)]
fn run_cache_thrash(array_size_kb: usize, duration: Duration) {
    let len = (array_size_kb * 1024) / 8;
    let mut data: Vec<u64> = (0..len as u64).collect();
    let mut rng = 0xdeadbeef_u64;
    let mut sum = 0_u64;
    let deadline = Instant::now() + duration;
    let mut iter = 0_u64;

    loop {
        let idx = (xorshift64(&mut rng) as usize) % len;
        sum = sum.wrapping_add(data[idx]);
        data[idx] = data[idx].wrapping_add(1);
        iter += 1;
        if iter & 0x3FF == 0 && Instant::now() >= deadline {
            break;
        }
    }

    black_box(sum);
}

/// Runs the floating-point workload for `duration`, repeatedly computing dot-product-style FMA operations over two `f64` vectors of length `vector_size`.
#[inline(never)]
fn run_fp_heavy(vector_size: usize, duration: Duration) {
    let a: Vec<f64> = (0..vector_size).map(|i| (i as f64 + 1.0).sqrt()).collect();
    let mut b: Vec<f64> = (0..vector_size).map(|i| (i as f64 + 1.0).cbrt()).collect();
    let deadline = Instant::now() + duration;
    let mut sum = 1.0_f64;
    let mut iter = 0_u64;

    loop {
        for i in 0..vector_size {
            sum = sum * 1.000_000_1 + a[i] * b[i];
        }
        // Feed result back to prevent hoisting; perturbation is tiny enough to be numerically stable
        b[0] += black_box(sum) * 1e-15;
        iter += 1;
        if iter & 0x3FF == 0 && Instant::now() >= deadline {
            break;
        }
    }

    black_box(sum);
}

// Separate non-inlined functions for each branch path.
// The compiler cannot speculatively call both and then cmov-select the result,
// so these force a real conditional branch instruction at the call site.
/// Branch-mispredict helper: wrapping addition path called from `run_branch_mispredict`.
#[inline(never)]
fn bp_add(sum: u64, val: u64) -> u64 {
    sum.wrapping_add(val)
}

/// Branch-mispredict helper: wrapping multiplication path called from `run_branch_mispredict`.
#[inline(never)]
fn bp_mul(sum: u64, val: u64) -> u64 {
    sum.wrapping_mul(val | 1)
}

/// Runs the branch-mispredict workload for `duration`, issuing `batch_size` unpredictable conditional branches per inner loop.
#[inline(never)]
fn run_branch_mispredict(batch_size: usize, duration: Duration) {
    let mut rng = 0xdeadbeef_u64;
    let deadline = Instant::now() + duration;
    let mut sum = 1_u64;
    let mut iter = 0_u64;

    loop {
        // Generate branch conditions on-the-fly so the predictor cannot
        // learn a fixed pattern from a pre-computed array.
        for _ in 0..batch_size {
            let val = xorshift64(&mut rng);
            sum = if val & 128 != 0 {
                bp_add(sum, val)
            } else {
                bp_mul(sum, val)
            };
        }
        iter += 1;
        if iter & 0x3FF == 0 && Instant::now() >= deadline {
            break;
        }
    }

    black_box(sum);
}

/// Runs the TLB-thrash workload for `duration`, accessing `num_pages` pages in a shuffled order to evict TLB entries.
#[inline(never)]
fn run_tlb_thrash(num_pages: usize, duration: Duration) {
    let total_bytes = num_pages * 4096;
    let mut data = vec![0u8; total_bytes];
    // Initialize one byte per page to prevent zero-page deduplication
    let mut rng = 0xdeadbeef_u64;
    for i in 0..num_pages {
        data[i * 4096] = xorshift64(&mut rng) as u8;
    }

    // Fisher-Yates shuffle for random page access order
    let mut page_order: Vec<usize> = (0..num_pages).collect();
    for i in (1..num_pages).rev() {
        let j = (xorshift64(&mut rng) as usize) % (i + 1);
        page_order.swap(i, j);
    }

    let deadline = Instant::now() + duration;
    let mut sum = 0_u64;
    let mut iter = 0_u64;

    loop {
        for &page_idx in &page_order {
            let offset = page_idx * 4096;
            sum = sum.wrapping_add(data[offset] as u64);
            data[offset] = data[offset].wrapping_add(1);
        }
        iter += 1;
        if iter & 0x3FF == 0 && Instant::now() >= deadline {
            break;
        }
    }

    black_box(sum);
}

/// Runs the memory-streaming workload for `duration`, sweeping a `buffer_size_mb` MiB buffer sequentially to saturate DRAM bandwidth.
#[inline(never)]
fn run_mem_stream(buffer_size_mb: usize, duration: Duration) {
    let len = buffer_size_mb * 1024 * 1024 / 8;
    let mut data: Vec<u64> = (0..len as u64).collect();
    let deadline = Instant::now() + duration;
    let mut sum = 0_u64;
    let mut iter = 0_u64;

    loop {
        for item in data.iter_mut().take(len) {
            sum = sum.wrapping_add(*item);
            *item = sum;
        }
        iter += 1;
        // Fewer iterations between time checks since each pass is slow for large buffers
        if iter & 0x3 == 0 && Instant::now() >= deadline {
            break;
        }
    }

    black_box(sum);
}

/// Runs the integer-division workload for `duration`, repeatedly dividing a LCG-generated dividend by a pre-built pool of random divisors bounded by `divisor_range`.
#[inline(never)]
fn run_int_div(divisor_range: usize, duration: Duration) {
    let mut rng = 0xdeadbeef_u64;
    let divisor_count = 1024;
    let divisors: Vec<u64> = (0..divisor_count)
        .map(|_| (xorshift64(&mut rng) % divisor_range as u64) + 1)
        .collect();

    let deadline = Instant::now() + duration;
    let mut dividend = 0xcafebabe_u64;
    let mut sum = 0_u64;
    let mut iter = 0_u64;

    loop {
        let idx = (iter as usize) % divisor_count;
        dividend = dividend.wrapping_mul(6364136223846793005).wrapping_add(1);
        let quotient = dividend / divisors[idx];
        let remainder = dividend % divisors[idx];
        sum = sum.wrapping_add(quotient).wrapping_add(remainder);
        iter += 1;
        if iter & 0x3FF == 0 && Instant::now() >= deadline {
            break;
        }
    }

    black_box(sum);
}

/// Spawns `threads` worker threads that each execute the workload described by `kind` for `duration`, then joins all threads.
fn run_phase(kind: &PhaseKind, duration: Duration, threads: usize) {
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let kind = kind.clone();
            thread::spawn(move || match kind {
                PhaseKind::CacheThrash { array_size_kb } => {
                    run_cache_thrash(array_size_kb, duration)
                }
                PhaseKind::FpHeavy { vector_size } => run_fp_heavy(vector_size, duration),
                PhaseKind::BranchMispredict { array_size } => {
                    run_branch_mispredict(array_size, duration)
                }
                PhaseKind::TlbThrash { num_pages } => run_tlb_thrash(num_pages, duration),
                PhaseKind::MemStream { buffer_size_mb } => run_mem_stream(buffer_size_mb, duration),
                PhaseKind::IntDiv { divisor_range } => run_int_div(divisor_range, duration),
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

/// Checks that all phases have non-zero durations, thread counts, and workload-specific parameters.
fn validate(config: &WorkloadConfig) -> Result<(), String> {
    if config.phases.is_empty() {
        return Err("phases must not be empty".to_string());
    }
    for (i, phase) in config.phases.iter().enumerate() {
        if phase.duration_secs == 0 {
            return Err(format!("phase {i}: duration_secs must be > 0"));
        }
        if phase.threads == 0 {
            return Err(format!("phase {i}: threads must be > 0"));
        }
        match &phase.kind {
            PhaseKind::CacheThrash { array_size_kb } => {
                if *array_size_kb == 0 {
                    return Err(format!("phase {i}: array_size_kb must be > 0"));
                }
            }
            PhaseKind::FpHeavy { vector_size } => {
                if *vector_size == 0 {
                    return Err(format!("phase {i}: vector_size must be > 0"));
                }
            }
            PhaseKind::BranchMispredict { array_size } => {
                if *array_size == 0 {
                    return Err(format!("phase {i}: array_size must be > 0"));
                }
            }
            PhaseKind::TlbThrash { num_pages } => {
                if *num_pages == 0 {
                    return Err(format!("phase {i}: num_pages must be > 0"));
                }
            }
            PhaseKind::MemStream { buffer_size_mb } => {
                if *buffer_size_mb == 0 {
                    return Err(format!("phase {i}: buffer_size_mb must be > 0"));
                }
            }
            PhaseKind::IntDiv { divisor_range } => {
                if *divisor_range == 0 {
                    return Err(format!("phase {i}: divisor_range must be > 0"));
                }
            }
        }
    }
    Ok(())
}

/// Formats a human-readable one-line description of `phase` for progress output.
fn phase_label(phase: &Phase) -> String {
    let kind_str = match &phase.kind {
        PhaseKind::CacheThrash { array_size_kb } => {
            format!("cache_thrash: array_size_kb={array_size_kb}")
        }
        PhaseKind::FpHeavy { vector_size } => format!("fp_heavy: vector_size={vector_size}"),
        PhaseKind::BranchMispredict { array_size } => {
            format!("branch_mispredict: batch_size={array_size}")
        }
        PhaseKind::TlbThrash { num_pages } => format!("tlb_thrash: num_pages={num_pages}"),
        PhaseKind::MemStream { buffer_size_mb } => {
            format!("mem_stream: buffer_size_mb={buffer_size_mb}")
        }
        PhaseKind::IntDiv { divisor_range } => format!("int_div: divisor_range={divisor_range}"),
    };
    format!(
        "{kind_str}, duration_secs={}, threads={}",
        phase.duration_secs, phase.threads
    )
}

/// Entry point: reads and validates a JSON config file, then runs each phase in sequence.
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <config.json>", args[0]);
        process::exit(1);
    }

    let config_str = fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("error reading {}: {e}", args[1]);
        process::exit(1);
    });

    let config: WorkloadConfig = serde_json::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!("error parsing config: {e}");
        process::exit(1);
    });

    if let Err(e) = validate(&config) {
        eprintln!("invalid config: {e}");
        process::exit(1);
    }

    let n = config.phases.len();
    eprintln!("[workload] pid={}, phases={n}", process::id());

    for (i, phase) in config.phases.iter().enumerate() {
        let idx = i + 1;
        eprintln!("[phase {idx}/{n}] {}", phase_label(phase));
        run_phase(
            &phase.kind,
            Duration::from_secs(phase.duration_secs),
            phase.threads,
        );
    }

    eprintln!("[workload] done");
}
