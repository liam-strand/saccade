use serde::Deserialize;
use std::hint::black_box;
use std::time::{Duration, Instant};
use std::{env, fs, process};

#[derive(Deserialize)]
struct WorkloadConfig {
    phases: Vec<Phase>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Phase {
    CacheThrash {
        array_size_kb: usize,
        duration_secs: u64,
    },
    FpHeavy {
        vector_size: usize,
        duration_secs: u64,
    },
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

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

#[inline(never)]
fn run_fp_heavy(vector_size: usize, duration: Duration) {
    let a: Vec<f64> = (0..vector_size)
        .map(|i| (i as f64 + 1.0).sqrt())
        .collect();
    let mut b: Vec<f64> = (0..vector_size)
        .map(|i| (i as f64 + 1.0).cbrt())
        .collect();
    let deadline = Instant::now() + duration;
    let mut sum = 1.0_f64;

    loop {
        for i in 0..vector_size {
            sum = sum * 1.000_000_1 + a[i] * b[i];
        }
        // Feed result back to prevent hoisting; perturbation is tiny enough to be numerically stable
        b[0] += black_box(sum) * 1e-15;
        if Instant::now() >= deadline {
            break;
        }
    }

    black_box(sum);
}

fn validate(config: &WorkloadConfig) -> Result<(), String> {
    if config.phases.is_empty() {
        return Err("phases must not be empty".to_string());
    }
    for (i, phase) in config.phases.iter().enumerate() {
        match phase {
            Phase::CacheThrash {
                array_size_kb,
                duration_secs,
            } => {
                if *array_size_kb == 0 {
                    return Err(format!("phase {i}: array_size_kb must be > 0"));
                }
                if *duration_secs == 0 {
                    return Err(format!("phase {i}: duration_secs must be > 0"));
                }
            }
            Phase::FpHeavy {
                vector_size,
                duration_secs,
            } => {
                if *vector_size == 0 {
                    return Err(format!("phase {i}: vector_size must be > 0"));
                }
                if *duration_secs == 0 {
                    return Err(format!("phase {i}: duration_secs must be > 0"));
                }
            }
        }
    }
    Ok(())
}

fn phase_label(phase: &Phase) -> String {
    match phase {
        Phase::CacheThrash {
            array_size_kb,
            duration_secs,
        } => format!("cache_thrash: array_size_kb={array_size_kb}, duration_secs={duration_secs}"),
        Phase::FpHeavy {
            vector_size,
            duration_secs,
        } => format!("fp_heavy: vector_size={vector_size}, duration_secs={duration_secs}"),
    }
}

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
        eprintln!("[phase {i}/{n}] {}", phase_label(phase));
        match phase {
            Phase::CacheThrash {
                array_size_kb,
                duration_secs,
            } => run_cache_thrash(*array_size_kb, Duration::from_secs(*duration_secs)),
            Phase::FpHeavy {
                vector_size,
                duration_secs,
            } => run_fp_heavy(*vector_size, Duration::from_secs(*duration_secs)),
        }
        eprintln!("[phase {i}/{n}] complete");
    }

    eprintln!("[workload] done");
}
