use crate::config::{EstimatorKind, SchedulerKind};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Top-level CLI entry point holding global flags and the active subcommand.
#[derive(Parser)]
#[command(name = "saccade")]
#[command(about = "Saccade Performance Tool", long_about = None)]
pub struct Cli {
    /// Enable verbose debug output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Config file path (default: saccade.toml in current directory, if present)
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Available saccade subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Generate performance library and save to file
    Generate {
        /// Output file path
        output: PathBuf,
    },
    /// Profile a target process with dynamic counter rotation
    Run {
        /// Use library from specified file
        #[arg(short, long)]
        library: Option<PathBuf>,

        /// Scheduler to use
        #[arg(long)]
        scheduler: Option<SchedulerKind>,

        /// State estimator to use
        #[arg(long)]
        estimator: Option<EstimatorKind>,

        /// Guidance hint for LLM-guided schedulers (e.g. "focus on memory behavior")
        #[arg(long)]
        guidance: Option<String>,

        /// LLM model name, overrides [llm] model in config file
        #[arg(long)]
        llm_model: Option<String>,

        /// LLM inference server base URL, overrides [llm] base_url in config file
        #[arg(long)]
        llm_base_url: Option<String>,

        /// q-schedule: scheduler quantum in nanoseconds
        #[arg(short = 'q', long = "q-schedule")]
        q_schedule: Option<u64>,

        /// q-sample: minimum interval between eBPF-emitted samples, in nanoseconds
        #[arg(long = "q-sample")]
        q_sample: Option<u64>,

        /// q-output: Perfetto emission cadence in nanoseconds (0 = emit every q-schedule)
        #[arg(long = "q-output")]
        q_output: Option<u64>,

        /// Output Perfetto trace file for VCS state
        #[arg(long, default_value = "trace.perfetto")]
        trace: PathBuf,

        /// Output CSV file
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Target program and arguments
        #[arg(last = true, required = true)]
        target: Vec<String>,
    },
    /// Run the target repeatedly, each time with a different fixed batch of 4 counters,
    /// until all available counters have been covered once.
    Sweep {
        /// Use library from specified file
        #[arg(short, long)]
        library: Option<PathBuf>,

        /// q-schedule: scheduler quantum in nanoseconds
        #[arg(short = 'q', long = "q-schedule")]
        q_schedule: Option<u64>,

        /// q-sample: minimum interval between eBPF-emitted samples, in nanoseconds
        #[arg(long = "q-sample")]
        q_sample: Option<u64>,

        /// Output Perfetto trace file with per-event time-varying rates
        #[arg(long, default_value = "trace.perfetto")]
        trace: PathBuf,

        /// Output HDF5 matrix file for ML training data (N_events × T per thread)
        #[arg(long)]
        matrix: Option<PathBuf>,

        /// Suppress the batch progress bar (for use from scripts)
        #[arg(long)]
        quiet: bool,

        /// Target program and arguments
        #[arg(last = true, required = true)]
        target: Vec<String>,
    },
    /// Compare a simulate output trace against a sweep ground-truth trace
    Evaluate {
        /// Ground-truth Perfetto trace (from saccade sweep --trace)
        #[arg(long)]
        ground_truth: PathBuf,

        /// Estimated Perfetto trace (from saccade simulate --trace)
        #[arg(long)]
        estimated: PathBuf,

        /// Time bin width in milliseconds (must be > 0)
        #[arg(long, default_value = "100")]
        bin_ms: u64,

        /// Output results as JSON instead of a text table
        #[arg(long)]
        json: bool,
    },
    /// Run simulation replaying time-varying rates from a sweep trace
    Simulate {
        /// Event library JSON file (required, no perf fallback)
        #[arg(short, long)]
        library: PathBuf,

        /// Perfetto trace file with time-varying rates (from sweep --trace)
        #[arg(short = 'r', long)]
        rates_trace: PathBuf,

        /// Scheduler to use
        #[arg(long)]
        scheduler: Option<SchedulerKind>,

        /// State estimator to use
        #[arg(long)]
        estimator: Option<EstimatorKind>,

        /// Guidance hint for LLM-guided schedulers (e.g. "focus on memory behavior")
        #[arg(long)]
        guidance: Option<String>,

        /// LLM model name, overrides [llm] model in config file
        #[arg(long)]
        llm_model: Option<String>,

        /// LLM inference server base URL, overrides [llm] base_url in config file
        #[arg(long)]
        llm_base_url: Option<String>,

        /// q-schedule: scheduler quantum in nanoseconds
        #[arg(short = 'q', long = "q-schedule")]
        q_schedule: Option<u64>,

        /// q-output: Perfetto emission cadence in nanoseconds (0 = emit every q-schedule)
        #[arg(long = "q-output")]
        q_output: Option<u64>,

        /// q-sample: minimum interval between samples within a scheduling quantum, in nanoseconds
        #[arg(long = "q-sample")]
        q_sample: Option<u64>,

        /// Gaussian noise standard deviation on simulated rates (0 = no noise)
        #[arg(long)]
        noise_stddev: Option<f64>,

        /// RNG seed for reproducible simulation (omit for OS-random)
        #[arg(long)]
        seed: Option<u64>,

        /// Number of hardware counter slots available during simulation
        #[arg(long, default_value_t = 4)]
        num_slots: usize,

        /// Output CSV file
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Output Perfetto trace file for VCS state
        #[arg(long, default_value = "trace.perfetto")]
        trace: PathBuf,

        /// JSON latency profile (from q7_llm_latency.py) — samples override measured LLM call latency
        #[arg(long)]
        llm_latency_profile: Option<PathBuf>,

        /// JSON file listing combos to run in parallel (batch mode).
        /// Each entry: {scheduler, estimator, trace, seed?, guidance?, csv?}.
        /// When set, --scheduler/--estimator/--trace/--seed/--guidance are ignored.
        #[arg(long)]
        batch: Option<PathBuf>,

        /// Number of Rayon threads for batch mode (0 = logical CPU count).
        /// Mirrors the -j flag in the Python layer to keep peak memory bounded.
        #[arg(long, default_value_t = 0)]
        jobs: usize,
    },
}
