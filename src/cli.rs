use crate::config::{EstimatorKind, SchedulerKind};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate performance library and save to file
    Generate {
        /// Output file path
        output: PathBuf,
    },
    /// Run saccade
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

        /// q-schedule: scheduler quantum in nanoseconds
        #[arg(short = 'q', long = "q-schedule")]
        q_schedule: Option<u64>,

        /// q-output: Perfetto emission cadence in nanoseconds (0 = emit every q-schedule)
        #[arg(long = "q-output")]
        q_output: Option<u64>,

        /// Gaussian noise standard deviation on simulated rates (0 = no noise)
        #[arg(long)]
        noise_stddev: Option<f64>,

        /// RNG seed for reproducible simulation (omit for OS-random)
        #[arg(long)]
        seed: Option<u64>,

        /// Output CSV file
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Output Perfetto trace file for VCS state
        #[arg(long, default_value = "trace.perfetto")]
        trace: PathBuf,
    },
}
