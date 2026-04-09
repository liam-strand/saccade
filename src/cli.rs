use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "saccade")]
#[command(about = "Saccade Performance Tool", long_about = None)]
pub struct Cli {
    /// Enable verbose debug output
    #[arg(short, long, global = true)]
    pub verbose: bool,

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

        /// Default Scheduler Quantum (in nanoseconds)
        #[arg(short, long, default_value_t = 10_000_000)]
        quantum: u64,

        /// Output Perfetto trace file for VCS state
        #[arg(long)]
        trace: Option<PathBuf>,

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

        /// Scheduler quantum in nanoseconds
        #[arg(short, long, default_value_t = 10_000_000)]
        quantum: u64,

        /// Output Perfetto trace file with per-event time-varying rates
        #[arg(long)]
        trace: Option<PathBuf>,

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

        /// Scheduler quantum in nanoseconds
        #[arg(short, long, default_value_t = 10_000_000)]
        quantum: u64,

        /// Number of quanta to simulate
        #[arg(short, long, default_value_t = 1000)]
        steps: u64,

        /// Output CSV file (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Scheduler to use: random, round_robin
        #[arg(long, default_value = "random")]
        scheduler: String,

        /// Output Perfetto trace file for VCS state
        #[arg(long)]
        trace: Option<PathBuf>,
    },
}
