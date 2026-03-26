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
        #[arg(short, long, default_value_t = 1000000)]
        quantum: u64,

        /// Target program and arguments
        #[arg(last = true, required = true)]
        target: Vec<String>,
    },
}
