use clap::Parser;
use saccade::cli::{Cli, Commands};
use saccade::commands::{generate, run, simulate, sweep};

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    match cli.command {
        Commands::Generate { output } => generate(output)?,
        Commands::Run {
            library,
            quantum,
            trace,
            target,
        } => run(library, quantum, trace, target)?,
        Commands::Sweep {
            library,
            quantum,
            trace,
            target,
        } => sweep(library, quantum, trace, target)?,
        Commands::Simulate {
            library,
            rates_trace,
            quantum,
            steps,
            output,
            scheduler,
            trace,
        } => simulate(
            library,
            rates_trace,
            quantum,
            steps,
            output,
            scheduler,
            trace,
        )?,
    }

    Ok(())
}
