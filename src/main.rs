use clap::Parser;
use saccade::cli::{Cli, Commands};
use saccade::commands::{evaluate, generate, run, simulate, sweep};
use saccade::config::{CliOverrides, load_config};

/// Entry point: parses CLI arguments, initialises the tracing subscriber, and dispatches to the appropriate subcommand handler.
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

    let explicit = cli.config.is_some();
    let config_path = cli.config.clone();

    match cli.command {
        Commands::Generate { output } => generate(output)?,
        Commands::Evaluate {
            ground_truth,
            estimated,
            bin_ms,
            json,
        } => {
            if bin_ms == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--bin-ms must be > 0",
                ));
            }
            evaluate(ground_truth, estimated, bin_ms, json)?;
        }
        Commands::Run {
            library,
            scheduler,
            estimator,
            guidance,
            q_schedule,
            q_sample,
            q_output,
            trace,
            csv,
            target,
        } => {
            let config = load_config(
                config_path,
                explicit,
                CliOverrides {
                    scheduler,
                    estimator,
                    q_schedule_ns: q_schedule,
                    q_sample_ns: q_sample,
                    q_output_ns: q_output,
                    noise_stddev: None,
                    seed: None,
                    guidance,
                },
            )?;
            run(library, config, trace, csv, target)?;
        }
        Commands::Sweep {
            library,
            q_schedule,
            q_sample,
            trace,
            matrix,
            quiet,
            target,
        } => {
            let config = load_config(
                config_path,
                explicit,
                CliOverrides {
                    scheduler: None,
                    estimator: None,
                    q_schedule_ns: q_schedule,
                    q_sample_ns: q_sample,
                    q_output_ns: None,
                    noise_stddev: None,
                    seed: None,
                    guidance: None,
                },
            )?;
            sweep(library, config, trace, matrix, quiet, target)?;
        }
        Commands::Simulate {
            library,
            rates_trace,
            scheduler,
            estimator,
            guidance,
            q_schedule,
            q_output,
            noise_stddev,
            seed,
            csv,
            trace,
        } => {
            let config = load_config(
                config_path,
                explicit,
                CliOverrides {
                    scheduler,
                    estimator,
                    q_schedule_ns: q_schedule,
                    q_sample_ns: None,
                    q_output_ns: q_output,
                    noise_stddev,
                    seed,
                    guidance,
                },
            )?;
            simulate(library, rates_trace, config, csv, trace)?;
        }
    }

    Ok(())
}
