use crate::commands::load_library;
use crate::event::EventRegistry;
use crate::perfetto;
use crate::profiler::ProfilerBuilder;
use crate::scheduler::random::RandomScheduler;
use crate::scheduler::round_robin::RoundRobinScheduler;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
use crate::source::virtual_source::{TimeVaryingRates, VirtualSampleSource};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::debug;

pub fn simulate(
    library: PathBuf,
    rates_trace: PathBuf,
    quantum: u64,
    steps: u64,
    output: Option<PathBuf>,
    scheduler: String,
    trace: Option<PathBuf>,
) -> std::io::Result<()> {
    let lib = load_library(Some(library))?;
    let registry = EventRegistry::new(lib);
    let num_events = registry.get_event_ids().len();
    let all_ids = registry.get_event_ids();
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();
    debug!("Loaded {} events.", num_events);

    debug!("Loading rate time-series from {:?}", rates_trace);
    let timeseries = perfetto::read_rate_timeseries(&rates_trace)?;

    let mut series_map: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
    for (name, data) in timeseries.series {
        if let Some(id) = registry.lookup(&name) {
            debug!("Rate series: {} (id={}) -> {} points", name, id, data.len());
            series_map.insert(id, data);
        } else {
            tracing::warn!("Unknown event in rates trace: {}", name);
        }
    }

    let source = VirtualSampleSource::new(
        TimeVaryingRates { series: series_map },
        0.0,
        quantum,
        None,
        4,
    );

    let mut builder = ProfilerBuilder::new().num_events(num_events).source(source);

    builder = match scheduler.as_str() {
        "random" => builder.scheduler(RandomScheduler::default(), all_ids),
        "round_robin" => builder.scheduler(RoundRobinScheduler::default(), all_ids),
        other => {
            eprintln!("Unknown scheduler: {}. Using random.", other);
            builder.scheduler(RandomScheduler::default(), all_ids)
        }
    };

    if let Some(path) = output {
        builder = builder.add_sink(CsvSink::new(path)?);
    }
    if let Some(path) = trace {
        builder = builder.add_sink(PerfettoSink::new(path, event_names)?);
    }

    let mut profiler = builder.build();

    tracing::info!("Simulating {} steps (quantum={}ns)...", steps, quantum);
    for _ in 0..steps {
        profiler.step();
    }
    profiler.finish_sinks();

    let vcs = profiler.vcs();
    eprintln!(
        "\n{:<6} {:<14} {:<14} Samples",
        "ID", "Rate (ev/ns)", "Uncertainty"
    );
    eprintln!("{}", "-".repeat(50));
    for (i, est) in vcs.all_estimates().iter().enumerate() {
        if est.sample_count > 0 || est.rate > 0.0 {
            eprintln!(
                "{:<6} {:<14.6} {:<14.6} {}",
                i, est.rate, est.uncertainty, est.sample_count
            );
        }
    }
    tracing::info!("Simulation complete.");

    Ok(())
}
