use crate::commands::load_library;
use crate::event::EventRegistry;
use crate::perfetto;
use crate::profiler::ProfilerBuilder;
use crate::scheduler::random::RandomScheduler;
use crate::scheduler::round_robin::RoundRobinScheduler;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
use crate::source::virtual_source::{TimeVaryingRates, VirtualSampleSource};
use crate::state::propagate::PropagateEstimator;
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::debug;

pub fn simulate(
    library: PathBuf,
    rates_trace: PathBuf,
    q_schedule: u64,
    q_output: u64,
    output: Option<PathBuf>,
    scheduler: String,
    trace: Option<PathBuf>,
) -> std::io::Result<()> {
    let lib = load_library(Some(library))?;
    let registry = EventRegistry::new(lib);
    let all_ids = registry.get_event_ids();
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();
    debug!("Loaded {} events.", all_ids.len());

    debug!("Loading rate time-series from {:?}", rates_trace);
    let timeseries = perfetto::read_rate_timeseries(&rates_trace)?;

    let mut series_map: HashMap<(u32, u32), Vec<(u64, f64)>> = HashMap::new();
    for ((name, tid), data) in timeseries.series {
        if let Some(id) = registry.lookup(&name) {
            debug!(
                "Rate series: {} tid={} (id={}) -> {} points",
                name,
                tid,
                id,
                data.len()
            );
            series_map.insert((id, tid), data);
        } else {
            tracing::warn!("Unknown event in rates trace: {}", name);
        }
    }

    let max_ts_ns = series_map
        .values()
        .filter_map(|pts| pts.last().map(|&(ts, _)| ts))
        .max()
        .unwrap_or(0);
    let steps = max_ts_ns.div_ceil(q_schedule.max(1));

    let source = VirtualSampleSource::new(
        TimeVaryingRates { series: series_map },
        0.0,
        q_schedule,
        None,
        4,
    );

    let mut builder = ProfilerBuilder::new()
        .source(source)
        .estimator(PropagateEstimator::new());

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
        builder = builder.add_sink(PerfettoSink::new(path, event_names, q_output)?);
    }

    let mut profiler = builder.build();

    tracing::info!(
        "Simulating {} steps (q_schedule={}ns, q_output={}ns, duration={}ns from input trace)...",
        steps,
        q_schedule,
        q_output,
        max_ts_ns
    );
    for _ in 0..steps {
        profiler.step();
    }
    profiler.finish_sinks();

    let estimator = profiler.estimator();
    eprintln!(
        "\n{:<6} {:<8} {:<14} {:<14} Samples",
        "EvtID", "TID", "Rate (ev/ns)", "Uncertainty"
    );
    eprintln!("{}", "-".repeat(60));
    for (&(tid, event_id), est) in estimator.all_estimates() {
        if est.sample_count > 0 || est.rate > 0.0 {
            eprintln!(
                "{:<6} {:<8} {:<14.6} {:<14.6} {}",
                event_id, tid, est.rate, est.uncertainty, est.sample_count
            );
        }
    }
    tracing::info!("Simulation complete.");

    Ok(())
}
