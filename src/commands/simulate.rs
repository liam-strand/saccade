use crate::commands::load_library;
use crate::config::ResolvedConfig;
use crate::event::EventRegistry;
use crate::perfetto;
use crate::profiler::ProfilerBuilder;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
use crate::sink::{self, OutputSink};
use crate::source::virtual_source::{TimeVaryingRates, VirtualSampleSource};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::debug;

pub fn simulate(
    library: PathBuf,
    rates_trace: PathBuf,
    config: ResolvedConfig,
    csv: Option<PathBuf>,
    trace: PathBuf,
) -> std::io::Result<()> {
    let lib = load_library(Some(library))?;
    let registry = EventRegistry::new(lib);
    let all_ids = registry.get_event_ids();
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();
    debug!("Loaded {} events.", all_ids.len());

    let scheduler = config.build_scheduler(&registry);

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

    // Re-anchor timestamps to t=0 so VirtualSampleSource's clock (which starts
    // at 0) correctly traverses the time-varying rate profile from the trace.
    let min_ts_ns: u64 = series_map
        .values()
        .filter_map(|pts| pts.first().map(|&(ts, _)| ts))
        .min()
        .unwrap_or(0);
    if min_ts_ns > 0 {
        for pts in series_map.values_mut() {
            for (ts, _) in pts.iter_mut() {
                *ts -= min_ts_ns;
            }
        }
    }

    let max_ts_ns = series_map
        .values()
        .filter_map(|pts| pts.last().map(|&(ts, _)| ts))
        .max()
        .unwrap_or(0);
    let steps = max_ts_ns.div_ceil(config.q_schedule_ns.max(1));

    let source = VirtualSampleSource::new(
        TimeVaryingRates { series: series_map },
        config.noise_stddev,
        config.q_schedule_ns,
        config.seed,
        4,
    );

    let mut sinks: Vec<Box<dyn OutputSink>> = Vec::new();
    if let Some(path) = csv {
        sinks.push(Box::new(CsvSink::new(path)?));
    }
    sinks.push(Box::new(PerfettoSink::new(
        trace,
        event_names,
        config.q_output_ns,
    )?));

    let mut profiler = ProfilerBuilder::new()
        .source(source)
        .scheduler_boxed(scheduler, all_ids)
        .map_err(|e| std::io::Error::other(e.to_string()))?
        .estimator_boxed(config.build_estimator())
        .sinks(&mut sinks)
        .build();

    tracing::info!(
        "Simulating {} steps (q_schedule={}ns, q_output={}ns, duration={}ns from input trace)...",
        steps,
        config.q_schedule_ns,
        config.q_output_ns,
        max_ts_ns
    );
    for _ in 0..steps {
        profiler.step();
    }

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

    drop(profiler);
    sink::finish_sinks(&mut sinks);
    tracing::info!("Simulation complete.");

    Ok(())
}
