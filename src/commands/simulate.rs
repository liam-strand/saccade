//! Implementation of the `simulate` subcommand: replay a ground-truth Perfetto rate trace through the profiler pipeline without real hardware.

use crate::commands::load_library;
use crate::config::{CliOverrides, EstimatorKind, ResolvedConfig, SchedulerKind, load_config};
use crate::event::{EventId, EventRegistry};
use crate::llm::LlmLatencyProfile;
use crate::perfetto;
use crate::profiler::ProfilerBuilder;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
use crate::sink::{self, OutputSink};
use crate::source::virtual_source::{TimeVaryingRates, VirtualSampleSource};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::debug;

/// Accept both kebab-case ("round-robin") and snake_case ("round_robin") scheduler names in JSON.
fn de_scheduler<'de, D: serde::Deserializer<'de>>(d: D) -> Result<SchedulerKind, D::Error> {
    let s = String::deserialize(d)?;
    let normalized = s.replace('-', "_");
    SchedulerKind::deserialize(serde::de::value::StrDeserializer::<D::Error>::new(
        &normalized,
    ))
}

/// Accept both kebab-case and snake_case estimator names in JSON.
fn de_estimator<'de, D: serde::Deserializer<'de>>(d: D) -> Result<EstimatorKind, D::Error> {
    let s = String::deserialize(d)?;
    let normalized = s.replace('-', "_");
    EstimatorKind::deserialize(serde::de::value::StrDeserializer::<D::Error>::new(
        &normalized,
    ))
}

/// One combo entry in a `--batch` spec JSON file.
#[derive(serde::Deserialize)]
pub struct BatchCombo {
    #[serde(deserialize_with = "de_scheduler")]
    pub scheduler: SchedulerKind,
    #[serde(deserialize_with = "de_estimator")]
    pub estimator: EstimatorKind,
    /// Output Perfetto trace path for this combo.
    pub trace: PathBuf,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub guidance: Option<String>,
    /// Optional CSV output path for this combo.
    #[serde(default)]
    pub csv: Option<PathBuf>,
    /// Optional per-combo TOML config file.  When set, this file is loaded in
    /// place of the global `--config`, letting combos differ in estimator
    /// hyperparameters (e.g. Kalman correlation matrices) while still sharing
    /// the same rates trace and q_schedule/num_slots settings from the base config.
    #[serde(default)]
    pub config: Option<PathBuf>,
}

/// Load the rates trace, filter to known events, re-anchor to t=0, and return an Arc-wrapped TimeVaryingRates.
fn load_rates(
    rates_trace: &Path,
    registry: &EventRegistry,
) -> std::io::Result<(Arc<TimeVaryingRates>, u64)> {
    debug!("Loading rate time-series from {:?}", rates_trace);
    let timeseries = perfetto::read_rate_timeseries(rates_trace)?;

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

    Ok((Arc::new(TimeVaryingRates { series: series_map }), max_ts_ns))
}

/// Run a single simulation combo, writing output to the paths in `combo`.
#[allow(clippy::too_many_arguments)]
fn run_one_combo(
    registry: &EventRegistry,
    all_ids: &[EventId],
    event_names: &[String],
    rates: Arc<TimeVaryingRates>,
    steps: u64,
    base_config: &ResolvedConfig,
    combo: &BatchCombo,
    llm_latency_profile_path: Option<&Path>,
) -> std::io::Result<()> {
    // When the combo carries its own config file (e.g. a different Kalman
    // correlation matrix), reload from that file while preserving the
    // non-hyperparameter fields (quantum sizes, slots, etc.) from base_config.
    let config = if let Some(ref combo_cfg_path) = combo.config {
        load_config(
            Some(combo_cfg_path.clone()),
            true,
            CliOverrides {
                scheduler: Some(combo.scheduler.clone()),
                estimator: Some(combo.estimator.clone()),
                q_schedule_ns: Some(base_config.q_schedule_ns),
                q_sample_ns: Some(base_config.q_sample_ns),
                q_output_ns: Some(base_config.q_output_ns),
                noise_stddev: Some(base_config.noise_stddev),
                seed: combo.seed.or(base_config.seed),
                num_slots: Some(base_config.num_slots),
                guidance: combo
                    .guidance
                    .clone()
                    .or_else(|| base_config.llm.guidance.clone()),
                llm_model: Some(base_config.llm.model.clone()),
                llm_base_url: Some(base_config.llm.base_url.clone()),
                llm_api_key: base_config.llm.api_key.clone(),
            },
        )?
    } else {
        let mut c = base_config.clone();
        c.scheduler = combo.scheduler.clone();
        c.estimator = combo.estimator.clone();
        if let Some(seed) = combo.seed {
            c.seed = Some(seed);
        }
        if let Some(guidance) = &combo.guidance {
            c.llm.guidance = Some(guidance.clone());
        }
        c
    };

    let latency_profile = llm_latency_profile_path
        .map(|p| LlmLatencyProfile::load(p, config.seed))
        .transpose()?;
    let scheduler = config.build_scheduler(registry, true, latency_profile);

    let source = VirtualSampleSource::new(
        rates,
        config.noise_stddev,
        config.q_schedule_ns,
        config.q_sample_ns,
        config.seed,
        config.num_slots,
    );

    let mut sinks: Vec<Box<dyn OutputSink>> = Vec::new();
    if let Some(csv_path) = &combo.csv {
        sinks.push(Box::new(CsvSink::new(csv_path.clone())?));
    }
    sinks.push(Box::new(PerfettoSink::new(
        combo.trace.clone(),
        event_names.to_vec(),
        config.q_output_ns,
    )?));

    let mut profiler = ProfilerBuilder::new()
        .source(source)
        .scheduler_boxed(scheduler, all_ids.to_vec())
        .map_err(|e| std::io::Error::other(e.to_string()))?
        .estimator_boxed(config.build_estimator(registry))
        .sinks(&mut sinks)
        .build();

    for _ in 0..steps {
        profiler.step();
    }

    drop(profiler);
    sink::finish_sinks(&mut sinks);
    Ok(())
}

/// Drives the profiler against a `VirtualSampleSource` seeded from `rates_trace`, running for as many quanta as the trace spans, then writes output to a Perfetto file (and optionally a CSV).
pub fn simulate(
    library: PathBuf,
    rates_trace: PathBuf,
    config: ResolvedConfig,
    csv: Option<PathBuf>,
    trace: PathBuf,
    llm_latency_profile: Option<PathBuf>,
) -> std::io::Result<()> {
    let lib = load_library(Some(library))?;
    let registry = EventRegistry::new(lib);
    let all_ids = registry.get_event_ids();
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();
    debug!("Loaded {} events.", all_ids.len());

    let latency_profile = llm_latency_profile
        .map(|p| LlmLatencyProfile::load(&p, config.seed))
        .transpose()?;
    let scheduler = config.build_scheduler(&registry, true, latency_profile);

    let (rates, max_ts_ns) = load_rates(&rates_trace, &registry)?;
    let steps = max_ts_ns.div_ceil(config.q_schedule_ns.max(1));

    let source = VirtualSampleSource::new(
        rates,
        config.noise_stddev,
        config.q_schedule_ns,
        config.q_sample_ns,
        config.seed,
        config.num_slots,
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
        .estimator_boxed(config.build_estimator(&registry))
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

/// Load rates trace once, then run all combos in `batch_spec` in parallel using a Rayon thread pool capped at `jobs` threads.
///
/// All combos share the same `Arc<TimeVaryingRates>`; only per-combo state (estimator, scheduler, sinks) is allocated per thread.
pub fn batch_simulate(
    library: PathBuf,
    rates_trace: PathBuf,
    base_config: ResolvedConfig,
    batch_spec: PathBuf,
    jobs: usize,
    llm_latency_profile: Option<PathBuf>,
) -> std::io::Result<()> {
    let lib = load_library(Some(library))?;
    let registry = EventRegistry::new(lib);
    let all_ids = registry.get_event_ids();
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();
    debug!("Loaded {} events.", all_ids.len());

    let (rates, max_ts_ns) = load_rates(&rates_trace, &registry)?;
    let steps = max_ts_ns.div_ceil(base_config.q_schedule_ns.max(1));

    let combos: Vec<BatchCombo> = serde_json::from_str(&std::fs::read_to_string(&batch_spec)?)
        .map_err(std::io::Error::other)?;

    tracing::info!(
        "Batch simulation: {} combos, {} steps each, {} Rayon threads",
        combos.len(),
        steps,
        jobs,
    );

    let latency_path = llm_latency_profile.as_deref();

    // Run every combo and collect results rather than short-circuiting: a single combo's
    // failure (e.g. a transient LLM timeout that survives the client's retries) must not
    // discard the other combos that completed successfully.
    let results: Vec<(&BatchCombo, std::io::Result<()>)> = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(std::io::Error::other)?
        .install(|| {
            combos
                .par_iter()
                .map(|combo| {
                    let r = run_one_combo(
                        &registry,
                        &all_ids,
                        &event_names,
                        Arc::clone(&rates),
                        steps,
                        &base_config,
                        combo,
                        latency_path,
                    );
                    (combo, r)
                })
                .collect()
        });

    let mut failures = 0usize;
    for (combo, result) in &results {
        if let Err(e) = result {
            failures += 1;
            tracing::error!(
                scheduler = %combo.scheduler,
                estimator = %combo.estimator,
                trace = ?combo.trace,
                "batch combo failed: {e}"
            );
        }
    }
    let total = results.len();
    let succeeded = total - failures;
    tracing::info!("batch: {succeeded}/{total} combos succeeded");

    // Keep partial results: only fail the process if nothing succeeded (or there was nothing
    // to do). Surviving combos' traces are written and remain usable downstream.
    if succeeded == 0 && total > 0 {
        return Err(std::io::Error::other(format!(
            "all {total} batch combos failed"
        )));
    }
    Ok(())
}
