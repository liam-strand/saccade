//! Configuration loading and resolution for the Saccade profiler.
//!
//! Settings are merged from three layers in increasing priority:
//! hard-coded defaults → TOML file → CLI overrides.
//! The final merged state is a [`crate::config::ResolvedConfig`] that the rest of the
//! application consumes.

use crate::event::{EventId, EventRegistry};
use crate::llm::LlmClient;
use crate::scheduler::Scheduler;
use crate::scheduler::max_uncertainty::MaxUncertaintyScheduler;
use crate::scheduler::dynamic_llm::DynamicLlmScheduler;
use crate::scheduler::random::RandomScheduler;
use crate::scheduler::rate_of_change::RateOfChangeScheduler;
use crate::scheduler::round_robin::RoundRobinScheduler;
use crate::scheduler::static_llm::StaticLlmScheduler;
use crate::scheduler::weighted_round_robin_llm::WeightedRoundRobinLlmScheduler;
use crate::state::StateEstimator;
use crate::state::ema::{EmaConfig, VirtualCounterState};
use crate::state::kalman::{KalmanConfig, KalmanFilterEstimator};
use crate::state::propagate::PropagateEstimator;
use config::{Config, File, FileFormat};
use std::fmt;
use std::io;
use std::path::PathBuf;

/// Selects which counter-rotation scheduler algorithm to use.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerKind {
    /// Selects the next counter set uniformly at random each quantum.
    Random,
    /// Cycles through all counter sets in a fixed order.
    RoundRobin,
    /// Selects the counters that are most uncertain at the time of the decision.
    MaxUncertainty,
    /// Queries an LLM once at startup to produce a static counter schedule.
    StaticLlm,
    /// Re-queries an LLM periodically to adapt the counter schedule at runtime.
    DynamicLlm,
    /// Round-robins through LLM-assigned per-counter weights.
    WeightedRoundRobinLlm,
    /// Prioritizes events with the highest rate-of-change (non-linearity) using Lim 2014 triangle cost.
    RateOfChange,
}

impl fmt::Display for SchedulerKind {
    /// Formats the variant as the snake_case string used in TOML and CLI flags.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulerKind::Random => write!(f, "random"),
            SchedulerKind::RoundRobin => write!(f, "round_robin"),
            SchedulerKind::MaxUncertainty => write!(f, "max_uncertainty"),
            SchedulerKind::StaticLlm => write!(f, "static_llm"),
            SchedulerKind::DynamicLlm => write!(f, "dynamic_llm"),
            SchedulerKind::WeightedRoundRobinLlm => write!(f, "weighted_round_robin_llm"),
            SchedulerKind::RateOfChange => write!(f, "rate_of_change"),
        }
    }
}

/// LLM connection and behaviour settings shared by all LLM-backed schedulers.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LlmConfig {
    /// Base URL of the Ollama-compatible inference server.
    #[serde(default = "LlmConfig::default_base_url")]
    pub base_url: String,
    /// Name of the model to request from the server.
    #[serde(default = "LlmConfig::default_model")]
    pub model: String,
    /// How many scheduling quanta elapse between LLM re-queries (used by `DynamicLlm`).
    #[serde(default = "LlmConfig::default_update_interval")]
    pub update_interval: u32,
    /// Optional free-text hint forwarded to the LLM to steer its counter selection.
    #[serde(default)]
    pub guidance: Option<String>,
}

impl LlmConfig {
    /// Returns the default inference server URL.
    fn default_base_url() -> String {
        "http://dubliner.cs.northwestern.edu:11434".into()
    }

    /// Returns the default model name.
    fn default_model() -> String {
        "gemma4".into()
    }

    /// Returns the default number of quanta between LLM re-queries.
    fn default_update_interval() -> u32 {
        1000
    }
}

impl Default for LlmConfig {
    /// Constructs an `LlmConfig` with all default field values.
    fn default() -> Self {
        Self {
            base_url: Self::default_base_url(),
            model: Self::default_model(),
            update_interval: Self::default_update_interval(),
            guidance: None,
        }
    }
}

/// Selects which state-estimation algorithm fills in unobserved counter values.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum EstimatorKind {
    /// Carries the last observed value forward unchanged until the counter is re-sampled.
    Propagate,
    /// Applies an exponential moving average to smooth counter readings over time.
    Ema,
    /// Uses a Kalman filter with optional cross-counter correlation to estimate unsampled values.
    Kalman,
}

impl fmt::Display for EstimatorKind {
    /// Formats the variant as the snake_case string used in TOML and CLI flags.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EstimatorKind::Propagate => write!(f, "propagate"),
            EstimatorKind::Ema => write!(f, "ema"),
            EstimatorKind::Kalman => write!(f, "kalman"),
        }
    }
}

/// Fully-resolved profiler configuration, ready for use by the rest of the application.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ResolvedConfig {
    /// Which counter-rotation scheduler to instantiate.
    pub scheduler: SchedulerKind,
    /// Which state-estimation algorithm to instantiate.
    pub estimator: EstimatorKind,
    /// Nanoseconds between successive scheduler decisions (counter-rotation quantum).
    pub q_schedule_ns: u64,
    /// Nanoseconds between hardware counter samples within a scheduling quantum.
    pub q_sample_ns: u64,
    /// Nanoseconds between output flushes; `0` disables periodic flushing.
    pub q_output_ns: u64,
    /// Tuning parameters for the Kalman filter estimator.
    pub kalman: KalmanConfig,
    /// Tuning parameters for the EMA estimator.
    pub ema: EmaConfig,
    /// Standard deviation of artificial Gaussian noise added to counter readings (0 = disabled).
    pub noise_stddev: f64,
    /// Optional RNG seed for reproducible runs; `None` uses a random seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Number of hardware counter slots available during simulation.
    pub num_slots: usize,
    /// Settings for LLM-backed schedulers; ignored when no LLM scheduler is selected.
    #[serde(default)]
    pub llm: LlmConfig,
}

impl ResolvedConfig {
    /// Constructs and returns the scheduler specified by `self.scheduler`, wired to the given event registry.
    /// `simulation` should be `true` when replaying a trace (no real-time sleep between quanta).
    pub fn build_scheduler(&self, registry: &EventRegistry, simulation: bool) -> Box<dyn Scheduler> {
        match self.scheduler {
            SchedulerKind::Random => Box::new(RandomScheduler::default()),
            SchedulerKind::RoundRobin => Box::new(RoundRobinScheduler::default()),
            SchedulerKind::MaxUncertainty => Box::new(MaxUncertaintyScheduler::default()),
            SchedulerKind::StaticLlm => {
                let event_info = registry
                    .get_event_ids()
                    .into_iter()
                    .map(|id| {
                        let ev = registry.get_event(id);
                        (id, ev.name.clone(), ev.desc.clone())
                    })
                    .collect();
                let client = LlmClient::new(&self.llm.base_url, &self.llm.model);
                Box::new(StaticLlmScheduler::new(
                    event_info,
                    client,
                    self.llm.guidance.clone(),
                ))
            }
            SchedulerKind::DynamicLlm => {
                let event_info = registry
                    .get_event_ids()
                    .into_iter()
                    .map(|id| {
                        let ev = registry.get_event(id);
                        (id, ev.name.clone(), ev.desc.clone())
                    })
                    .collect();
                let client = LlmClient::new(&self.llm.base_url, &self.llm.model);
                Box::new(DynamicLlmScheduler::new(
                    event_info,
                    client,
                    self.llm.update_interval,
                    self.llm.guidance.clone(),
                    simulation,
                ))
            }
            SchedulerKind::WeightedRoundRobinLlm => {
                let event_info = registry
                    .get_event_ids()
                    .into_iter()
                    .map(|id| {
                        let ev = registry.get_event(id);
                        (id, ev.name.clone(), ev.desc.clone())
                    })
                    .collect();
                let client = LlmClient::new(&self.llm.base_url, &self.llm.model);
                Box::new(WeightedRoundRobinLlmScheduler::new(
                    event_info,
                    client,
                    self.llm.guidance.clone(),
                ))
            }
            SchedulerKind::RateOfChange => Box::new(RateOfChangeScheduler::default()),
        }
    }

    /// Constructs and returns the state estimator specified by `self.estimator`, loading correlation data when applicable.
    pub fn build_estimator(&self, registry: &EventRegistry) -> Box<dyn StateEstimator> {
        match self.estimator {
            EstimatorKind::Propagate => Box::new(PropagateEstimator::new()),
            EstimatorKind::Ema => Box::new(VirtualCounterState::with_config(self.ema.clone())),
            EstimatorKind::Kalman => {
                let mut est = KalmanFilterEstimator::with_config(self.kalman.clone());
                if self.kalman.correlation_path.is_some() {
                    let name_to_id: std::collections::HashMap<String, EventId> = registry
                        .get_event_ids()
                        .into_iter()
                        .map(|id| (registry.get_event_name(id).to_string(), id))
                        .collect();
                    est.load_correlation(&name_to_id);
                }
                Box::new(est)
            }
        }
    }
}

/// CLI-supplied overrides that take precedence over file-based configuration.
///
/// Each field is `None` when the user did not supply the corresponding flag,
/// leaving the file or default value in effect.
pub struct CliOverrides {
    /// Overrides the scheduler algorithm.
    pub scheduler: Option<SchedulerKind>,
    /// Overrides the state estimator algorithm.
    pub estimator: Option<EstimatorKind>,
    /// Overrides the scheduling quantum in nanoseconds.
    pub q_schedule_ns: Option<u64>,
    /// Overrides the sampling quantum in nanoseconds.
    pub q_sample_ns: Option<u64>,
    /// Overrides the output flush quantum in nanoseconds.
    pub q_output_ns: Option<u64>,
    /// Overrides the noise standard deviation.
    pub noise_stddev: Option<f64>,
    /// Overrides the RNG seed.
    pub seed: Option<u64>,
    /// Overrides the number of hardware counter slots.
    pub num_slots: Option<usize>,
    /// Overrides the LLM guidance string.
    pub guidance: Option<String>,
}

/// Hard-coded baseline values serialized as the lowest-priority config layer.
#[derive(serde::Serialize)]
struct Defaults {
    /// Default scheduler variant name (must be a valid `SchedulerKind` snake_case string).
    scheduler: &'static str,
    /// Default estimator variant name (must be a valid `EstimatorKind` snake_case string).
    estimator: &'static str,
    /// Default scheduling quantum in nanoseconds (10 ms).
    q_schedule_ns: u64,
    /// Default sampling quantum in nanoseconds (100 µs).
    q_sample_ns: u64,
    /// Default output flush quantum in nanoseconds (0 = disabled).
    q_output_ns: u64,
    /// Default noise standard deviation (0 = no noise).
    noise_stddev: f64,
    /// Default number of hardware counter slots (4).
    num_slots: usize,
    /// Default Kalman filter configuration.
    kalman: KalmanConfig,
    /// Default EMA configuration.
    ema: EmaConfig,
    /// Default LLM configuration.
    llm: LlmConfig,
}

impl Default for Defaults {
    /// Constructs a `Defaults` with all baseline values.
    fn default() -> Self {
        Self {
            scheduler: "round_robin",
            estimator: "propagate",
            q_schedule_ns: 10_000_000,
            q_sample_ns: 100_000,
            q_output_ns: 0,
            noise_stddev: 0.0,
            num_slots: 4,
            kalman: KalmanConfig::default(),
            ema: EmaConfig::default(),
            llm: LlmConfig::default(),
        }
    }
}

/// Merges defaults, the TOML file at `config_path`, and CLI overrides into a `ResolvedConfig`.
///
/// When `explicit` is `false` the TOML file is optional; when `true` its absence is an error.
fn build_config(
    config_path: PathBuf,
    explicit: bool,
    ov: CliOverrides,
) -> Result<ResolvedConfig, config::ConfigError> {
    let defaults = Config::try_from(&Defaults::default())?;
    Config::builder()
        .add_source(defaults)
        .add_source(
            File::from(config_path)
                .format(FileFormat::Toml)
                .required(explicit),
        )
        .set_override_option("scheduler", ov.scheduler.map(|s| s.to_string()))?
        .set_override_option("estimator", ov.estimator.map(|e| e.to_string()))?
        .set_override_option("q_schedule_ns", ov.q_schedule_ns.map(|v| v as i64))?
        .set_override_option("q_sample_ns", ov.q_sample_ns.map(|v| v as i64))?
        .set_override_option("q_output_ns", ov.q_output_ns.map(|v| v as i64))?
        .set_override_option("noise_stddev", ov.noise_stddev)?
        .set_override_option("seed", ov.seed.map(|v| v as i64))?
        .set_override_option("num_slots", ov.num_slots.map(|v| v as i64))?
        .set_override_option("llm.guidance", ov.guidance)?
        .build()?
        .try_deserialize::<ResolvedConfig>()
}

/// Loads a [`ResolvedConfig`] by merging three layers in increasing priority:
/// hard-coded defaults → TOML file → CLI overrides.
///
/// `path` defaults to `saccade.toml` in the current directory when `None`.
/// When `explicit` is `true` and the file does not exist, an `io::ErrorKind::NotFound`
/// error is returned; otherwise a missing file is silently skipped.
pub fn load_config(
    path: Option<PathBuf>,
    explicit: bool,
    ov: CliOverrides,
) -> io::Result<ResolvedConfig> {
    let config_path = path.unwrap_or_else(|| PathBuf::from("saccade.toml"));
    if explicit && !config_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Config file not found: {}", config_path.display()),
        ));
    }

    let result = build_config(config_path, explicit, ov)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Returns a `CliOverrides` with every field set to `None` (no overrides applied).
    fn no_overrides() -> CliOverrides {
        CliOverrides {
            scheduler: None,
            estimator: None,
            q_schedule_ns: None,
            q_sample_ns: None,
            q_output_ns: None,
            noise_stddev: None,
            seed: None,
            num_slots: None,
            guidance: None,
        }
    }

    /// Writes `content` to a uniquely-named temp file and returns its path.
    fn write_toml(content: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "saccade_test_{}.toml",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
        path
    }

    #[test]
    fn guidance_from_toml() {
        let path = write_toml("[llm]\nguidance = \"from-toml\"\n");
        let cfg = load_config(Some(path.clone()), true, no_overrides()).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(cfg.llm.guidance.as_deref(), Some("from-toml"));
    }

    #[test]
    fn guidance_cli_overrides_toml() {
        let path = write_toml("[llm]\nguidance = \"from-toml\"\n");
        let mut ov = no_overrides();
        ov.guidance = Some("from-cli".to_string());
        let cfg = load_config(Some(path.clone()), true, ov).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(cfg.llm.guidance.as_deref(), Some("from-cli"));
    }

    #[test]
    fn guidance_defaults_to_none() {
        let path = write_toml("");
        let cfg = load_config(Some(path.clone()), true, no_overrides()).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(cfg.llm.guidance, None);
    }
}
