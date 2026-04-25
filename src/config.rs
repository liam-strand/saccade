use crate::scheduler::Scheduler;
use crate::scheduler::distribution::DistributionScheduler;
use crate::scheduler::random::RandomScheduler;
use crate::scheduler::round_robin::RoundRobinScheduler;
use crate::state::StateEstimator;
use crate::state::ema::{EmaConfig, VirtualCounterState};
use crate::state::kalman::{KalmanConfig, KalmanFilterEstimator};
use crate::state::propagate::PropagateEstimator;
use config::{Config, File, FileFormat};
use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerKind {
    Random,
    RoundRobin,
    Distribution,
}

impl fmt::Display for SchedulerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulerKind::Random => write!(f, "random"),
            SchedulerKind::RoundRobin => write!(f, "round_robin"),
            SchedulerKind::Distribution => write!(f, "distribution"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum EstimatorKind {
    Propagate,
    Ema,
    Kalman,
}

impl fmt::Display for EstimatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EstimatorKind::Propagate => write!(f, "propagate"),
            EstimatorKind::Ema => write!(f, "ema"),
            EstimatorKind::Kalman => write!(f, "kalman"),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ResolvedConfig {
    pub scheduler: SchedulerKind,
    pub estimator: EstimatorKind,
    pub q_schedule_ns: u64,
    pub q_sample_ns: u64,
    pub q_output_ns: u64,
    pub kalman: KalmanConfig,
    pub ema: EmaConfig,
    pub noise_stddev: f64,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl ResolvedConfig {
    pub fn build_scheduler(&self) -> Box<dyn Scheduler> {
        match self.scheduler {
            SchedulerKind::Random => Box::new(RandomScheduler::default()),
            SchedulerKind::RoundRobin => Box::new(RoundRobinScheduler::default()),
            SchedulerKind::Distribution => Box::new(DistributionScheduler::default()),
        }
    }

    pub fn build_estimator(&self) -> Box<dyn StateEstimator> {
        match self.estimator {
            EstimatorKind::Propagate => Box::new(PropagateEstimator::new()),
            EstimatorKind::Ema => Box::new(VirtualCounterState::with_config(self.ema.clone())),
            EstimatorKind::Kalman => {
                Box::new(KalmanFilterEstimator::with_config(self.kalman.clone()))
            }
        }
    }
}

pub struct CliOverrides {
    pub scheduler: Option<SchedulerKind>,
    pub estimator: Option<EstimatorKind>,
    pub q_schedule_ns: Option<u64>,
    pub q_sample_ns: Option<u64>,
    pub q_output_ns: Option<u64>,
    pub noise_stddev: Option<f64>,
    pub seed: Option<u64>,
}

/// Single source of truth for default values, fed into the config builder via Serialize.
#[derive(serde::Serialize)]
struct Defaults {
    scheduler: &'static str,
    estimator: &'static str,
    q_schedule_ns: u64,
    q_sample_ns: u64,
    q_output_ns: u64,
    noise_stddev: f64,
    kalman: KalmanConfig,
    ema: EmaConfig,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            scheduler: "round_robin",
            estimator: "propagate",
            q_schedule_ns: 10_000_000,
            q_sample_ns: 100_000,
            q_output_ns: 0,
            noise_stddev: 0.0,
            kalman: KalmanConfig::default(),
            ema: EmaConfig::default(),
        }
    }
}

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
        .build()?
        .try_deserialize::<ResolvedConfig>()
}

/// Load a `ResolvedConfig` by merging three layers:
/// - P1: hard-coded defaults (from `Defaults`)
/// - P2: TOML config file (optional unless `explicit = true`)
/// - P3: CLI overrides (`ov` fields; `None` = no-op)
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
