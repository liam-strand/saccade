//! Perfetto trace file I/O: writing per-thread hardware counter timeseries and reading them back.

pub mod reader;
pub mod trace;

pub use reader::{RateTimeSeries, read_rate_timeseries};
pub use trace::PerfettoWriter;
