pub mod reader;
pub mod trace;

pub use reader::{RateTimeSeries, read_rate_timeseries};
pub use trace::PerfettoWriter;
