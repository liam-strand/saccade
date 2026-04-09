use super::trace::read_trace_packets;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Per-event time-series of rates, keyed by event name.
/// Each entry is a sorted Vec of (timestamp_ns, rate_events_per_ns).
pub struct RateTimeSeries {
    pub series: HashMap<String, Vec<(u64, f64)>>,
}

/// Parse a `.perfetto-trace` written by `PerfettoWriter` and extract
/// the rate time-series for each event.
///
/// Track names are expected to follow the `{event_name}/rate` convention.
/// `{event_name}/uncertainty` tracks are ignored.
pub fn read_rate_timeseries(path: impl AsRef<Path>) -> io::Result<RateTimeSeries> {
    let data = std::fs::read(path)?;
    let packets = read_trace_packets(&data)?;

    // Pass 1: build uuid -> event_name from TrackDescriptor packets.
    let mut uuid_to_name: HashMap<u64, String> = HashMap::new();
    for packet in &packets {
        if packet.has_track_descriptor() {
            let desc = packet.track_descriptor();
            let name = desc.name().to_string();
            if let Some(event_name) = name.strip_suffix("/rate") {
                uuid_to_name.insert(desc.uuid(), event_name.to_string());
            }
        }
    }

    // Pass 2: collect counter values from TrackEvent packets.
    let mut series: HashMap<String, Vec<(u64, f64)>> = HashMap::new();
    for packet in &packets {
        if !packet.has_track_event() {
            continue;
        }
        let event = packet.track_event();
        let Some(event_name) = uuid_to_name.get(&event.track_uuid()) else {
            continue;
        };
        let timestamp = packet.timestamp();
        let rate = event.double_counter_value();
        series
            .entry(event_name.clone())
            .or_default()
            .push((timestamp, rate));
    }

    // Ensure each series is sorted by timestamp.
    for pts in series.values_mut() {
        pts.sort_unstable_by_key(|&(ts, _)| ts);
    }

    Ok(RateTimeSeries { series })
}
