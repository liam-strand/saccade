use super::trace::read_trace_packets;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Per-(event, thread) time-series of rates.
/// Key: (event_name, tid). For traces with no thread info, tid = 0.
/// Each entry is a sorted Vec of (timestamp_ns, rate_events_per_ns).
pub struct RateTimeSeries {
    pub series: HashMap<(String, u32), Vec<(u64, f64)>>,
}

/// Parse a `.perfetto-trace` written by `PerfettoWriter` and extract
/// per-thread rate time-series for each event.
///
/// Handles two track name formats:
/// - New: counter track named `{event_name}/rate` with `parent_uuid` pointing to a thread track
///   (thread track has a `ThreadDescriptor` with a `tid` field) → key = (event_name, tid)
/// - Old: counter track named `{event_name}/rate` with no thread parent → key = (event_name, 0)
///
/// `{event_name}/uncertainty` tracks are ignored.
pub fn read_rate_timeseries(path: impl AsRef<Path>) -> io::Result<RateTimeSeries> {
    let data = std::fs::read(path)?;
    let packets = read_trace_packets(&data)?;

    // Pass 1: scan TrackDescriptors.
    //   - uuid_to_tid: uuid → tid, for tracks that have a ThreadDescriptor
    //   - uuid_to_rate_key: uuid → (event_name, parent_uuid) for rate counter tracks
    let mut uuid_to_tid: HashMap<u64, u32> = HashMap::new();
    let mut uuid_to_rate_key: HashMap<u64, (String, Option<u64>)> = HashMap::new();

    for packet in &packets {
        if !packet.has_track_descriptor() {
            continue;
        }
        let desc = packet.track_descriptor();
        let uuid = desc.uuid();
        let name = desc.name().to_string();

        // Detect thread tracks via ThreadDescriptor.
        if let Some(thread) = desc.thread.as_ref() {
            let tid = thread.tid() as u32;
            uuid_to_tid.insert(uuid, tid);
        }

        // Detect rate counter tracks by name suffix.
        if let Some(event_name) = name.strip_suffix("/rate") {
            let parent_uuid = if desc.has_parent_uuid() {
                Some(desc.parent_uuid())
            } else {
                None
            };
            uuid_to_rate_key.insert(uuid, (event_name.to_string(), parent_uuid));
        }
    }

    // Pass 2: for each rate counter track, resolve its tid via parent_uuid.
    //   If the parent is a thread track, use its tid; otherwise tid = 0.
    let mut uuid_to_event_tid: HashMap<u64, (String, u32)> = HashMap::new();
    for (uuid, (event_name, parent_uuid)) in &uuid_to_rate_key {
        let tid = parent_uuid
            .and_then(|p| uuid_to_tid.get(&p))
            .copied()
            .unwrap_or(0);
        uuid_to_event_tid.insert(*uuid, (event_name.clone(), tid));
    }

    // Pass 3: collect counter values from TrackEvent packets.
    let mut series: HashMap<(String, u32), Vec<(u64, f64)>> = HashMap::new();
    for packet in &packets {
        if !packet.has_track_event() {
            continue;
        }
        let event = packet.track_event();
        let Some((event_name, tid)) = uuid_to_event_tid.get(&event.track_uuid()) else {
            continue;
        };
        let timestamp = packet.timestamp();
        let rate = event.double_counter_value();
        series
            .entry((event_name.clone(), *tid))
            .or_default()
            .push((timestamp, rate));
    }

    // Ensure each series is sorted by timestamp.
    for pts in series.values_mut() {
        pts.sort_unstable_by_key(|&(ts, _)| ts);
    }

    Ok(RateTimeSeries { series })
}
