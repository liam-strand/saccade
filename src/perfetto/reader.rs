//! Reading Perfetto proto binary trace files produced by `PerfettoWriter`.

use super::trace::for_each_packet;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

/// Per-(event, thread) time-series of rates decoded from a Perfetto trace.
pub struct RateTimeSeries {
    /// Maps `(event_name, tid)` to a timestamp-sorted list of `(timestamp_ns, rate_events_per_ns)` samples.
    pub series: HashMap<(String, u32), Vec<(u64, f64)>>,
}

/// Parse a `.perfetto-trace` written by `PerfettoWriter` and extract per-thread rate time-series.
///
/// Counter tracks named `{event_name}/rate` are identified; their `parent_uuid` is followed to a
/// thread track bearing a `ThreadDescriptor` to resolve the `tid`.  If no thread parent is found,
/// `tid` defaults to `0` as a defensive fallback.  Tracks with other name suffixes are ignored.
pub fn read_rate_timeseries(path: impl AsRef<Path>) -> io::Result<RateTimeSeries> {
    // SAFETY: standard mmap precaution — the file must not be truncated while
    // the mapping is live. Simulation inputs are read-only and stable.
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Pass 1 (streaming): scan TrackDescriptors only — one packet live at a time.
    //   - uuid_to_tid: uuid → tid, for tracks that have a ThreadDescriptor
    //   - uuid_to_rate_key: uuid → (event_name, parent_uuid) for rate counter tracks
    let mut uuid_to_tid: HashMap<u64, u32> = HashMap::new();
    let mut uuid_to_rate_key: HashMap<u64, (String, Option<u64>)> = HashMap::new();

    for_each_packet(&mmap, |packet| {
        if !packet.has_track_descriptor() {
            return;
        }
        let desc = packet.track_descriptor();
        let uuid = desc.uuid();
        let name = desc.name().to_string();

        if let Some(thread) = desc.thread.as_ref() {
            uuid_to_tid.insert(uuid, thread.tid() as u32);
        }

        if let Some(event_name) = name.strip_suffix("/rate") {
            let parent_uuid = desc.has_parent_uuid().then(|| desc.parent_uuid());
            uuid_to_rate_key.insert(uuid, (event_name.to_string(), parent_uuid));
        }
    })?;

    // Resolve: for each rate counter track, map uuid → (event_name, tid).
    let mut uuid_to_event_tid: HashMap<u64, (String, u32)> = HashMap::new();
    for (uuid, (event_name, parent_uuid)) in &uuid_to_rate_key {
        let tid = parent_uuid
            .and_then(|p| uuid_to_tid.get(&p))
            .copied()
            .unwrap_or(0);
        uuid_to_event_tid.insert(*uuid, (event_name.clone(), tid));
    }
    drop(uuid_to_tid);
    drop(uuid_to_rate_key);

    // Pass 2 (streaming): collect counter values from TrackEvent packets.
    let mut series: HashMap<(String, u32), Vec<(u64, f64)>> = HashMap::new();
    for_each_packet(&mmap, |packet| {
        if !packet.has_track_event() {
            return;
        }
        let event = packet.track_event();
        let Some((event_name, tid)) = uuid_to_event_tid.get(&event.track_uuid()) else {
            return;
        };
        series
            .entry((event_name.clone(), *tid))
            .or_default()
            .push((packet.timestamp(), event.double_counter_value()));
    })?;

    // Ensure each series is sorted by timestamp.
    for pts in series.values_mut() {
        pts.sort_unstable_by_key(|&(ts, _)| ts);
    }

    Ok(RateTimeSeries { series })
}
