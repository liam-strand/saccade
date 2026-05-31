//! Latency distribution loader for injecting pre-measured LLM call delays into simulation.

use rand::SeedableRng;
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::io;
use std::path::Path;

#[derive(serde::Deserialize)]
struct ProfileEntry {
    samples: Vec<u64>,
}

/// Holds per-call-type latency samples drawn from a q7_llm_latency.py profile JSON.
pub struct LlmLatencyProfile {
    entries: HashMap<String, Vec<u64>>,
    rng: StdRng,
}

impl LlmLatencyProfile {
    /// Load a profile JSON produced by `q7_llm_latency.py`.
    ///
    /// The file format is `{call_type: {samples: [u64, ...], median_ms: f, p95_ms: f}}`.
    /// Seeds RNG with `seed + 1` to keep it distinct from the simulation's VirtualSampleSource RNG.
    pub fn load(path: &Path, seed: Option<u64>) -> io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let raw: HashMap<String, ProfileEntry> = serde_json::from_str(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let entries = raw.into_iter().map(|(k, v)| (k, v.samples)).collect();
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s.wrapping_add(1)),
            None => StdRng::from_os_rng(),
        };
        Ok(Self { entries, rng })
    }

    /// Draw a uniform-random sample (ms) for the given call type.
    ///
    /// Returns `None` if the call type is absent from the profile or its sample list is empty.
    pub fn sample(&mut self, call_type: &str) -> Option<u64> {
        self.entries.get(call_type)?.choose(&mut self.rng).copied()
    }
}
