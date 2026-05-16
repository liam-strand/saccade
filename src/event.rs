//! Hardware performance counter event definitions, `perf list` output parsing, and name-to-id lookup.

use nom::{
    IResult, Parser,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{alpha1, hex_digit1, multispace0, multispace1},
    combinator::{map, map_res, opt},
    multi::separated_list1,
    sequence::{delimited, preceded, separated_pair},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, str};

/// Stable numeric identifier for an event within an [`EventRegistry`]; corresponds to its index in the backing `Vec`.
pub type EventId = u32;

/// Runtime index over a loaded [`EventLibrary`], providing O(1) lookup from event name to [`EventId`].
pub struct EventRegistry {
    /// Ordered list of all events; an event's position is its [`EventId`].
    events: Vec<Event>,
    /// Maps each event's canonical name to its index in `events` for fast lookup.
    event_names: HashMap<String, usize>,
}

impl EventRegistry {
    /// Builds the registry from an [`EventLibrary`], constructing the name-to-index map.
    pub fn new(events: EventLibrary) -> Self {
        let mut event_names = HashMap::new();
        for (i, event) in events.events.iter().enumerate() {
            event_names.insert(event.name.clone(), i);
        }

        Self {
            events: events.events,
            event_names,
        }
    }

    /// Returns the [`EventId`] for the given event name, or `None` if the name is not in the library.
    pub fn lookup(&self, name: &str) -> Option<EventId> {
        self.event_names.get(name).map(|&e| e as u32)
    }

    /// Returns the [`Event`] corresponding to `id`; panics if `id` is out of range.
    pub fn get_event(&self, id: EventId) -> &Event {
        &self.events[id as usize]
    }

    /// Returns all valid [`EventId`]s in insertion order.
    pub fn get_event_ids(&self) -> Vec<EventId> {
        (0..self.events.len() as u32).collect()
    }

    /// Returns the canonical name of the event with the given `id`; panics if `id` is out of range.
    pub fn get_event_name(&self, id: EventId) -> &str {
        &self.events[id as usize].name
    }

    /// Returns a cloned snapshot of all events, suitable for serialization.
    pub fn dump(&self) -> Vec<Event> {
        self.events.clone()
    }
}

/// Serializable collection of hardware performance counter events parsed from `perf list --details`.
#[derive(Clone, Serialize, Deserialize)]
pub struct EventLibrary {
    /// All parsed events; order is preserved from the `perf list` output.
    pub events: Vec<Event>,
}

impl Default for EventLibrary {
    /// Returns an empty library; prefer [`EventLibrary::from_bytes`] to populate from `perf list` output.
    fn default() -> Self {
        Self::new()
    }
}

impl EventLibrary {
    /// Creates an empty library with no events.
    pub fn new() -> EventLibrary {
        EventLibrary { events: Vec::new() }
    }

    /// Parses raw `perf list --details` stdout bytes into an `EventLibrary`, silently skipping unrecognized lines.
    pub fn from_bytes(input: &[u8]) -> Result<Self, String> {
        let mut events = Vec::new();
        let mut i = input;

        while !i.is_empty() {
            // Try to parse a Counter
            match Event::parse_nom(i) {
                Ok((rem, counter)) => {
                    events.push(counter);
                    i = rem;
                }
                Err(_) => {
                    // Start of line didn't match a Counter.
                    // Consume until newline to skip this line
                    match take_until::<_, _, nom::error::Error<&[u8]>>("\n")(i) {
                        Ok((rem, _)) => {
                            // Skip the newline itself
                            if !rem.is_empty() {
                                i = &rem[1..];
                            } else {
                                i = rem;
                            }
                        }
                        Err(_) => {
                            // No newline found, consume all
                            i = &[];
                        }
                    }
                }
            }
        }
        Ok(EventLibrary { events })
    }
}

/// A single hardware performance counter event, as enumerated by `perf list --details`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Canonical event name used as the lookup key (e.g., `"fp_ret_sse_avx_ops.all"`).
    pub name: String,
    /// Human-readable description from the `perf list` output brackets.
    pub desc: String,
    /// Event select value written to the hardware performance monitoring MSR.
    pub event: u64,
    /// Unit mask value written alongside `event` to select a counter sub-type.
    pub umask: u64,
}

impl Event {
    /// Parses a single event record from `perf list` bytes, returning an error string on failure.
    pub fn parse(i: &[u8]) -> Result<Self, String> {
        match Self::parse_nom(i) {
            Ok((_, counter)) => Ok(counter),
            Err(e) => Err(format!("Parse error: {:?}", e)),
        }
    }

    /// nom combinator that matches one `perf list` event stanza (name, bracketed description, `cpu/…/` config line).
    pub fn parse_nom(i: &[u8]) -> IResult<&[u8], Event> {
        let (i, _) = multispace0(i)?;
        let (i, name) = parse_name(i)?;
        let (i, _) = multispace1(i)?;
        let (i, desc) = map(delimited(tag("["), take_until("]\n"), tag("]")), |s| {
            str::from_utf8(s)
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .parse(i)?;
        let (i, _) = multispace1(i)?;
        let (i, (event, umask)) = parse_config(i)?;
        let (i, _) = multispace0(i)?;

        Ok((
            i,
            Event {
                name,
                desc,
                event,
                umask,
            },
        ))
    }
}

/// Parses an optional `0x`-prefixed hexadecimal integer.
fn parse_hex(i: &[u8]) -> IResult<&[u8], u64> {
    map_res(preceded(opt(tag("0x")), hex_digit1), |out: &[u8]| {
        u64::from_str_radix(str::from_utf8(out).unwrap(), 16)
    })
    .parse(i)
}

/// Parses one `key=hexvalue` pair from a `cpu/…/` config string.
fn parse_key_value(i: &[u8]) -> IResult<&[u8], (&[u8], u64)> {
    separated_pair(alpha1, tag("="), parse_hex).parse(i)
}

/// Parses the `cpu/event=…,umask=…/` config line and returns `(event, umask)`, ignoring unknown keys.
fn parse_config(i: &[u8]) -> IResult<&[u8], (u64, u64)> {
    let (i, _) = tag("cpu/")(i)?;
    let (i, kvs) = separated_list1(tag(","), parse_key_value).parse(i)?;
    let (i, _) = tag("/")(i)?;

    let mut event = 0;
    let mut umask = 0;
    for (k, v) in kvs {
        match k {
            b"event" => event = v,
            b"umask" => umask = v,
            _ => {}
        }
    }
    Ok((i, (event, umask)))
}

/// Returns `true` for bytes that may appear in a `perf` event name: ASCII alphanumeric, `_`, `.`, or `-`.
fn is_name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'-'
}

/// Parses a non-empty event name token using [`is_name_char`] and decodes it as UTF-8.
fn parse_name(i: &[u8]) -> IResult<&[u8], String> {
    map(take_while1(is_name_char), |s| {
        str::from_utf8(s).unwrap().to_owned()
    })
    .parse(i)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bp_l1_btb_correct() {
        let text = br#"
  bp_l1_btb_correct                                 
       [L1 Branch Prediction Overrides Existing Prediction (speculative)]
        cpu/event=0x8a/
"#;

        let res = Event::parse(text).unwrap();

        assert_eq!(
            res,
            Event {
                name: "bp_l1_btb_correct".to_owned(),
                desc: "L1 Branch Prediction Overrides Existing Prediction (speculative)".to_owned(),
                event: 0x8a,
                umask: 0,
            }
        );
    }
    #[test]
    fn bp_l1_tlb_fetch_hit() {
        let text = br#"
  bp_l1_tlb_fetch_hit                               
       [The number of instruction fetches that hit in the L1 ITLB]
        cpu/umask=0xff,event=0x94/ 
"#;

        let res = Event::parse(text).unwrap();

        assert_eq!(
            res,
            Event {
                name: "bp_l1_tlb_fetch_hit".to_owned(),
                desc: "The number of instruction fetches that hit in the L1 ITLB".to_owned(),
                event: 0x94,
                umask: 0xff,
            }
        );
    }
    #[test]
    fn fp_ret_sse_avx_ops_all() {
        let text = br#"
  fp_ret_sse_avx_ops.all
       [All FLOPS. This is a retire-based event. The number of retired SSE/AVX
        FLOPS. The number of events logged per cycle can vary from 0 to 64.
        This event can count above 15]
        cpu/umask=0xff,event=0x3/
"#;

        let res = Event::parse(text).unwrap();

        assert_eq!(
            res,
            Event {
                name: "fp_ret_sse_avx_ops.all".to_owned(),
                desc: "All FLOPS. This is a retire-based event. The number of retired SSE/AVX FLOPS. The number of events logged per cycle can vary from 0 to 64. This event can count above 15"
                    .to_owned(),
                event: 0x3,
                umask: 0xff,
            }
        );
    }

    #[test]
    fn ex_ret_mmx_fp_instr_sse_instr() {
        let text = br#"
  ex_ret_mmx_fp_instr.sse_instr
       [SSE instructions (SSE, SSE2, SSE3, SSSE3, SSE4A, SSE41, SSE42, AVX)]
        cpu/umask=0x4,event=0xcb/
"#;

        let res = Event::parse(text).unwrap();

        assert_eq!(
            res,
            Event {
                name: "ex_ret_mmx_fp_instr.sse_instr".to_owned(),
                desc: "SSE instructions (SSE, SSE2, SSE3, SSSE3, SSE4A, SSE41, SSE42, AVX)"
                    .to_owned(),
                event: 0xcb,
                umask: 0x4,
            }
        );
    }

    #[test]
    fn ex_tagged_ibs_ops_ibs_count_rollover() {
        let text = br#"
  ex_tagged_ibs_ops.ibs_count_rollover       
       [Tagged IBS Ops. Number of times an op could not be tagged by IBS
        because of a previous tagged op that has not retired]
        cpu/umask=0x4,event=0x1cf/"#;
        let res = Event::parse(text).unwrap();
        assert_eq!(res.name, "ex_tagged_ibs_ops.ibs_count_rollover");
    }
}
