use crate::counter::Counter;
use nom::{IResult, bytes::complete::take_until};

pub struct EventLibrary {
    pub events: Vec<Counter>,
}

impl EventLibrary {
    pub fn new() -> EventLibrary {
        EventLibrary { events: Vec::new() }
    }

    pub fn from_bytes(input: &[u8]) -> Result<Self, String> {
        let mut events = Vec::new();
        let mut i = input;

        while !i.is_empty() {
            // Try to parse a Counter
            match Counter::parse_nom(i) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_perf_out() {
        let perf_out = include_bytes!("../perf.out");
        let lib = EventLibrary::from_bytes(perf_out).unwrap();

        // Check for a specific known event
        let bp_l1_btb_correct = lib.events.iter().find(|e| e.name == "bp_l1_btb_correct");
        assert!(bp_l1_btb_correct.is_some());
        let event = bp_l1_btb_correct.unwrap();
        assert_eq!(event.event, 0x8a);

        // Verify count is correct
        assert_eq!(lib.events.len(), 223);
    }
}
