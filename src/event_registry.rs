use crate::event_library::{Event, EventLibrary};
use perf_event::{Builder, Counter, events};
use std::collections::HashMap;

/// ID representing an event in the EventRegistry
pub type EventId = usize;

pub struct EventRegistry {
    counters: Vec<(Event, Counter)>,
    event_names: HashMap<String, usize>,
}

impl EventRegistry {
    pub fn new(events: EventLibrary) -> Self {
        let mut event_names = HashMap::new();
        for (i, event) in events.events.iter().enumerate() {
            event_names.insert(event.name.clone(), i);
        }

        let counters = events
            .events
            .into_iter()
            .map(|event| {
                let counter = Builder::new(events::Raw::new(event.event).config1(event.umask))
                    .enabled(false)
                    .build()
                    .unwrap();
                (event, counter)
            })
            .collect();
        Self {
            counters,
            event_names,
        }
    }

    pub fn enable(&mut self, event_id: EventId) {
        self.counters[event_id].1.enable().unwrap();
    }

    pub fn disable(&mut self, event_id: EventId) {
        self.counters[event_id].1.disable().unwrap();
    }

    pub fn lookup(&self, name: &str) -> Option<EventId> {
        self.event_names.get(name).cloned()
    }
}
