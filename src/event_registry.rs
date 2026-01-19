use crate::event_library::{Event, EventLibrary};
use perf_event::{Builder, Counter, events};
use std::collections::HashMap;

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
}
