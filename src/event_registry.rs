use crate::event_library::{Event, EventLibrary};
use perf_event::{Builder, Counter, events};
use std::collections::HashMap;

pub type EventId = u32;

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
        self.counters[event_id as usize].1.enable().unwrap();
    }

    pub fn disable(&mut self, event_id: EventId) {
        self.counters[event_id as usize].1.disable().unwrap();
    }

    pub fn lookup(&self, name: &str) -> Option<EventId> {
        self.event_names.get(name).map(|&e| e as u32)
    }

    pub fn get_event(&self, id: EventId) -> &Event {
        &self.counters[id as usize].0
    }

    pub fn get_event_ids(&self) -> Vec<EventId> {
        (0..self.counters.len() as u32).collect()
    }
}
