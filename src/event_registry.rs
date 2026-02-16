use crate::event_library::{Event, EventLibrary};

use std::collections::HashMap;

pub type EventId = u32;

pub struct EventRegistry {
    events: Vec<Event>,
    event_names: HashMap<String, usize>,
}

impl EventRegistry {
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

    pub fn lookup(&self, name: &str) -> Option<EventId> {
        self.event_names.get(name).map(|&e| e as u32)
    }

    pub fn get_event(&self, id: EventId) -> &Event {
        &self.events[id as usize]
    }

    pub fn get_event_ids(&self) -> Vec<EventId> {
        (0..self.events.len() as u32).collect()
    }
}
