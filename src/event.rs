/// Merged event module: EventLibrary parser + EventRegistry lookup.
///
/// The old `event_library` and `event_registry` modules are preserved for backwards
/// compatibility during the rewrite transition and will be removed in the cleanup step.
pub use crate::event_library::{Event, EventLibrary};
pub use crate::event_registry::{EventId, EventRegistry};
