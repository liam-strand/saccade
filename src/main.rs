use saccade::event_library::EventLibrary;

fn main() {
    let lib = EventLibrary::from_bytes(include_bytes!("../perf.out")).unwrap();
    println!("{:#?}", lib.events);
}
