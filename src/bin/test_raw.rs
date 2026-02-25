use perf_event::events;

fn main() {
    let r = events::Raw::new(0x94).config1(0xff);
    println!("Raw(0x94).config(0xff) -> {:#x}", r.config);
}
