//! Build script for the `runqslower` example.

use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;

use libbpf_cargo::SkeletonBuilder;

const SRC: &str = "src/bpf/sampler.bpf.c";
const HEADER: &str = "src/bpf/sampler.h";

#[derive(Debug)]
struct RenameCallbacks;

impl bindgen::callbacks::ParseCallbacks for RenameCallbacks {
    fn enum_variant_name(
        &self,
        _enum_name: Option<&str>,
        original: &str,
        _value: bindgen::callbacks::EnumVariantValue,
    ) -> Option<String> {
        let stripped = original.strip_prefix("SAMPLE_TYPE_")?;
        Some(stripped[..1].to_uppercase() + &stripped[1..].to_lowercase())
    }

    fn int_macro(&self, name: &str, _value: i64) -> Option<bindgen::callbacks::IntKind> {
        match name {
            "MAX_COUNTERS" | "MAX_CPUS" | "TASK_COMM_LEN" => {
                Some(bindgen::callbacks::IntKind::Custom {
                    name: "usize",
                    is_signed: false,
                })
            }
            _ => None,
        }
    }
}

fn main() {
    let out = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set in build script"),
    )
    .join("src")
    .join("bpf")
    .join("sampler.skel.rs");

    let arch = env::var("CARGO_CFG_TARGET_ARCH")
        .expect("CARGO_CFG_TARGET_ARCH must be set in build script");

    SkeletonBuilder::new()
        .source(SRC)
        .clang_args([
            OsStr::new("-I"),
            vmlinux::include_path_root().join(arch).as_os_str(),
        ])
        .build_and_generate(&out)
        .unwrap();
    println!("cargo:rerun-if-changed={SRC}");
    println!("cargo:rerun-if-changed={HEADER}");

    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set in build script"));

    bindgen::Builder::default()
        .header(HEADER)
        .allowlist_type("saccade_sample|SampleType")
        .allowlist_var("MAX_COUNTERS|MAX_CPUS|TASK_COMM_LEN")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .parse_callbacks(Box::new(RenameCallbacks))
        .derive_default(true)
        .derive_copy(true)
        .generate()
        .expect("bindgen failed to generate wire_types.rs")
        .write_to_file(out_dir.join("wire_types.rs"))
        .expect("failed to write wire_types.rs");
}
