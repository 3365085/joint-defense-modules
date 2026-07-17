use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE_ROOT_FILES: [&str; 4] = ["Cargo.toml", "Cargo.lock", "pyproject.toml", "build.rs"];

const SHA256_INITIAL_STATE: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
];

fn collect_rust_sources(directory: &Path, paths: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory)
        .unwrap_or_else(|error| {
            panic!(
                "failed to enumerate native Rust source directory {}: {error}",
                directory.display()
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|error| {
            panic!(
                "failed to read a native Rust source entry below {}: {error}",
                directory.display()
            )
        });

    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "failed to inspect native Rust source entry {}: {error}",
                path.display()
            )
        });
        if file_type.is_dir() {
            collect_rust_sources(&path, paths);
        } else if file_type.is_file() && path.extension().is_some_and(|value| value == "rs") {
            paths.push(path);
        }
    }
}

fn manifest_relative(crate_root: &Path, path: &Path) -> String {
    path.strip_prefix(crate_root)
        .unwrap_or_else(|_| {
            panic!(
                "native source {} is outside crate root {}",
                path.display(),
                crate_root.display()
            )
        })
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn manifest_sort_key(crate_root: &Path, path: &Path) -> String {
    let relative = manifest_relative(crate_root, path);
    if cfg!(windows) {
        // pathlib.WindowsPath ordering case-folds path parts before comparing.
        relative.to_lowercase()
    } else {
        relative
    }
}

fn source_manifest_paths(crate_root: &Path) -> Vec<PathBuf> {
    let mut paths = SOURCE_ROOT_FILES
        .iter()
        .map(|relative| crate_root.join(relative))
        .collect::<Vec<_>>();
    for path in &paths {
        if !path.is_file() {
            panic!(
                "native source manifest is incomplete; missing {}",
                path.display()
            );
        }
    }

    let source_directory = crate_root.join("src");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&source_directory, &mut rust_sources);
    rust_sources.sort_by_key(|path| manifest_sort_key(crate_root, path));
    paths.extend(rust_sources);
    paths
}

fn sha256(input: &[u8]) -> [u8; 32] {
    let bit_length = u64::try_from(input.len())
        .ok()
        .and_then(|length| length.checked_mul(8))
        .expect("native source manifest is too large to hash with SHA-256");
    let mut message = input.to_vec();
    message.push(0x80);
    while (message.len() + 8) % 64 != 0 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = SHA256_INITIAL_STATE;
    for chunk in message.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).take(16).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(SHA256_ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut digest = [0u8; 32];
    for (index, value) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
    }
    digest
}

fn source_manifest_sha256(crate_root: &Path) -> String {
    let paths = source_manifest_paths(crate_root);
    let mut aggregate = Vec::new();
    for path in paths {
        let relative = manifest_relative(crate_root, &path);
        let content = fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "failed to read native source manifest file {}: {error}",
                path.display()
            )
        });
        aggregate.extend_from_slice(relative.as_bytes());
        aggregate.push(0);
        aggregate.extend_from_slice(&content);
        aggregate.push(0);
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let mut encoded = String::with_capacity(64);
    for byte in sha256(&aggregate) {
        write!(&mut encoded, "{byte:02X}").expect("writing SHA-256 hex cannot fail");
    }
    encoded
}

fn main() {
    let crate_root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is required"));
    println!(
        "cargo:rerun-if-changed={}",
        crate_root.join("src").display()
    );
    let manifest_sha256 = source_manifest_sha256(&crate_root);

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());

    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned());
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned());

    println!("cargo:rustc-env=MODULE_A_NATIVE_RUSTC_VERSION={rustc_version}");
    println!("cargo:rustc-env=MODULE_A_NATIVE_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=MODULE_A_NATIVE_BUILD_TARGET={target}");
    println!("cargo:rustc-env=MODULE_A_NATIVE_SOURCE_MANIFEST_SHA256={manifest_sha256}");
}
