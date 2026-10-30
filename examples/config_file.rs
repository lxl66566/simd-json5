//! Read, edit and write a JSON5 config file with the DOM API.
//!
//! `Value` borrows string data from the input text, so parsing a config
//! allocates only for strings that needed unescaping. Serialization emits
//! strict JSON: comments, unquoted keys and hex numbers do not survive a
//! round trip.
//!
//! Usage: `cargo run --example config_file -- [config.json5]`
//! (without a path, edits the embedded sample and prints the result)

use std::{env, fs};

const SAMPLE: &str = r#"{
  // HTTP server settings
  server: {
    host: "0.0.0.0",
    port: 8080,
    timeout_ms: 30000,
    retries: 0x3, // hex literals are JSON5
  },
  features: ["cache", "gzip",],
  limits: { ratio: 0.75, },
  fallback: Infinity,
}"#;

fn main() {
    let (path, text) = match env::args().nth(1) {
        Some(path) => {
            let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            (Some(path), text)
        },
        None => (None, SAMPLE.to_string()),
    };

    let mut config = simd_json5::from_str(&text).expect("config must be valid JSON5");

    // Indexing yields Null for missing members instead of panicking, so a
    // partial config falls back to defaults naturally.
    let host = config["server"]["host"].as_str().unwrap_or("127.0.0.1");
    let port = config["server"]["port"].as_u64().unwrap_or(80);
    println!("serving on {host}:{port}");

    let root = config.as_object_mut().expect("config root is an object");
    if let Some(retries) = root
        .get_mut("server")
        .and_then(simd_json5::Value::as_object_mut)
        .and_then(|server| server.get_mut("retries"))
    {
        *retries = 5u64.into();
    }
    // as_array_mut lends a slice, which cannot grow; match the variant to
    // append.
    if let Some(simd_json5::Value::Array(features)) = root.get_mut("features") {
        features.push("tls".into());
    }

    let out = simd_json5::to_string_pretty(&config);
    match path {
        Some(path) => fs::write(path, out).expect("write config"),
        None => println!("{out}"),
    }
}
