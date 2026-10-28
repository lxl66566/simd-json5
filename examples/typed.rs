//! Parse JSON5 into serde structs and write them back.
//!
//! The typed API validates the whole document against the struct in one
//! pass; `#[serde(default)]` fills members the document omits and `Option`
//! keeps "absent" distinguishable from a default value.
//!
//! Usage: `cargo run --example typed`

use simd_json5::{de, typed};

const SAMPLE: &str = r#"{
  // log-level, database and workers omitted: defaults apply
  name: "production",
}"#;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct Config {
    #[serde(default)]
    name: String,
    #[serde(default = "default_workers")]
    workers: usize,
    #[serde(rename = "log-level", default)]
    log_level: LogLevel,
    #[serde(default)]
    database: Option<Database>,
}

fn default_workers() -> usize {
    4
}

#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum LogLevel {
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct Database {
    url: String,
    pool: u32,
}

fn main() {
    let mut config: Config = de::from_str(SAMPLE).expect("valid JSON5");
    println!("{config:#?}");

    config.workers = 16;
    config.log_level = LogLevel::Warn;
    config.database = Some(Database {
        url: "postgres://localhost/app".into(),
        pool: 8,
    });

    let json = typed::to_string_pretty(&config).expect("config serializes");
    println!("{json}");

    // Round trip: the emitted JSON parses back into the same value.
    let reparsed: Config = de::from_str(&json).unwrap();
    assert_eq!(
        typed::to_string(&reparsed).unwrap(),
        typed::to_string(&config).unwrap()
    );
}
