//! Differential tests: pure-JSON documents parsed by both `simd_json5` and
//! `serde_json` (with `preserve_order` + `float_roundtrip`) and compared
//! structurally. serde_json is the oracle for value classification, string
//! unescaping and correctly rounded float parsing.

use std::fmt::Write as _;

use simd_json5::{Str, Value};

fn conv(v: &serde_json::Value) -> Value<'static> {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Value::U64(u)
            } else if let Some(i) = n.as_i64() {
                Value::I64(i)
            } else {
                Value::F64(n.as_f64().expect("serde_json numbers are finite"))
            }
        },
        serde_json::Value::String(s) => Value::Str(Str::Dynamic(s.clone())),
        serde_json::Value::Array(a) => Value::Array(a.iter().map(conv).collect()),
        serde_json::Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (Str::Dynamic(k.clone()), conv(v)))
                .collect(),
        ),
    }
}

fn check(json: &str) {
    let theirs: serde_json::Value =
        serde_json::from_str(json).unwrap_or_else(|e| panic!("serde_json rejected: {e}"));
    let ours = simd_json5::from_str(json).unwrap_or_else(|e| panic!("we rejected: {e}"));
    assert_eq!(ours, conv(&theirs), "document: {json}");
}

#[test]
fn handwritten_documents() {
    check("null");
    check("true");
    check("42");
    check(r#""plain""#);
    check(r#"{"a":1,"b":[1,2,3],"c":{"d":null},"e":"f"}"#);
    // Every JSON escape.
    check(r#""\" \\ \/ \b \f \n \r \t \u0041 \u00e9 \uD83D\uDE00""#);
    // Integer boundaries.
    check("[0, -1, 1, 18446744073709551615, -9223372036854775808]");
    // Past u64 / i64 range: both sides become exactly converted f64.
    check("[18446744073709551616, -9223372036854775809, 9007199254740993]");
}

#[test]
fn tricky_floats() {
    // Each entry is one float; both parsers must produce bit-identical f64
    // values (correctly rounded).
    let floats = [
        "0.1",
        "0.2",
        "1.5",
        "1e0",
        "1e3",
        "1E+3",
        "1e-3",
        "1.7976931348623157e308",
        "8.98846567431158e307",
        "2.2250738585072011e-308",
        "5e-324",
        "2.2250738585072012e-308",
        "1e-320",
        "123456789012345678901234567890.123456789",
        "0.000000000000000000000000000000000000789",
        "4.9406564584124654e-324",
        "9007199254740993.0",
        "9007199254740992.5",
        "1.0000000000000002",
        "1234567890.1234567890123456789",
    ];
    let doc = format!("[{}]", floats.join(","));
    check(&doc);
}

#[test]
fn whitespace_and_surrogates() {
    check("{\t\"a\"\r\n:\r\n1\n,\n\"b\"\t:\t[ 2 , 3 ]\t}");
    check(r#"{"emoji":"😀","pair":"😀"}"#);
    check("[\"中文\", \"\u{e9}\", \"\u{1f600}\"]");
}

/// xorshift64* deterministic generator: the oracle corpus must be stable.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform value below `n` (always small in this generator).
    #[allow(clippy::cast_possible_truncation)]
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A decimal digit 1-9 for the leading position: strict JSON rejects
    /// leading zeros.
    #[allow(clippy::cast_possible_truncation)]
    fn nz_digit(&mut self) -> char {
        char::from(b'1' + (self.next() % 9) as u8)
    }

    /// A decimal digit 0-9.
    #[allow(clippy::cast_possible_truncation)]
    fn digit(&mut self) -> char {
        char::from(b'0' + (self.next() % 10) as u8)
    }
}

fn digits(rng: &mut Rng, max: usize) -> String {
    let len = 1 + rng.below(max);
    let mut s = String::from(rng.nz_digit());
    s.extend((1..len).map(|_| rng.digit()));
    s
}

fn int_text(rng: &mut Rng) -> String {
    // A mix of random digit runs and interesting boundary values.
    match rng.below(8) {
        0 => "0".to_owned(),
        1 => "18446744073709551615".to_owned(),
        2 => "-9223372036854775808".to_owned(),
        3 => "9223372036854775807".to_owned(),
        4 => "18446744073709551616".to_owned(),
        5 => "-9223372036854775809".to_owned(),
        6 => "9007199254740993".to_owned(),
        _ => {
            let s = digits(rng, 20);
            if rng.below(2) == 0 {
                format!("-{s}")
            } else {
                s
            }
        },
    }
}

fn float_text(rng: &mut Rng) -> String {
    // Digit counts and exponent ranges stay inside f64 so the serde_json
    // oracle accepts them (it rejects out-of-range floats).
    let mantissa = digits(rng, 15);
    // Keep the corpus strict JSON: no trailing-dot forms like `5.`.
    let mut s = if rng.below(4) == 0 {
        format!("0.{mantissa}")
    } else {
        format!("{}.{mantissa}", digits(rng, 10))
    };
    if rng.below(2) == 0 {
        let sign = if rng.below(2) == 0 {
            '+'
        } else {
            '-'
        };
        let exp = rng.below(280);
        let _ = write!(s, "e{sign}{exp}");
    } else if rng.below(8) == 0 {
        s = format!("-{s}");
    }
    s
}

fn string_text(rng: &mut Rng) -> String {
    let parts = rng.below(6);
    let mut s = String::from("\"");
    for i in 0..parts {
        match rng.below(8) {
            0 => s.push_str("plain"),
            1 => s.push_str("with space"),
            2 => s.push_str("\\\"quoted\\\""),
            3 => s.push_str("\\\\backslash\\\\"),
            4 => s.push_str("\\n\\t\\r\\b\\f"),
            5 => s.push_str("\\u00e9\\u0041"),
            6 => s.push_str("\\uD83D\\uDE00"),
            _ => s.push_str(if i % 2 == 0 {
                "中文テキスト"
            } else {
                "raw😀"
            }),
        }
    }
    s.push('"');
    s
}

fn key_text(rng: &mut Rng, i: usize) -> String {
    if rng.below(4) == 0 {
        format!("\"key {i}\"")
    } else {
        format!("\"k{i}\"")
    }
}

fn gen_value(rng: &mut Rng, depth: u32, out: &mut String) {
    // Near the depth cap only scalars are generated.
    match rng.below(if depth >= 28 {
        5
    } else {
        8
    }) {
        0 | 4 => out.push_str(&int_text(rng)),
        1 | 5 => out.push_str(&float_text(rng)),
        2 => out.push_str(&string_text(rng)),
        3 => out.push_str("null"),
        6 => {
            out.push('[');
            let n = rng.below(8);
            for i in 0..n {
                if i > 0 {
                    out.push(',');
                }
                gen_value(rng, depth + 1, out);
            }
            out.push(']');
        },
        _ => {
            out.push('{');
            let n = rng.below(6);
            for i in 0..n {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&key_text(rng, i));
                out.push(':');
                gen_value(rng, depth + 1, out);
            }
            out.push('}');
        },
    }
}

fn generated(seed: u64, rounds: usize) -> String {
    let mut rng = Rng(seed);
    let mut out = String::new();
    out.push('[');
    for i in 0..rounds {
        if i > 0 {
            out.push(',');
        }
        gen_value(&mut rng, 0, &mut out);
    }
    out.push(']');
    out
}

fn gen_deep(rng: &mut Rng, depth: u32, out: &mut String) {
    if depth >= 48 || rng.below(6) == 0 {
        gen_value(rng, depth, out);
        return;
    }
    if rng.below(2) == 0 {
        out.push('[');
        let n = 1 + rng.below(4);
        for i in 0..n {
            if i > 0 {
                out.push(',');
            }
            gen_deep(rng, depth + 1, out);
        }
        out.push(']');
    } else {
        out.push('{');
        let n = 1 + rng.below(4);
        for i in 0..n {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&key_text(rng, depth as usize * 100 + i));
            out.push(':');
            gen_deep(rng, depth + 1, out);
        }
        out.push('}');
    }
}

#[test]
fn generated_corpus_small() {
    check(&generated(0xdead_beef, 200));
}

#[test]
fn generated_corpus_deep() {
    let mut rng = Rng(0x1234_5678);
    let mut out = String::new();
    gen_deep(&mut rng, 0, &mut out);
    check(&out);
}

#[test]
fn generated_corpus_large() {
    let doc = generated(0xc0f_fee, 20_000);
    // "Moderately large": assert the corpus actually exercises the parser.
    assert!(doc.len() > 500_000, "corpus too small: {}", doc.len());
    check(&doc);
}

#[test]
fn generated_corpus_wide() {
    let doc = generated(0xfeed, 60_000);
    check(&doc);
}
