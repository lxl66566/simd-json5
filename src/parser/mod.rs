//! Two-stage parser: SIMD structural indexing followed by semantic parsing.

pub(crate) mod number;
pub(crate) mod simd;
pub(crate) mod stage1;
pub(crate) mod stage2;

use crate::{Result, value::Value};

pub(crate) fn parse(bytes: &[u8]) -> Result<Value<'_>> {
    // One index entry per ~8 input bytes covers typical documents up front
    // (twitter needs ~1/11, canada ~1/6); the early reallocation chain of
    // the empty vec is what this avoids, dense inputs still grow.
    let mut index = Vec::with_capacity(bytes.len() / 8);
    simd::build_index(bytes, &mut index)?;
    stage2::build(bytes, &index)
}
