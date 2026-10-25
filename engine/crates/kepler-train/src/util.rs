//! Small helpers: UTC timestamps and a tiny xorshift RNG.

use std::time::{SystemTime, UNIX_EPOCH};

/// Convert days since 1970-01-01 to (year, month, day) (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn utc_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        s / 3600,
        (s / 60) % 60,
        s % 60
    )
}

pub fn utc_date() -> String {
    utc_iso()[..10].to_string()
}

/// xorshift64* generator, seeded through splitmix64 so small seeds are fine.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        XorShift(if z == 0 { 0x1234_5678_9ABC_DEF1 } else { z })
    }

    /// Raw state, for checkpoints.
    pub fn state(&self) -> u64 {
        self.0
    }

    /// Rebuild from a raw state saved with `state` (not a seed).
    pub fn from_state(s: u64) -> Self {
        XorShift(if s == 0 { 0x1234_5678_9ABC_DEF1 } else { s })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform-ish integer in 0..n (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// splitmix64 finaliser: derive an independent seed from two words.
pub fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Keys left out of a shard's `result_hash` (removed at every depth before hashing):
/// - `elapsed_ms`, `created`: wall-clock facts.
/// - `result_hash`: the hash itself.
/// - `threads`: thread count must not change a result (rule D4, tested).
/// - `gen_file`, `openings`, `file`: host paths. The inputs they name are pinned by
///   `gen_hash` / `a_hash` / `b_hash` / `centre_hash` and `config.openings_hash`.
///
/// `hash_mb` is deliberately NOT here: the transposition table size changes cutoffs and
/// replacement, so it can change which move is chosen. Everything else, including
/// `deterministic`, `generation` and the search settings, is hashed.
pub const HASH_EXCLUDED_KEYS: &[&str] = &[
    "elapsed_ms",
    "created",
    "result_hash",
    "threads",
    "gen_file",
    "openings",
    "file",
];

fn strip_excluded(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(m) => {
            m.retain(|k, _| !HASH_EXCLUDED_KEYS.contains(&k.as_str()));
            m.values_mut().for_each(strip_excluded);
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(strip_excluded),
        _ => {}
    }
}

/// D6: FNV-1a 64 (16 hex digits) of the compact canonical JSON of `v` without the
/// `HASH_EXCLUDED_KEYS`. Object keys are sorted (serde_json's `Map` is a `BTreeMap`) and
/// numbers are written the way serde_json writes them.
pub fn canonical_hash(v: &serde_json::Value) -> String {
    let mut v = v.clone();
    strip_excluded(&mut v);
    format!(
        "{:016x}",
        crate::selfgame::fnv1a64(v.to_string().as_bytes())
    )
}

/// `canonical_hash` of any serialisable shard.
pub fn result_hash_of<T: serde::Serialize>(t: &T) -> String {
    canonical_hash(&serde_json::to_value(t).expect("shard serialises"))
}

/// FNV-1a 64 of a file's bytes (pins an input such as the opening suite).
pub fn file_hash(path: &std::path::Path) -> Result<String, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("{:016x}", crate::selfgame::fnv1a64(&b)))
}

/// Read a shard file, check its `result_hash` against the recomputed one, deserialise.
pub fn load_shard<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
    let stored = v
        .get("result_hash")
        .and_then(|h| h.as_str())
        .ok_or_else(|| {
            format!(
                "{}: no result_hash (shard from an older binary)",
                path.display()
            )
        })?;
    let want = canonical_hash(&v);
    if stored != want {
        return Err(format!(
            "{}: result_hash {stored} does not match the content ({want}); shard is corrupt or edited",
            path.display()
        ));
    }
    serde_json::from_value(v).map_err(|e| format!("{}: {e}", path.display()))
}

/// D3: a time limit is refused unless `--allow-nondeterministic` is given.
pub fn check_time_limit(time_limited: bool, allow: bool) -> Result<(), String> {
    if time_limited && !allow {
        return Err(
            "time limits (--movetime) make a unit non-reproducible (determinism rule D3); \
             use --depth, or pass --allow-nondeterministic"
                .into(),
        );
    }
    Ok(())
}

/// D3/D7: all shards of one reduce must agree on `deterministic`.
pub fn check_same_determinism(flags: impl Iterator<Item = bool>) -> Result<(), String> {
    let v: Vec<bool> = flags.collect();
    if v.iter().any(|&d| d != v[0]) {
        return Err("refusing to mix deterministic and non-deterministic shards".into());
    }
    Ok(())
}

pub fn yes() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_000), (2024, 10, 4));
    }

    #[test]
    fn rng_deterministic() {
        let mut a = XorShift::new(7);
        let mut b = XorShift::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        assert!(a.below(10) < 10);
    }
}
