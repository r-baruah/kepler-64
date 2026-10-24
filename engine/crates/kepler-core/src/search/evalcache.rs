//! Exact evaluation cache. The key is the position's Zobrist hash plus a hash of
//! the accreted-mass array (the only other input to the evaluation besides the
//! constants), both stored in full, so a hit returns the identical `f32`.
//! The cache is dropped whenever the constants change.

use crate::eval::eval_f32;
use crate::params::Params;
use crate::position::Position;

const BITS: usize = 16;

#[derive(Clone, Copy)]
struct Entry {
    hash: u64,
    acc: u64,
    score: u32,
}

pub struct EvalCache {
    table: Vec<Entry>,
    params_key: String,
}

#[inline]
fn acc_hash(acc: &[f32; 64]) -> u64 {
    let mut h = 0u64;
    for (i, a) in acc.iter().enumerate() {
        let bits = a.to_bits();
        if bits != 0 {
            let mut x = (bits as u64) ^ ((i as u64 + 1) << 40) ^ 0x9E37_79B9_7F4A_7C15;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            h ^= x ^ (x >> 31);
        }
    }
    h
}

impl EvalCache {
    pub fn new() -> Self {
        EvalCache {
            // hash 0 with acc 0 and score 0 is a valid-looking empty entry; the
            // `acc` field of an empty slot is set to a value acc_hash can't give
            // for hash 0 positions only by chance, so mark empties explicitly.
            table: vec![
                Entry {
                    hash: 0,
                    acc: u64::MAX,
                    score: 0
                };
                1 << BITS
            ],
            params_key: String::new(),
        }
    }

    /// Call once per search: invalidates the cache if the constants differ.
    pub fn set_params(&mut self, p: &Params) {
        let key = format!("{p:?}");
        if key != self.params_key {
            self.params_key = key;
            self.table.fill(Entry {
                hash: 0,
                acc: u64::MAX,
                score: 0,
            });
        }
    }

    #[inline]
    pub fn eval(&mut self, pos: &Position, p: &Params) -> f32 {
        let hash = pos.board.hash();
        let acc = acc_hash(&pos.acc);
        let idx = ((hash ^ acc.rotate_left(17)) as usize) & ((1 << BITS) - 1);
        let e = &mut self.table[idx];
        if e.hash == hash && e.acc == acc {
            return f32::from_bits(e.score);
        }
        let s = eval_f32(pos, p);
        *e = Entry {
            hash,
            acc,
            score: s.to_bits(),
        };
        s
    }
}

impl Default for EvalCache {
    fn default() -> Self {
        Self::new()
    }
}
