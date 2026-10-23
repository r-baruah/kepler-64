//! Transposition table: two slots per bucket, depth-preferred + always-replace.

use cozy_chess::Move;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: u64,
    pub score: f32,
    pub mv: Option<Move>,
    pub depth: i16,
    pub bound: Bound,
    pub age: u8,
    pub used: bool,
}

const EMPTY: Entry = Entry {
    key: 0,
    score: 0.0,
    mv: None,
    depth: 0,
    bound: Bound::Exact,
    age: 0,
    used: false,
};

pub struct Tt {
    slots: Vec<Entry>,
    buckets: usize,
    pub age: u8,
}

impl Tt {
    pub fn new(mb: usize) -> Self {
        let bytes = mb.max(1) * 1024 * 1024;
        let buckets = (bytes / (2 * std::mem::size_of::<Entry>())).max(1);
        Tt {
            slots: vec![EMPTY; buckets * 2],
            buckets,
            age: 0,
        }
    }

    pub fn clear(&mut self) {
        self.slots.fill(EMPTY);
        self.age = 0;
    }

    #[inline]
    fn index(&self, key: u64) -> usize {
        (((key as u128) * (self.buckets as u128)) >> 64) as usize * 2
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        let i = self.index(key);
        for e in &self.slots[i..i + 2] {
            if e.used && e.key == key {
                return Some(*e);
            }
        }
        None
    }

    pub fn store(&mut self, key: u64, depth: i16, score: f32, bound: Bound, mv: Option<Move>) {
        let i = self.index(key);
        let age = self.age;
        let mut new = Entry {
            key,
            score,
            mv,
            depth,
            bound,
            age,
            used: true,
        };
        let first = self.slots[i];
        if first.used && first.key == key {
            // Same position: keep the old move if we have none.
            if new.mv.is_none() {
                new.mv = first.mv;
            }
            if depth >= first.depth || bound == Bound::Exact {
                self.slots[i] = new;
            } else if new.mv.is_some() {
                self.slots[i].mv = new.mv;
            }
            return;
        }
        if !first.used || first.age != age || depth >= first.depth {
            self.slots[i] = new;
        } else {
            let second = self.slots[i + 1];
            if second.used && second.key == key && new.mv.is_none() {
                new.mv = second.mv;
            }
            self.slots[i + 1] = new;
        }
    }
}
