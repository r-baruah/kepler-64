//! Regression: every principal variation must be a legal line. A stale tail
//! from a non-PV child once produced illegal PV moves, which crashed the UCI
//! layer in 29 ladder games (gen002, 2026-10-24).

use kepler_core::search::{Limits, Searcher};
use kepler_core::{Params, Position};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn every_reported_pv_is_legal() {
    let p = Params::default();
    let mut rng = Rng(0xC0FF_EE11_2233_4455);
    let mut s = Searcher::new(8);
    let mut checked = 0;
    for _ in 0..60 {
        let mut pos = Position::startpos();
        let mut history = Vec::new();
        for _ in 0..(10 + rng.next() % 60) {
            let mut ms = Vec::new();
            pos.board.generate_moves(|m| {
                ms.extend(m);
                false
            });
            if ms.is_empty() {
                break;
            }
            history.push(pos.board.hash());
            pos = pos.play(ms[(rng.next() % ms.len() as u64) as usize], &p);
        }
        if pos.board.status() != kepler_core::cozy_chess::GameStatus::Ongoing {
            continue;
        }
        let lim = Limits {
            depth: Some(5),
            ..Default::default()
        };
        let mut lines = Vec::new();
        s.search(&pos, &history, &p, &lim, &mut |r| lines.push(r.pv.clone()));
        for pv in lines {
            let mut b = pos.board.clone();
            for m in pv {
                assert!(b.is_legal(m), "illegal PV move {m} in {}", pos.board);
                b.play_unchecked(m);
            }
            checked += 1;
        }
    }
    assert!(checked > 100, "too few PVs checked: {checked}");
}
