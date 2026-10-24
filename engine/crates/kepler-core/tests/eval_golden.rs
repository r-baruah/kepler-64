//! Bit-exact golden test: evaluation and search results must not change by a
//! single bit under performance work. Regenerate (only for an intended change
//! to the laws) with `KEPLER_BLESS=1 cargo test --release -p kepler-core --test eval_golden`.

use kepler_core::search::{Limits, Searcher, move_to_uci};
use kepler_core::{Dual, Params, Position, eval_f32, evaluate, params::Generation};
use std::fmt::Write;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// 2,000 positions from seeded random play (accretion carried by `play`).
fn positions(p: &Params) -> Vec<Position> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::new();
    while out.len() < 2000 {
        let mut pos = Position::startpos();
        for _ in 0..60 {
            let mut moves = Vec::new();
            pos.board.generate_moves(|pm| {
                moves.extend(pm);
                false
            });
            if moves.is_empty() || out.len() >= 2000 {
                break;
            }
            let m = moves[(rng.next() % moves.len() as u64) as usize];
            pos = pos.play(m, p);
            out.push(pos.clone());
        }
    }
    out
}

fn param_sets() -> Vec<(String, Params)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/generations/gen-002.json"
    );
    let g = Generation::from_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut v = vec![("gen0c2".to_string(), Params::default())];
    for c in 1..=3u8 {
        let mut p = g.constants;
        p.c = c;
        v.push((format!("gen2c{c}"), p));
    }
    v
}

fn generate() -> String {
    let mut s = String::new();
    let sets = param_sets();
    let ps = positions(&Params::default());
    for (name, p) in &sets {
        for (i, pos) in ps.iter().enumerate() {
            let b = evaluate::<f32>(pos, p);
            // The portable (baseline-instruction) build must agree bit for bit.
            let portable = kepler_core::laws::fast::score_white_f32_portable(pos, p);
            let stm = if pos.board.side_to_move() == kepler_core::cozy_chess::Color::White {
                portable
            } else {
                -portable
            };
            assert_eq!(stm.to_bits(), eval_f32(pos, p).to_bits(), "{name} {i}");
            writeln!(
                s,
                "eval {name} {i} {:08x} {:08x}",
                b.score_white.to_bits(),
                eval_f32(pos, p).to_bits()
            )
            .unwrap();
        }
    }
    // Dual path (training): value and all partials, 100 positions.
    let p = Params::default();
    for (i, pos) in ps.iter().enumerate().step_by(20) {
        let b = kepler_core::eval::evaluate::<Dual>(pos, &p);
        write!(s, "dual {i} {:08x}", b.score_white.v.to_bits()).unwrap();
        for d in b.score_white.d {
            write!(s, " {:08x}", d.to_bits()).unwrap();
        }
        writeln!(s).unwrap();
    }
    // Search: 20 positions, depth 5, fresh searcher each.
    for (name, p) in sets.iter().take(2) {
        for (i, pos) in ps.iter().enumerate().step_by(100) {
            let mut sr = Searcher::new(16);
            let lim = Limits {
                depth: Some(5),
                ..Default::default()
            };
            let r = sr.search(pos, &[], p, &lim, &mut |_| {});
            let best = r
                .best
                .map(|m| move_to_uci(&pos.board, m))
                .unwrap_or_default();
            writeln!(
                s,
                "search {name} {i} {best} {:08x} {}",
                r.score_ku.to_bits(),
                r.nodes
            )
            .unwrap();
        }
    }
    s
}

#[test]
fn golden() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/eval_golden.txt");
    let got = generate();
    if std::env::var("KEPLER_BLESS").is_ok() {
        std::fs::write(path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(path).expect("golden file missing");
    for (n, (a, b)) in got.lines().zip(want.lines()).enumerate() {
        assert_eq!(a, b, "golden mismatch at line {n}");
    }
    assert_eq!(got.lines().count(), want.lines().count());
}
