//! `cargo run --release -p kepler-core --example bench [depth]`: search nodes/second.
use kepler_core::search::{Limits, Searcher, move_to_uci};
use kepler_core::{Params, Position};

fn main() {
    let depth: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let fens = [
        (
            "startpos",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
    ];
    // Evaluation throughput: eval_f32 over a fixed set of positions.
    {
        let p = Params::default();
        // 4,000 positions from seeded random play, so the set is diverse.
        let mut positions = Vec::new();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        while positions.len() < 4000 {
            let mut pos = Position::startpos();
            for _ in 0..60 {
                let mut moves = Vec::new();
                pos.board.generate_moves(|pm| {
                    moves.extend(pm);
                    false
                });
                if moves.is_empty() {
                    break;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                pos = pos.play(moves[(rng % moves.len() as u64) as usize], &p);
                positions.push(pos.clone());
            }
        }
        // Best of 8 windows of 300 ms.
        let mut best = 0u64;
        let mut sum = 0.0f32;
        for _ in 0..8 {
            let t = std::time::Instant::now();
            let mut n = 0u64;
            while t.elapsed().as_millis() < 300 {
                for pos in &positions {
                    sum += kepler_core::eval_f32(pos, &p);
                }
                n += positions.len() as u64;
            }
            best = best.max((n as f64 / t.elapsed().as_secs_f64()) as u64);
        }
        println!("eval: {best} evals/s (checksum {sum})");
    }
    // Best of 5 fresh searches (the machine is noisy; the node count is fixed).
    for (name, fen) in fens {
        let pos = Position::from_fen(fen).unwrap();
        let mut best_ms = u64::MAX;
        let mut last = None;
        for _ in 0..5 {
            let mut s = Searcher::new(64);
            let lim = Limits {
                depth: Some(depth),
                ..Default::default()
            };
            let r = s.search(&pos, &[], &Params::default(), &lim, &mut |_| {});
            best_ms = best_ms.min(r.elapsed_ms.max(1));
            last = Some(r);
        }
        let r = last.unwrap();
        let nps = r.nodes * 1000 / best_ms;
        println!(
            "{name}: depth {} best {} score {:.3} nodes {} time {} ms nps {}",
            r.depth,
            move_to_uci(&pos.board, r.best.unwrap()),
            r.score_ku,
            r.nodes,
            best_ms,
            nps
        );
    }
}
