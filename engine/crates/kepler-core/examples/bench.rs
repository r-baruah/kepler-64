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
    for (name, fen) in fens {
        let pos = Position::from_fen(fen).unwrap();
        let mut s = Searcher::new(64);
        let lim = Limits {
            depth: Some(depth),
            ..Default::default()
        };
        let r = s.search(&pos, &[], &Params::default(), &lim, &mut |_| {});
        let nps = r.nodes * 1000 / r.elapsed_ms.max(1);
        println!(
            "{name}: depth {} best {} score {:.3} nodes {} time {} ms nps {}",
            r.depth,
            move_to_uci(&pos.board, r.best.unwrap()),
            r.score_ku,
            r.nodes,
            r.elapsed_ms,
            nps
        );
    }
}
