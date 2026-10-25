//! Native timing of the same call the wasm `Engine.search` makes:
//! depth 6 from the start position, best of 3 fresh games.
use kepler_wasm::{Game, LimitsIn};
use std::time::Instant;

fn main() {
    let gen_json = include_str!("../../../data/generations/gen-002.json");
    let mut best = u128::MAX;
    for _ in 0..3 {
        let mut g = Game::new(gen_json).unwrap();
        let t = Instant::now();
        let r = g
            .search(&LimitsIn {
                depth: Some(6),
                ..Default::default()
            })
            .unwrap();
        let ms = t.elapsed().as_millis();
        best = best.min(ms);
        println!(
            "depth {} nodes {} best {} {} ms",
            r.depth, r.nodes, r.best, ms
        );
    }
    println!("native depth 6 best of 3: {best} ms");
}
