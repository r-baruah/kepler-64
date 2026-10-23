//! TDLeaf diagnostics on self-play games (`kepler-train selfplay` JSONL).
//!
//! 1. Leaf fidelity: does the static eval of the replayed PV leaf match the
//!    root search score it is supposed to explain?
//! 2. Outcome scale: which `s` in tanh(eval / s) best fits game results?
//! 3. Capped games: how lopsided were games scored as draws at the ply cap?
//!
//!   cargo run --release -p kepler-core --example tdleaf_diag -- games.jsonl [depth] [stride]

use kepler_core::cozy_chess::Color;
use kepler_core::search::{Limits, Searcher, move_from_uci};
use kepler_core::{Params, Position, eval_f32, evaluate};
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let depth: u32 = args.get(2).map_or(3, |s| s.parse().unwrap());
    let stride: usize = args.get(3).map_or(3, |s| s.parse().unwrap());
    let p = Params::default();
    let mut searcher = Searcher::new(16);
    let limits = Limits { depth: Some(depth), ..Default::default() };

    let (mut n, mut big, mut sum_abs) = (0usize, 0usize, 0f64);
    let mut short_pv = 0usize;
    let mut fit: Vec<(f32, f32)> = Vec::new();
    let mut fit_root: Vec<(f32, f32)> = Vec::new();

    let f = std::fs::File::open(path).expect("games file");
    for line in std::io::BufReader::new(f).lines() {
        let g: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let fen = g["fen"].as_str().unwrap();
        let fen = if fen.split(' ').count() == 4 { format!("{fen} 0 1") } else { fen.to_string() };
        let result = match g["result"].as_str().unwrap() {
            "1-0" => 1.0f32,
            "0-1" => -1.0,
            _ => 0.0,
        };
        let capped = g["capped"].as_bool().unwrap_or(false);
        let mut pos = Position::from_fen(&fen).unwrap();
        let mut history: Vec<u64> = Vec::new();
        let moves: Vec<&str> = g["moves"].as_array().unwrap().iter().map(|m| m.as_str().unwrap()).collect();
        for (ply, mv) in moves.iter().enumerate() {
            if ply % stride == 0 {
                searcher.clear();
                let r = searcher.search(&pos, &history, &p, &limits, &mut |_| {});
                if r.score_ku.abs() < 20000.0 {
                    let mut leaf = pos.clone();
                    for m in &r.pv {
                        leaf = leaf.play(*m, &p);
                    }
                    let mover = pos.board.side_to_move();
                    let leaf_white = evaluate::<f32>(&leaf, &p).score_white;
                    let leaf_mover = if mover == Color::White { leaf_white } else { -leaf_white };
                    let d = (leaf_mover - r.score_ku).abs();
                    n += 1;
                    sum_abs += d as f64;
                    if d > 0.5 {
                        big += 1;
                    }
                    if (r.pv.len() as u32) < depth {
                        short_pv += 1;
                    }
                    if !capped {
                        let res_mover = if mover == Color::White { result } else { -result };
                        fit.push((leaf_mover, res_mover));
                        fit_root.push((r.score_ku, res_mover));
                    }
                }
            }
            history.push(pos.board.hash());
            let m = move_from_uci(&pos.board, mv).expect("legal move");
            pos = pos.play(m, &p);
        }
        if capped {
            let b = evaluate::<f32>(&pos, &p);
            println!(
                "capped game {}: material {:+.1} ku, static eval {:+.2} (white view), side to move {:+.2}",
                g["game"], b.material, b.score_white, eval_f32(&pos, &p)
            );
        }
    }

    println!("\n[1] leaf fidelity at depth {depth}: {n} plies sampled");
    println!("    mean |leaf static eval - root search score| = {:.3} ku", sum_abs / n as f64);
    println!("    share with difference > 0.5 ku: {:.1}%", 100.0 * big as f64 / n as f64);
    println!("    share with PV shorter than the depth: {:.1}%", 100.0 * short_pv as f64 / n as f64);

    for (name, data) in [("replayed leaf static eval", &fit), ("root search score", &fit_root)] {
        let mut best = (f32::MAX, 0f32);
        let mut table = Vec::new();
        for i in 1..=80 {
            let s = i as f32 * 0.25;
            let loss: f32 = data.iter().map(|(e, r)| ((e / s).tanh() - r).powi(2)).sum::<f32>() / data.len() as f32;
            if [1.0, 2.0, 4.0, 8.0, 12.0, 20.0].contains(&s) {
                table.push(format!("s={s}: {loss:.4}"));
            }
            if loss < best.0 {
                best = (loss, s);
            }
        }
        println!("\n[2] outcome-scale fit using {name} ({} positions, capped games excluded)", data.len());
        println!("    best s = {:.2} (loss {:.4});  {}", best.1, best.0, table.join(", "));
    }
}
