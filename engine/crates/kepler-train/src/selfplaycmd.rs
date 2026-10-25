//! `kepler-train selfplay`: N games of one generation against itself, as JSONL.

use crate::chess::End;
use crate::selfgame::{
    self, AdjArgs, GameCfg, load_generation, load_openings, par_run, result_str,
};
use crate::util::{self, XorShift};
use clap::Args;
use kepler_core::search::{Limits, Searcher};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Args, Debug)]
pub struct SelfplayArgs {
    #[arg(long = "gen")]
    gen_file: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long, default_value_t = 100)]
    games: usize,
    #[arg(long, default_value_t = 4)]
    depth: u32,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    /// Exploration plies after the opening (uniform among moves within 0.15 ku)
    #[arg(long = "explore-plies", default_value_t = 8)]
    explore_plies: usize,
    /// Transposition table per searcher, MB
    #[arg(long, default_value_t = 8)]
    hash: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[command(flatten)]
    adj: AdjArgs,
    #[arg(long)]
    out: PathBuf,
}

pub fn run(a: SelfplayArgs) -> Result<(), String> {
    let gen_ = load_generation(&a.gen_file)?;
    let params = gen_.constants;
    let openings = load_openings(&a.openings)?;
    let cfg = GameCfg {
        limits: Limits {
            depth: Some(a.depth),
            ..Default::default()
        },
        explore_depth: a.depth.saturating_sub(1).max(1),
        max_plies: a.max_plies,
        explore_plies: a.explore_plies,
        explore_margin: 0.15,
        record: false,
        quiet_leaf: false,
        temperature: 4.0,
        adj: a.adj.cfg(),
    };
    let off = XorShift::new(a.seed).below(openings.len());
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    let games = par_run(
        a.games,
        a.threads,
        || [Searcher::new(a.hash), Searcher::new(a.hash)],
        |st, i| {
            let o = &openings[(off + i / 2) % openings.len()];
            st[0].clear();
            st[1].clear();
            let [s0, s1] = st;
            let g = selfgame::play_game(
                &cfg,
                o,
                [s0, s1],
                [&params, &params],
                util::mix(a.seed, i as u64),
            );
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(10) {
                eprintln!("{n}/{} games, {:.1}s", a.games, t0.elapsed().as_secs_f64());
            }
            (o.clone(), g)
        },
    );
    let secs = t0.elapsed().as_secs_f64();
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?,
    );
    let (mut w, mut d, mut l, mut capped, mut adjud, mut plies) = (0, 0, 0, 0, 0, 0usize);
    for (i, (o, g)) in games.into_iter().enumerate() {
        let g = g?;
        match g.result {
            End::WhiteWins => w += 1,
            End::Draw => d += 1,
            End::BlackWins => l += 1,
        }
        capped += g.capped as usize;
        adjud += g.adjudicated as usize;
        plies += g.moves.len();
        let line = json!({
            "game": i, "opening": o.id, "fen": o.fen4, "moves": g.moves,
            "result": result_str(g.result), "reason": g.reason,
            "capped": g.capped, "adjudicated": g.adjudicated, "plies": g.moves.len(),
        });
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
    }
    println!(
        "{} games in {:.1}s = {:.2} games/s; white {w} draw {d} black {l} ({adjud} adjudicated; of the draws, {capped} hit the {}-ply cap); mean {:.1} plies",
        a.games,
        secs,
        a.games as f64 / secs,
        a.max_plies,
        plies as f64 / a.games.max(1) as f64
    );
    Ok(())
}
