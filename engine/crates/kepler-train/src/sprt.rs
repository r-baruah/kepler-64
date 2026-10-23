//! `kepler-train sprt`: SPRT between two generation files by in-process self-play.
//!
//! Games are played in pairs (same opening, colours swapped). Pairs are scored in pair
//! order (finished pairs wait for earlier ones) so stopping early cannot bias the sample.
//! Statistics: `stats::llr_penta` (pentanomial GSPRT, normal approximation).
//!
//! The engines are deterministic, so a pair would repeat whenever an opening comes
//! round again, and identical engines would score exactly 1.0 on every pair (zero
//! variance). Both games of a pair therefore use random exploration for `--explore-plies`
//! (default 6) plies after the opening, seeded per game, as in `selfplay`.

#![allow(clippy::needless_range_loop)]

use crate::chess::End;
use crate::selfgame::{self, GameCfg, load_generation, load_openings};
use crate::stats;
use crate::util::{self, XorShift};
use clap::Args;
use kepler_core::search::{Limits, Searcher};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Args, Debug)]
pub struct SprtArgs {
    #[arg(long)]
    a: PathBuf,
    #[arg(long)]
    b: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long)]
    depth: Option<u32>,
    /// Milliseconds per move
    #[arg(long)]
    movetime: Option<u64>,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long = "max-games", default_value_t = 4000)]
    max_games: usize,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    elo0: f64,
    #[arg(long, default_value_t = 10.0, allow_hyphen_values = true)]
    elo1: f64,
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,
    #[arg(long, default_value_t = 0.05)]
    beta: f64,
    /// Do not stop on a bound before this many games (the normal approximation is
    /// unreliable on a handful of pairs)
    #[arg(long = "min-games", default_value_t = 100)]
    min_games: usize,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    #[arg(long = "explore-plies", default_value_t = 6)]
    explore_plies: usize,
    #[arg(long, default_value_t = 8)]
    hash: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Default)]
struct Shared {
    next: usize,
    done: BTreeMap<usize, Result<(f64, [u32; 3]), String>>,
    evaluated: usize,
    penta: [u32; 5],
    wdl: [u32; 3],
    llr: f64,
    stop: Option<&'static str>,
    err: Option<String>,
}

/// Points for engine A (white `a_white`) from the result.
fn points(e: End, a_white: bool) -> f64 {
    match (e, a_white) {
        (End::Draw, _) => 0.5,
        (End::WhiteWins, true) | (End::BlackWins, false) => 1.0,
        _ => 0.0,
    }
}

pub fn run(a: SprtArgs) -> Result<(), String> {
    let ga = load_generation(&a.a)?;
    let gb = load_generation(&a.b)?;
    let (pa, pb) = (ga.constants, gb.constants);
    let openings = load_openings(&a.openings)?;
    let limits = match (a.depth, a.movetime) {
        (Some(d), None) => Limits {
            depth: Some(d),
            ..Default::default()
        },
        (None, Some(m)) => Limits {
            movetime_ms: Some(m),
            ..Default::default()
        },
        _ => return Err("give exactly one of --depth, --movetime".into()),
    };
    let cfg = GameCfg {
        limits,
        explore_depth: a.depth.map_or(3, |d| d.saturating_sub(1).max(1)),
        max_plies: a.max_plies,
        explore_plies: a.explore_plies,
        explore_margin: 0.15,
        record: false,
        scale: 4.0,
    };
    let upper = ((1.0 - a.beta) / a.alpha).ln();
    let lower = (a.beta / (1.0 - a.alpha)).ln();
    let max_pairs = a.max_games / 2;
    // shuffled opening order, repeated as needed
    let mut order: Vec<usize> = (0..openings.len()).collect();
    let mut rng = XorShift::new(a.seed);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let shared = Mutex::new(Shared::default());
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..a.threads.clamp(1, max_pairs.max(1)) {
            sc.spawn(|| {
                let mut st = [Searcher::new(a.hash), Searcher::new(a.hash)];
                loop {
                    let k = {
                        let mut s = shared.lock().unwrap();
                        if s.stop.is_some() || s.next >= max_pairs {
                            break;
                        }
                        s.next += 1;
                        s.next - 1
                    };
                    let o = &openings[order[k % order.len()]];
                    let mut pts = 0.0;
                    let mut wdl = [0u32; 3];
                    let mut res: Result<(), String> = Ok(());
                    for a_white in [true, false] {
                        // distinct seed per game: with a shared seed, identical engines would
                        // replay the same game with colours swapped and every pair would tie 1.0
                        let seed = util::mix(a.seed, 2 * k as u64 + a_white as u64);
                        st[0].clear();
                        st[1].clear();
                        let [sa, sb] = &mut st;
                        let g = if a_white {
                            selfgame::play_game(&cfg, o, [sa, sb], [&pa, &pb], seed)
                        } else {
                            selfgame::play_game(&cfg, o, [sb, sa], [&pb, &pa], seed)
                        };
                        match g {
                            Ok(g) => {
                                let p = points(g.result, a_white);
                                pts += p;
                                wdl[if p == 1.0 {
                                    0
                                } else if p == 0.5 {
                                    1
                                } else {
                                    2
                                }] += 1;
                            }
                            Err(e) => res = Err(e),
                        }
                    }
                    let mut s = shared.lock().unwrap();
                    s.done.insert(k, res.map(|_| (pts, wdl)));
                    while s.stop.is_none() {
                        let ev = s.evaluated;
                        let Some(r) = s.done.remove(&ev) else { break };
                        s.evaluated += 1;
                        match r {
                            Err(e) => {
                                s.err = Some(e);
                                s.stop = Some("error");
                            }
                            Ok((pts, wdl)) => {
                                s.penta[((pts * 2.0).round() as usize).min(4)] += 1;
                                for i in 0..3 {
                                    s.wdl[i] += wdl[i];
                                }
                                s.llr = stats::llr_penta(&s.penta, a.elo0, a.elo1);
                                if s.evaluated * 2 < a.min_games {
                                } else if s.llr >= upper {
                                    s.stop = Some("H1");
                                } else if s.llr <= lower {
                                    s.stop = Some("H0");
                                }
                                if s.evaluated % 20 == 0 {
                                    eprintln!(
                                        "{} games, llr {:.3} [{:.3}, {:.3}], {:.1}s",
                                        s.evaluated * 2,
                                        s.llr,
                                        lower,
                                        upper,
                                        t0.elapsed().as_secs_f64()
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    let s = shared.into_inner().unwrap();
    if let Some(e) = s.err {
        return Err(e);
    }
    let result = s.stop.unwrap_or("inconclusive");
    let games = s.evaluated * 2;
    let (elo, lo, hi) = stats::elo_penta(&s.penta);
    let secs = t0.elapsed().as_secs_f64();
    let doc = json!({
        "llr": s.llr,
        "bounds": { "lower": lower, "upper": upper },
        "result": result,
        "pentanomial": s.penta,
        "wdl": { "w": s.wdl[0], "d": s.wdl[1], "l": s.wdl[2] },
        "elo": { "estimate": elo, "lo95": lo, "hi95": hi, "reliable": games >= 200 },
        "games": games,
        "seconds": secs,
        "config": {
            "a": { "file": a.a, "generation": ga.generation },
            "b": { "file": a.b, "generation": gb.generation },
            "depth": a.depth, "movetime_ms": a.movetime,
            "elo0": a.elo0, "elo1": a.elo1, "alpha": a.alpha, "beta": a.beta,
            "max_games": a.max_games, "min_games": a.min_games, "max_plies": a.max_plies,
            "explore_plies": a.explore_plies, "openings": a.openings,
            "seed": a.seed, "created": util::utc_iso(),
            "note": "result is for A versus B; pentanomial = pairs scoring 0, 0.5, 1, 1.5, 2 for A",
        },
    });
    std::fs::write(&a.out, serde_json::to_string_pretty(&doc).unwrap() + "\n")
        .map_err(|e| format!("{}: {e}", a.out.display()))?;
    println!(
        "SPRT {result}: llr {:.3} in [{:.3}, {:.3}], {games} games (W{} D{} L{}), Elo {:+.1} [{:+.1}, {:+.1}], {:.1}s",
        s.llr, lower, upper, s.wdl[0], s.wdl[1], s.wdl[2], elo, lo, hi, secs
    );
    Ok(())
}
