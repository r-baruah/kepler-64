//! `kepler-train suite-build-tactics`: a tactics suite drawn from Kepler's own ladder games.
//!
//! Every position from ply 10 on of every game in a ladder directory is a candidate. Stockfish
//! (MultiPV 2, fixed depth) examines a seeded shuffle of the candidates and a position is kept
//! when the best move wins clearly, is the only winning move, and the position was not already
//! won. Stockfish is the examiner only: the suite is used for measurement, never for training.
//! Deterministic given `--seed`, the source files and the Stockfish build.

use crate::chess;
use crate::mates::mate1_moves;
use crate::uci::{Engine, EngineSpec, Score};
use crate::util::{self, XorShift};
use clap::Args;
use cozy_chess::{Board, GameStatus};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

/// Best move must score at least this much for the side to move (centipawns).
pub const MIN_BEST: i32 = 200;
/// The best move must not be a trivial cleanup: score at most this.
pub const MAX_BEST: i32 = 600;
/// The second-best move must be at least this much worse than the best.
pub const MIN_GAP: i32 = 200;

#[derive(Args, Debug)]
pub struct TacticsArgs {
    /// Stockfish binary (default: env KEPLER_STOCKFISH)
    #[arg(long)]
    stockfish: Option<PathBuf>,
    /// Directory of ladder result JSON files (each has `games` with `opening` and `moves_uci`)
    #[arg(long, default_value = "engine/results/ladder-gen002")]
    games: PathBuf,
    #[arg(long, default_value_t = 200)]
    count: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long = "sf-depth", default_value_t = 16)]
    sf_depth: u32,
    #[arg(long = "sf-threads", default_value_t = 4)]
    sf_threads: u32,
    /// Shallow pre-screen depth (relaxed thresholds); 0 disables. Only positions that pass it
    /// get the full-depth examination, which alone decides what is kept.
    #[arg(long = "pre-depth", default_value_t = 8)]
    pre_depth: u32,
    /// First ply (moves already played) to consider
    #[arg(long = "min-ply", default_value_t = 10)]
    min_ply: usize,
    /// Most positions kept from one game
    #[arg(long = "per-game", default_value_t = 2)]
    per_game: usize,
    /// Give up after examining this many positions
    #[arg(long = "max-scan", default_value_t = 30_000)]
    max_scan: usize,
    #[arg(long, default_value = "engine/data/suites/tactics-own.epd")]
    out: PathBuf,
}

/// The selection filter. `best` and `second` are Stockfish scores for the side to move.
/// Mate scores are never kept (a mate-in-n best move is not a tactics-suite position).
pub fn accept(best: Score, second: Option<Score>) -> bool {
    let Score::Cp(b) = best else { return false };
    if !(MIN_BEST..=MAX_BEST).contains(&b) {
        return false;
    }
    match second {
        None => false,
        Some(s) => b - s.to_cp() >= MIN_GAP,
    }
}

/// Loose version of `accept` for the shallow pre-screen: it only decides what is worth a
/// full-depth look, never what is kept.
pub fn accept_prescreen(best: Score, second: Option<Score>) -> bool {
    let Score::Cp(b) = best else { return false };
    (MIN_BEST / 2..=MAX_BEST + 300).contains(&b)
        && second.is_some_and(|s| b - s.to_cp() >= MIN_GAP / 2)
}

fn fmt_score(s: Score) -> String {
    match s {
        Score::Cp(c) => format!("{c:+}"),
        Score::Mate(m) => format!("M{m}"),
    }
}

fn fen4(b: &Board) -> String {
    b.to_string()
        .split(' ')
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

struct Cand {
    game: usize,
    file: String,
    gi: usize,
    ply: usize,
    board: Board,
}

struct Kept {
    cand: usize,
    san: String,
    best: Score,
    second: Score,
}

fn collect(args: &TacticsArgs) -> Result<(Vec<Cand>, usize, Vec<String>), String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(&args.games)
        .map_err(|e| format!("{}: {e}", args.games.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    let (mut out, mut seen) = (Vec::new(), HashSet::new());
    let (mut ngames, mut names) = (0usize, Vec::new());
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", f.display()))?;
        let Some(games) = v["games"].as_array() else {
            continue;
        };
        names.push(name.clone());
        for (gi, g) in games.iter().enumerate() {
            let (Some(op), Some(mv)) = (g["opening"].as_str(), g["moves_uci"].as_array()) else {
                continue;
            };
            let mut b = chess::board_from_fen4(op)?;
            let game = ngames;
            ngames += 1;
            for (ply, m) in mv.iter().enumerate() {
                if ply >= args.min_ply
                    && b.status() == GameStatus::Ongoing
                    && chess::legal_moves(&b).len() >= 2
                    && seen.insert(fen4(&b))
                {
                    out.push(Cand {
                        game,
                        file: name.clone(),
                        gi,
                        ply,
                        board: b.clone(),
                    });
                }
                let mv = chess::uci_to_move(&b, m.as_str().unwrap_or(""))?;
                b.play_unchecked(mv);
            }
        }
    }
    Ok((out, ngames, names))
}

pub fn run(args: TacticsArgs) -> Result<(), String> {
    let sf_path = match &args.stockfish {
        Some(p) => p.to_string_lossy().to_string(),
        None => std::env::var("KEPLER_STOCKFISH")
            .map_err(|_| "no --stockfish and KEPLER_STOCKFISH is not set".to_string())?,
    };
    let spec = EngineSpec::new(
        &sf_path,
        &[],
        None,
        &[
            format!("Threads={}", args.sf_threads),
            "Hash=64".into(),
            "MultiPV=2".into(),
        ],
    )?;
    let mut sf = Engine::start(&spec)?;
    let sf_name = sf.id_name.clone();
    let (cands, ngames, names) = collect(&args)?;
    let mut order: Vec<usize> = (0..cands.len()).collect();
    let mut rng = XorShift::new(args.seed);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let mut per_game = vec![0usize; ngames];
    let mut kept: Vec<Kept> = Vec::new();
    let (mut scanned, mut prescreen_pass) = (0usize, 0usize);
    for &ci in &order {
        if kept.len() >= args.count || scanned >= args.max_scan {
            break;
        }
        let c = &cands[ci];
        if per_game[c.game] >= args.per_game {
            continue;
        }
        scanned += 1;
        let pos = format!("position fen {}", c.board);
        let look = |sf: &mut Engine, depth: u32| -> Result<Vec<(Score, String)>, String> {
            sf.new_game()?;
            sf.go_multi(&pos, &format!("depth {depth}"), 2, Duration::from_secs(60))
        };
        if args.pre_depth > 0 {
            match look(&mut sf, args.pre_depth) {
                Ok(r) => {
                    let (Some(&(b, _)), s) = (r.first(), r.get(1).map(|x| x.0)) else {
                        continue;
                    };
                    if !accept_prescreen(b, s) {
                        continue;
                    }
                    prescreen_pass += 1;
                }
                Err(_) => {
                    sf = Engine::start(&spec)?;
                    continue;
                }
            }
        }
        let r = match look(&mut sf, args.sf_depth) {
            Ok(r) => r,
            Err(_) => {
                sf = Engine::start(&spec)?;
                continue;
            }
        };
        let (Some(&(best, ref bm)), second) = (r.first(), r.get(1).map(|x| x.0)) else {
            continue;
        };
        if !accept(best, second) {
            continue;
        }
        let Ok(mv) = chess::uci_to_move(&c.board, bm) else {
            continue;
        };
        if mate1_moves(&c.board).contains(&mv) {
            continue;
        }
        per_game[c.game] += 1;
        kept.push(Kept {
            cand: ci,
            san: chess::move_to_san(&c.board, mv),
            best,
            second: second.unwrap(),
        });
        if kept.len().is_multiple_of(20) {
            eprintln!("kept {}/{} after {scanned} scanned", kept.len(), args.count);
        }
    }
    if kept.len() < args.count {
        return Err(format!(
            "gave up: kept {} of {} after scanning {scanned} of {} candidates",
            kept.len(),
            args.count,
            cands.len()
        ));
    }
    let mut text = format!(
        "# tactics-own: built from Kepler-64's own ladder games against Stockfish 19 (gen-002 vs a strength-limited Stockfish).\n\
         # Stockfish ({sf_name}) was used only as the examiner, to verify each position. The suite is for measurement only and is never used for training.\n\
         # source: {} game files in {} ({} games, {} distinct candidate positions from ply {} on, both sides to move): {}\n\
         # kept when Stockfish (MultiPV 2, depth {}, Threads {}, Hash 64) scores the best move in [+{MIN_BEST}, +{MAX_BEST}] cp for the side to move, not a mate-in-1,\n\
         # and the second-best move is at least {MIN_GAP} cp worse (a unique winning move); positions with fewer than 2 legal moves skipped; deduplicated by FEN (first four fields); at most {} per game.\n\
         # selection: candidates shuffled with seed {}, examined in that order until {} were kept ({scanned} examined, {prescreen_pass} passed the depth-{} pre-screen with relaxed thresholds [+{}, +{}] cp and gap {}; thresholds below decide).\n\
         # command: kepler-train suite-build-tactics --games {} --count {} --seed {} --sf-depth {} --sf-threads {} --pre-depth {} --min-ply {} --per-game {} --out {}; built {}\n\
         # c0 holds Stockfish's best and second-best scores in cp for the side to move (centipawns here are Stockfish's, not Kepler units).\n",
        names.len(),
        args.games.display(),
        ngames,
        cands.len(),
        args.min_ply,
        names
            .first()
            .zip(names.last())
            .map(|(a, b)| format!("{a} .. {b}"))
            .unwrap_or_default(),
        args.sf_depth,
        args.sf_threads,
        args.per_game,
        args.seed,
        args.count,
        args.pre_depth,
        MIN_BEST / 2,
        MAX_BEST + 300,
        MIN_GAP / 2,
        args.games.display(),
        args.count,
        args.seed,
        args.sf_depth,
        args.sf_threads,
        args.pre_depth,
        args.min_ply,
        args.per_game,
        args.out.display(),
        util::utc_date(),
    );
    for (i, k) in kept.iter().enumerate() {
        let c = &cands[k.cand];
        text.push_str(&format!(
            "{} bm {}; id \"{}#{} ply{} t{:03}\"; c0 \"best {} second {}\";\n",
            fen4(&c.board),
            k.san,
            c.file.trim_end_matches(".json"),
            c.gi,
            c.ply,
            i + 1,
            fmt_score(k.best),
            fmt_score(k.second),
        ));
    }
    if let Some(p) = args.out.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    std::fs::write(&args.out, text).map_err(|e| format!("{}: {e}", args.out.display()))?;
    println!(
        "tactics: {} candidates, {scanned} examined, {} kept -> {}",
        cands.len(),
        kept.len(),
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use Score::{Cp, Mate};

    #[test]
    fn filter_thresholds() {
        assert!(accept(Cp(300), Some(Cp(50))));
        assert!(accept(Cp(200), Some(Cp(0))));
        assert!(accept(Cp(600), Some(Cp(400))));
        assert!(!accept(Cp(199), Some(Cp(-500))), "best too low");
        assert!(!accept(Cp(601), Some(Cp(-500))), "already won");
        assert!(!accept(Cp(300), Some(Cp(101))), "second within 200");
        assert!(!accept(Cp(300), None), "single legal move");
        assert!(!accept(Mate(1), Some(Cp(0))), "mate is not kept");
        assert!(accept(Cp(300), Some(Mate(-3))), "second loses to mate");
        assert!(!accept(Cp(300), Some(Mate(3))));
        assert!(accept_prescreen(Cp(120), Some(Cp(0))));
        assert!(!accept_prescreen(Cp(50), Some(Cp(-200))));
    }

    #[test]
    fn committed_suite_parses_and_is_unique_bm() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/suites/tactics-own.epd");
        let rows = crate::epd::parse_file(&path).unwrap();
        assert_eq!(rows.len(), 200);
        for r in rows.iter().take(25) {
            let b = chess::board_from_fen4(&r.fen4).unwrap();
            let bm = chess::san_to_move(&b, r.op("bm").unwrap()).unwrap();
            assert!(!mate1_moves(&b).contains(&bm));
            assert!(r.op("c0").is_some() && r.op("id").is_some());
        }
    }
}
