//! `kepler-train suite-build-mates`: build mate-in-1 and mate-in-2 suites.
//!
//! Positions come from random playouts that start at the opening suite. A small exact
//! mate solver (cozy-chess, no evaluation) finds candidates with a unique solution; each is
//! then verified by Stockfish (mate score of the right length, same first move).
//! Deterministic given `--seed` and the Stockfish build.

use crate::chess;
use crate::epd;
use crate::uci::{Engine, EngineSpec, Score};
use crate::util::{self, XorShift};
use clap::Args;
use cozy_chess::{Board, GameStatus, Move};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Args, Debug)]
pub struct MatesArgs {
    /// Stockfish binary (default: env KEPLER_STOCKFISH)
    #[arg(long)]
    stockfish: Option<PathBuf>,
    #[arg(long, default_value = "engine/data/openings/balanced-200.epd")]
    openings: PathBuf,
    #[arg(long, default_value_t = 100)]
    count: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long = "sf-depth", default_value_t = 14)]
    sf_depth: u32,
    #[arg(long = "sf-threads", default_value_t = 1)]
    sf_threads: u32,
    /// Plies of random play after the opening, at most
    #[arg(long = "max-plies", default_value_t = 120)]
    max_plies: usize,
    #[arg(long, default_value = "engine/data/suites/mate1.epd")]
    out1: PathBuf,
    #[arg(long, default_value = "engine/data/suites/mate2.epd")]
    out2: PathBuf,
    /// Give up after this many random games
    #[arg(long = "max-games", default_value_t = 200_000)]
    max_games: u64,
}

fn after(b: &Board, mv: Move) -> Board {
    let mut c = b.clone();
    c.play_unchecked(mv);
    c
}

/// Moves that checkmate at once.
pub fn mate1_moves(b: &Board) -> Vec<Move> {
    chess::legal_moves(b)
        .into_iter()
        .filter(|&m| after(b, m).status() == GameStatus::Won)
        .collect()
}

/// Moves after which every reply allows a mate in one (and the game is not over).
pub fn mate2_moves(b: &Board) -> Vec<Move> {
    let mut out = Vec::new();
    'mv: for m in chess::legal_moves(b) {
        let c = after(b, m);
        if c.status() != GameStatus::Ongoing {
            continue;
        }
        for r in chess::legal_moves(&c) {
            if mate1_moves(&after(&c, r)).is_empty() {
                continue 'mv;
            }
        }
        out.push(m);
    }
    out
}

fn fen4(b: &Board) -> String {
    b.to_string()
        .split(' ')
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

struct Found {
    board: Board,
    mv: Move,
}

fn verify(sf: &mut Engine, f: &Found, mate_len: i32, depth: u32) -> Result<bool, String> {
    sf.new_game()?;
    let r = sf.go(
        &format!("position fen {}", f.board),
        &format!("depth {depth}"),
        Duration::from_secs(120),
    )?;
    Ok(r.score == Some(Score::Mate(mate_len)) && r.bestmove == chess::move_to_uci(&f.board, f.mv))
}

fn write_suite(
    path: &PathBuf,
    kind: &str,
    a: &MatesArgs,
    sf_name: &str,
    list: &[Found],
) -> Result<(), String> {
    let mut text = format!(
        "# built by kepler-train suite-build-mates --seed {} --sf-depth {} --count {} on {}\n\
         # source: random playouts (up to {} plies) from {}; Stockfish ({sf_name}) verified\n\
         # each position has a mate in exactly {kind} with a unique first move (exact solver and Stockfish agree)\n",
        a.seed,
        a.sf_depth,
        a.count,
        util::utc_date(),
        a.max_plies,
        a.openings
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
    );
    for (i, f) in list.iter().enumerate() {
        text.push_str(&format!(
            "{} bm {}; id \"mate{kind}-{:03}\";\n",
            fen4(&f.board),
            chess::move_to_san(&f.board, f.mv),
            i + 1
        ));
    }
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn run(args: MatesArgs) -> Result<(), String> {
    let sf_path = match &args.stockfish {
        Some(p) => p.to_string_lossy().to_string(),
        None => std::env::var("KEPLER_STOCKFISH")
            .map_err(|_| "no --stockfish and KEPLER_STOCKFISH is not set".to_string())?,
    };
    let spec = EngineSpec::new(
        &sf_path,
        &[],
        None,
        &[format!("Threads={}", args.sf_threads), "Hash=64".into()],
    )?;
    let mut sf = Engine::start(&spec)?;
    let sf_name = sf.id_name.clone();
    let openings: Vec<Board> = epd::parse_file(&args.openings)?
        .iter()
        .map(|e| chess::board_from_fen4(&e.fen4))
        .collect::<Result<_, _>>()?;
    if openings.is_empty() {
        return Err("no openings".into());
    }
    let mut rng = XorShift::new(args.seed);
    let (mut l1, mut l2): (Vec<Found>, Vec<Found>) = (vec![], vec![]);
    let mut seen: HashSet<String> = HashSet::new();
    let (mut games, mut sf_reject) = (0u64, 0u64);
    while (l1.len() < args.count || l2.len() < args.count) && games < args.max_games {
        games += 1;
        let mut b = openings[rng.below(openings.len())].clone();
        let (mut got1, mut got2) = (false, false);
        for _ in 0..args.max_plies {
            let mv = chess::legal_moves(&b);
            if mv.is_empty() || b.status() != GameStatus::Ongoing {
                break;
            }
            b.play_unchecked(mv[rng.below(mv.len())]);
            if b.status() != GameStatus::Ongoing {
                break;
            }
            let m1 = mate1_moves(&b);
            let cand = if !m1.is_empty() {
                if got1 || l1.len() >= args.count || m1.len() != 1 {
                    continue;
                }
                Some((1, m1[0]))
            } else if !got2 && l2.len() < args.count {
                let m2 = mate2_moves(&b);
                if m2.len() == 1 {
                    Some((2, m2[0]))
                } else {
                    None
                }
            } else {
                None
            };
            let Some((len, mv)) = cand else { continue };
            if !seen.insert(fen4(&b)) {
                continue;
            }
            let f = Found {
                board: b.clone(),
                mv,
            };
            if !verify(&mut sf, &f, len, args.sf_depth)? {
                sf_reject += 1;
                continue;
            }
            if len == 1 {
                got1 = true;
                l1.push(f);
            } else {
                got2 = true;
                l2.push(f);
            }
            if (l1.len() + l2.len()) % 20 == 0 {
                eprintln!(
                    "mate1 {}/{} mate2 {}/{} after {games} games",
                    l1.len(),
                    args.count,
                    l2.len(),
                    args.count
                );
            }
        }
    }
    if l1.len() < args.count || l2.len() < args.count {
        return Err(format!(
            "gave up: mate1 {}, mate2 {} after {games} games",
            l1.len(),
            l2.len()
        ));
    }
    write_suite(&args.out1, "1", &args, &sf_name, &l1)?;
    write_suite(&args.out2, "2", &args, &sf_name, &l2)?;
    println!(
        "mates: {games} games, {sf_reject} Stockfish rejects -> {} and {}",
        args.out1.display(),
        args.out2.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(fen4: &str) -> Board {
        chess::board_from_fen4(fen4).unwrap()
    }

    #[test]
    fn solver_mate1() {
        let m = mate1_moves(&b("6k1/5ppp/8/8/8/8/8/R5K1 w - -"));
        assert_eq!(m.len(), 1);
        assert_eq!(
            chess::move_to_san(&b("6k1/5ppp/8/8/8/8/8/R5K1 w - -"), m[0]),
            "Ra8#"
        );
    }

    #[test]
    fn solver_mate2() {
        let p = b("7k/8/8/8/8/8/R7/1R4K1 w - -");
        assert!(mate1_moves(&p).is_empty());
        assert!(!mate2_moves(&p).is_empty());
    }

    /// The committed suites parse (comment header included) and every `bm` is the unique
    /// solution as the exact solver sees it; this is also what `suite` scores `bm_hits` on.
    #[test]
    fn committed_suites_bm_parse_and_solve() {
        for (name, n) in [("mate1", 1), ("mate2", 2)] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../data/suites/{name}.epd"));
            let rows = epd::parse_file(&path).unwrap();
            assert_eq!(rows.len(), 100, "{name}");
            for r in rows.iter().take(25) {
                let board = chess::board_from_fen4(&r.fen4).unwrap();
                let bm = chess::san_to_move(&board, r.op("bm").unwrap()).unwrap();
                let sols = if n == 1 {
                    mate1_moves(&board)
                } else {
                    mate2_moves(&board)
                };
                assert_eq!(sols, vec![bm], "{name} {}", r.fen4);
            }
        }
    }
}
