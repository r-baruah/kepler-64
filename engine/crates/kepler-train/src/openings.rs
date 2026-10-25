//! `kepler-train openings build`: random-play openings filtered by Stockfish.

use crate::chess;
use crate::uci::{Engine, EngineSpec};
use crate::util::{self, XorShift};
use clap::Args;
use cozy_chess::Board;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Stockfish binary (default: env KEPLER_STOCKFISH)
    #[arg(long)]
    stockfish: Option<PathBuf>,
    #[arg(long, default_value_t = 200)]
    count: usize,
    #[arg(long = "min-plies", default_value_t = 4)]
    min_plies: usize,
    #[arg(long = "max-plies", default_value_t = 8)]
    max_plies: usize,
    #[arg(long = "sf-depth", default_value_t = 14)]
    sf_depth: u32,
    #[arg(long = "max-cp", default_value_t = 50)]
    max_cp: i32,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value = "engine/data/openings/balanced-200.epd")]
    out: PathBuf,
}

fn fen4(b: &Board) -> String {
    b.to_string()
        .split(' ')
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn run(args: BuildArgs) -> Result<(), String> {
    if args.min_plies > args.max_plies || args.min_plies == 0 {
        return Err("need 1 <= min-plies <= max-plies".into());
    }
    let sf_path = match &args.stockfish {
        Some(p) => p.to_string_lossy().to_string(),
        None => std::env::var("KEPLER_STOCKFISH")
            .map_err(|_| "no --stockfish and KEPLER_STOCKFISH is not set".to_string())?,
    };
    let mut sf = Engine::start(&EngineSpec::plain(&sf_path))?;
    let mut rng = XorShift::new(args.seed);
    let mut seen: HashSet<String> = HashSet::new();
    let mut kept: Vec<Board> = Vec::new();
    let (mut tried, mut dupes, mut unbalanced, mut dead) = (0u64, 0u64, 0u64, 0u64);
    let t0 = Instant::now();
    while kept.len() < args.count {
        tried += 1;
        if tried > 1_000_000 {
            return Err("giving up after 1,000,000 attempts".into());
        }
        let plies = args.min_plies + rng.below(args.max_plies - args.min_plies + 1);
        let mut b = Board::default();
        let mut ok = true;
        for _ in 0..plies {
            let mv = chess::legal_moves(&b);
            if mv.is_empty() {
                ok = false;
                break;
            }
            b.play(mv[rng.below(mv.len())]);
        }
        if !ok || chess::legal_moves(&b).is_empty() {
            dead += 1;
            continue;
        }
        let key = fen4(&b);
        if seen.contains(&key) {
            dupes += 1;
            continue;
        }
        seen.insert(key);
        sf.new_game()?;
        let r = sf.go(
            &format!("position fen {b}"),
            &format!("depth {}", args.sf_depth),
            Duration::from_secs(120),
        )?;
        match r.score {
            Some(s) if s.to_cp().abs() <= args.max_cp && matches!(s, crate::uci::Score::Cp(_)) => {
                kept.push(b);
                if kept.len().is_multiple_of(20) {
                    eprintln!("kept {}/{} after {tried} tries", kept.len(), args.count);
                }
            }
            _ => unbalanced += 1,
        }
    }
    let mut text = format!(
        "# built by kepler-train openings build --seed {} --sf-depth {} --max-cp {} on {}\n",
        args.seed,
        args.sf_depth,
        args.max_cp,
        util::utc_date()
    );
    for (i, b) in kept.iter().enumerate() {
        text.push_str(&format!("{b} ; id \"open-{:03}\"\n", i + 1));
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(&args.out, text).map_err(|e| format!("{}: {e}", args.out.display()))?;
    println!(
        "openings: tried {tried}, kept {}, duplicates {dupes}, unbalanced {unbalanced}, dead {dead}, {:.1}s -> {}",
        kept.len(),
        t0.elapsed().as_secs_f64(),
        args.out.display()
    );
    Ok(())
}
