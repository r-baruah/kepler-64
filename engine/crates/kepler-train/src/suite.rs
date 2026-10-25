//! `kepler-train suite`: run an EPD suite through one UCI engine.

use crate::chess;
use crate::epd;
use crate::uci::{Engine, EngineSpec};
use crate::util;
use clap::Args;
use serde_json::json;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Args, Debug)]
pub struct SuiteArgs {
    #[arg(long)]
    engine: String,
    #[arg(long = "arg", allow_hyphen_values = true)]
    arg: Vec<String>,
    #[arg(long)]
    dir: Option<PathBuf>,
    #[arg(long = "opt")]
    opt: Vec<String>,
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    depth: Option<u32>,
    #[arg(long)]
    movetime: Option<u64>,
    #[arg(long)]
    out: Option<PathBuf>,
}

fn san_list(s: Option<&str>) -> Vec<String> {
    s.map(|s| s.split_whitespace().map(String::from).collect())
        .unwrap_or_default()
}

pub fn run(args: SuiteArgs) -> Result<(), String> {
    let (go, timeout, limits) = match (args.depth, args.movetime) {
        (Some(d), None) => (
            format!("depth {d}"),
            Duration::from_secs(900),
            json!({"type": "depth", "depth": d}),
        ),
        (None, Some(m)) => (
            format!("movetime {m}"),
            Duration::from_millis(m + 10_000),
            json!({"type": "movetime", "ms": m}),
        ),
        _ => return Err("give exactly one of --depth, --movetime".into()),
    };
    let spec = EngineSpec::new(&args.engine, &args.arg, args.dir.as_deref(), &args.opt)?;
    let mut eng = Engine::start(&spec)?;
    let engine_name = eng.id_name.clone();
    let positions = epd::parse_file(&args.suite)?;
    let started = util::utc_iso();
    let t0 = Instant::now();
    let (mut am_hits, mut bm_hits, mut am_pos, mut bm_pos, mut errors) = (0, 0, 0, 0, 0);
    let mut rows = Vec::new();
    for (i, p) in positions.iter().enumerate() {
        let id = p
            .op("id")
            .map(String::from)
            .unwrap_or_else(|| format!("pos-{}", i + 1));
        let am = san_list(p.op("am"));
        let bm = san_list(p.op("bm"));
        let board = chess::board_from_fen4(&p.fen4)?;
        let parse = |l: &[String]| -> Result<Vec<_>, String> {
            l.iter()
                .map(|s| chess::san_to_move(&board, s).map_err(|e| format!("{id}: {e}")))
                .collect()
        };
        let am_mv = parse(&am)?;
        let bm_mv = parse(&bm)?;
        let fen = chess::fen_with_counters(&p.fen4);
        let _ = eng.new_game();
        let res = eng.go(&format!("position fen {fen}"), &go, timeout);
        let depth = res.as_ref().ok().and_then(|r| r.depth);
        let (uci, san, score, err) = match res {
            Ok(r) => match chess::uci_to_move(&board, &r.bestmove) {
                Ok(mv) => (
                    r.bestmove.clone(),
                    Some(chess::move_to_san(&board, mv)),
                    r.score.map(|s| s.to_cp()),
                    None,
                ),
                Err(e) => (r.bestmove.clone(), None, None, Some(e)),
            },
            Err(e) => (String::new(), None, None, Some(e)),
        };
        let played = san
            .as_deref()
            .and_then(|_| chess::uci_to_move(&board, &uci).ok());
        let am_hit = played.is_some_and(|m| am_mv.contains(&m));
        let bm_hit = played.is_some_and(|m| bm_mv.contains(&m));
        if !am.is_empty() {
            am_pos += 1;
        }
        if !bm.is_empty() {
            bm_pos += 1;
        }
        am_hits += am_hit as u32;
        bm_hits += bm_hit as u32;
        errors += err.is_some() as u32;
        eprintln!(
            "{id}: {} am_hit={am_hit} bm_hit={bm_hit}{}",
            san.as_deref().unwrap_or("?"),
            err.as_ref()
                .map(|e| format!(" ERROR {e}"))
                .unwrap_or_default()
        );
        rows.push(json!({
            "id": id, "fen": p.fen4, "c0": p.op("c0"),
            "am": am, "bm": bm,
            "move_uci": uci, "move_san": san, "score_cp": score, "depth": depth,
            "am_hit": am_hit, "bm_hit": bm_hit, "error": err,
        }));
    }
    let totals = json!({
        "positions": positions.len(),
        "am_positions": am_pos, "am_hits": am_hits,
        "bm_positions": bm_pos, "bm_hits": bm_hits,
        "errors": errors,
    });
    let out = json!({
        "engine": {"cmd": args.engine, "id_name": engine_name, "options": args.opt},
        "suite": args.suite.to_string_lossy(),
        "limits": limits,
        "results": rows,
        "totals": totals,
        "started": started,
        "finished": util::utc_iso(),
        "seconds": t0.elapsed().as_secs_f64(),
    });
    if let Some(p) = &args.out {
        std::fs::write(p, serde_json::to_string_pretty(&out).unwrap())
            .map_err(|e| format!("{}: {e}", p.display()))?;
    }
    println!(
        "suite {}: {} positions, am hits {am_hits}/{am_pos}, bm hits {bm_hits}/{bm_pos}, errors {errors}",
        args.suite.display(),
        positions.len()
    );
    Ok(())
}
