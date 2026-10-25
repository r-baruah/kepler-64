//! `kepler-train match`: play games between two UCI engines.

use crate::chess::{self, End};
use crate::epd;
use crate::stats;
use crate::uci::{Engine, EngineSpec};
use crate::util;
use clap::Args;
use cozy_chess::{Board, Color};
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Args, Debug)]
pub struct MatchArgs {
    /// Engine A command (split on whitespace)
    #[arg(long)]
    a: String,
    /// Engine B command (split on whitespace)
    #[arg(long)]
    b: String,
    #[arg(long = "a-arg", allow_hyphen_values = true)]
    a_arg: Vec<String>,
    #[arg(long = "b-arg", allow_hyphen_values = true)]
    b_arg: Vec<String>,
    #[arg(long = "a-dir")]
    a_dir: Option<PathBuf>,
    #[arg(long = "b-dir")]
    b_dir: Option<PathBuf>,
    #[arg(long = "a-name", default_value = "A")]
    a_name: String,
    #[arg(long = "b-name", default_value = "B")]
    b_name: String,
    /// setoption Name=Value for A (repeatable)
    #[arg(long = "a-opt")]
    a_opt: Vec<String>,
    #[arg(long = "b-opt")]
    b_opt: Vec<String>,
    /// EPD/FEN opening file; each opening is played twice with colours swapped
    #[arg(long)]
    openings: PathBuf,
    /// Total games (default 2 x openings)
    #[arg(long)]
    games: Option<usize>,
    #[arg(long)]
    depth: Option<u32>,
    #[arg(long)]
    nodes: Option<u64>,
    /// Milliseconds per move
    #[arg(long)]
    movetime: Option<u64>,
    /// base+inc in seconds, e.g. 10+0.1
    #[arg(long)]
    tc: Option<String>,
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    #[arg(long = "max-plies", default_value_t = 400)]
    max_plies: usize,
    /// Stockfish used to adjudicate (depth 12, |eval| > 1000cp for 8 plies)
    #[arg(long = "adjudicate-sf")]
    adjudicate_sf: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    pgn: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
enum Limit {
    Depth(u32),
    Nodes(u64),
    MoveTime(u64),
    Tc { base_ms: u64, inc_ms: u64 },
}

impl Limit {
    fn to_json(self) -> serde_json::Value {
        match self {
            Limit::Depth(d) => json!({"type": "depth", "depth": d}),
            Limit::Nodes(n) => json!({"type": "nodes", "nodes": n}),
            Limit::MoveTime(m) => json!({"type": "movetime", "ms": m}),
            Limit::Tc { base_ms, inc_ms } => {
                json!({"type": "tc", "base_ms": base_ms, "inc_ms": inc_ms})
            }
        }
    }
}

fn parse_tc(s: &str) -> Result<Limit, String> {
    let (b, i) = s.split_once('+').unwrap_or((s, "0"));
    let b: f64 = b.parse().map_err(|_| format!("bad --tc '{s}'"))?;
    let i: f64 = i.parse().map_err(|_| format!("bad --tc '{s}'"))?;
    Ok(Limit::Tc {
        base_ms: (b * 1000.0).round() as u64,
        inc_ms: (i * 1000.0).round() as u64,
    })
}

#[derive(Serialize, Clone, Debug)]
pub struct GameRecord {
    pub opening: String,
    pub white: String,
    pub black: String,
    pub result: String,
    pub reason: String,
    pub plies: usize,
    pub moves_uci: Vec<String>,
}

fn result_str(e: End) -> &'static str {
    match e {
        End::WhiteWins => "1-0",
        End::BlackWins => "0-1",
        End::Draw => "1/2-1/2",
    }
}

struct Cfg {
    limit: Limit,
    max_plies: usize,
    sf: Option<PathBuf>,
}

fn position_cmd(fen: &str, moves: &[String]) -> String {
    if moves.is_empty() {
        format!("position fen {fen}")
    } else {
        format!("position fen {fen} moves {}", moves.join(" "))
    }
}

fn fail(loser: Color, reason: String) -> (End, String) {
    let e = if loser == Color::White {
        End::BlackWins
    } else {
        End::WhiteWins
    };
    (e, reason)
}

#[allow(clippy::too_many_arguments)]
fn play_game(
    cfg: &Cfg,
    fen4: &str,
    white_spec: &EngineSpec,
    black_spec: &EngineSpec,
    white_name: &str,
    black_name: &str,
) -> GameRecord {
    let mut moves: Vec<String> = Vec::new();
    let (end, reason) = run_game(cfg, fen4, white_spec, black_spec, &mut moves);
    GameRecord {
        opening: fen4.to_string(),
        white: white_name.to_string(),
        black: black_name.to_string(),
        result: result_str(end).to_string(),
        reason,
        plies: moves.len(),
        moves_uci: moves,
    }
}

#[allow(clippy::collapsible_if)]
fn run_game(
    cfg: &Cfg,
    fen4: &str,
    white_spec: &EngineSpec,
    black_spec: &EngineSpec,
    moves: &mut Vec<String>,
) -> (End, String) {
    let start_fen = chess::fen_with_counters(fen4);
    let mut board = match Board::from_fen(&start_fen, false) {
        Ok(b) => b,
        Err(e) => return (End::Draw, format!("bad_opening: {e:?}")),
    };
    let mut white = match Engine::start(white_spec) {
        Ok(e) => e,
        Err(m) => return (End::BlackWins, format!("engine_error(white): {m}")),
    };
    let mut black = match Engine::start(black_spec) {
        Ok(e) => e,
        Err(m) => return (End::WhiteWins, format!("engine_error(black): {m}")),
    };
    if let Err(m) = white.new_game() {
        return (End::BlackWins, format!("engine_error(white): {m}"));
    }
    if let Err(m) = black.new_game() {
        return (End::WhiteWins, format!("engine_error(black): {m}"));
    }
    let mut sf = match &cfg.sf {
        Some(p) => match Engine::start(&EngineSpec::plain(&p.to_string_lossy())) {
            Ok(mut e) => {
                let _ = e.new_game();
                Some(e)
            }
            Err(m) => {
                eprintln!("warning: adjudication engine failed to start: {m}");
                None
            }
        },
        None => None,
    };
    let (mut wtime, mut btime) = match cfg.limit {
        Limit::Tc { base_ms, .. } => (base_ms as i64, base_ms as i64),
        _ => (0, 0),
    };
    let mut history = vec![board.clone()];
    let mut streak = 0u32;
    let mut streak_sign = 0i32;
    loop {
        if let Some((e, r)) = chess::game_end(&board, &history) {
            return (e, r.to_string());
        }
        if moves.len() >= cfg.max_plies {
            return (End::Draw, "max_plies".to_string());
        }
        let stm = board.side_to_move();
        let (eng, name) = if stm == Color::White {
            (&mut white, "white")
        } else {
            (&mut black, "black")
        };
        let (go_args, timeout) = match cfg.limit {
            Limit::Depth(d) => (format!("depth {d}"), Duration::from_secs(900)),
            Limit::Nodes(n) => (format!("nodes {n}"), Duration::from_secs(900)),
            Limit::MoveTime(ms) => (format!("movetime {ms}"), Duration::from_millis(ms + 10_000)),
            Limit::Tc { inc_ms, .. } => {
                let mine = if stm == Color::White { wtime } else { btime };
                (
                    format!("wtime {wtime} btime {btime} winc {inc_ms} binc {inc_ms}"),
                    Duration::from_millis(mine.max(0) as u64 + 2_000),
                )
            }
        };
        let t0 = Instant::now();
        let res = eng.go(&position_cmd(&start_fen, moves), &go_args, timeout);
        let elapsed = t0.elapsed().as_millis() as i64;
        let r = match res {
            Ok(r) => r,
            Err(m) => {
                eprintln!("engine error ({name}) at ply {}: {m}", moves.len());
                let reason = if m == "timeout" {
                    "time_forfeit".to_string()
                } else {
                    format!("engine_error: {m}")
                };
                return fail(stm, reason);
            }
        };
        if let Limit::Tc { inc_ms, .. } = cfg.limit {
            let clock = if stm == Color::White {
                &mut wtime
            } else {
                &mut btime
            };
            *clock -= elapsed;
            if *clock < 0 {
                return fail(stm, "time_forfeit".to_string());
            }
            *clock += inc_ms as i64;
        }
        let mv = match chess::uci_to_move(&board, &r.bestmove) {
            Ok(m) => m,
            Err(m) => {
                eprintln!(
                    "ILLEGAL MOVE by {name} at ply {} in '{}': {m} (bestmove '{}')",
                    moves.len(),
                    position_cmd(&start_fen, moves),
                    r.bestmove
                );
                return fail(stm, format!("illegal_move: {}", r.bestmove));
            }
        };
        moves.push(r.bestmove.clone());
        board.play(mv);
        history.push(board.clone());
        if chess::game_end(&board, &history).is_some() {
            continue;
        }
        if let Some(sfe) = sf.as_mut() {
            if let Ok(g) = sfe.go(
                &position_cmd(&start_fen, moves),
                "depth 12",
                Duration::from_secs(120),
            ) {
                if let Some(sc) = g.score {
                    let cp = sc.to_cp();
                    let white_cp = if board.side_to_move() == Color::White {
                        cp
                    } else {
                        -cp
                    };
                    let sign = white_cp.signum();
                    if white_cp.abs() > 1000 {
                        if sign == streak_sign {
                            streak += 1;
                        } else {
                            streak = 1;
                            streak_sign = sign;
                        }
                    } else {
                        streak = 0;
                        streak_sign = 0;
                    }
                    if streak >= 8 {
                        let e = if streak_sign > 0 {
                            End::WhiteWins
                        } else {
                            End::BlackWins
                        };
                        return (e, "adjudicated_sf".to_string());
                    }
                }
            }
        }
    }
}

fn pgn_text(g: &GameRecord, round: usize) -> String {
    let mut out = String::new();
    out.push_str("[Event \"kepler-train match\"]\n");
    out.push_str(&format!("[Round \"{round}\"]\n"));
    out.push_str(&format!(
        "[White \"{}\"]\n[Black \"{}\"]\n",
        g.white, g.black
    ));
    out.push_str(&format!("[Result \"{}\"]\n", g.result));
    out.push_str(&format!("[Termination \"{}\"]\n", g.reason));
    out.push_str(&format!(
        "[SetUp \"1\"]\n[FEN \"{}\"]\n\n",
        chess::fen_with_counters(&g.opening)
    ));
    let mut board = match chess::board_from_fen4(&g.opening) {
        Ok(b) => b,
        Err(_) => return out,
    };
    let mut text = String::new();
    for (i, m) in g.moves_uci.iter().enumerate() {
        let Ok(mv) = chess::uci_to_move(&board, m) else {
            break;
        };
        if board.side_to_move() == Color::White {
            text.push_str(&format!("{}. ", board.fullmove_number()));
        } else if i == 0 {
            text.push_str(&format!("{}... ", board.fullmove_number()));
        }
        text.push_str(&chess::move_to_san(&board, mv));
        text.push(' ');
        board.play(mv);
    }
    text.push_str(&g.result);
    // wrap at ~80 columns
    let mut line = String::new();
    for w in text.split(' ') {
        if line.len() + w.len() + 1 > 80 {
            out.push_str(line.trim_end());
            out.push('\n');
            line.clear();
        }
        line.push_str(w);
        line.push(' ');
    }
    out.push_str(line.trim_end());
    out.push_str("\n\n");
    out
}

pub fn run(args: MatchArgs) -> Result<(), String> {
    let limit = match (args.depth, args.nodes, args.movetime, &args.tc) {
        (Some(d), None, None, None) => Limit::Depth(d),
        (None, Some(n), None, None) => Limit::Nodes(n),
        (None, None, Some(m), None) => Limit::MoveTime(m),
        (None, None, None, Some(t)) => parse_tc(t)?,
        _ => return Err("give exactly one of --depth, --nodes, --movetime, --tc".into()),
    };
    let a_spec = EngineSpec::new(&args.a, &args.a_arg, args.a_dir.as_deref(), &args.a_opt)?;
    let b_spec = EngineSpec::new(&args.b, &args.b_arg, args.b_dir.as_deref(), &args.b_opt)?;
    let openings = epd::parse_file(&args.openings)?;
    if openings.is_empty() {
        return Err("no openings".into());
    }
    for o in &openings {
        chess::board_from_fen4(&o.fen4)?;
    }
    let n_games = args.games.unwrap_or(openings.len() * 2);
    let cfg = Cfg {
        limit,
        max_plies: args.max_plies,
        sf: args.adjudicate_sf.clone(),
    };
    let started = util::utc_iso();
    let t0 = Instant::now();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<GameRecord>>> = Mutex::new(vec![None; n_games]);
    let done = AtomicUsize::new(0);
    let workers = args.concurrency.max(1).min(n_games.max(1));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n_games {
                        break;
                    }
                    // game i: pair i/2, A is white in even games
                    let a_white = i.is_multiple_of(2);
                    let fen4 = &openings[(i / 2) % openings.len()].fen4;
                    let rec = if a_white {
                        play_game(&cfg, fen4, &a_spec, &b_spec, &args.a_name, &args.b_name)
                    } else {
                        play_game(&cfg, fen4, &b_spec, &a_spec, &args.b_name, &args.a_name)
                    };
                    let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                    eprintln!(
                        "[{n}/{n_games}] game {} {} vs {}: {} ({}, {} plies)",
                        i + 1,
                        rec.white,
                        rec.black,
                        rec.result,
                        rec.reason,
                        rec.plies
                    );
                    results.lock().unwrap()[i] = Some(rec);
                }
            });
        }
    });
    let games: Vec<GameRecord> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let finished = util::utc_iso();

    // Score from A's point of view.
    let a_points = |idx: usize, g: &GameRecord| -> f64 {
        let a_white = idx.is_multiple_of(2);
        match (g.result.as_str(), a_white) {
            ("1-0", true) | ("0-1", false) => 1.0,
            ("1/2-1/2", _) => 0.5,
            _ => 0.0,
        }
    };
    let (mut w, mut d, mut l) = (0u32, 0u32, 0u32);
    let mut pts = Vec::new();
    for (i, g) in games.iter().enumerate() {
        let p = a_points(i, g);
        match p {
            1.0 => w += 1,
            0.5 => d += 1,
            _ => l += 1,
        }
        pts.push(p);
    }
    #[allow(clippy::chunks_exact_to_as_chunks)]
    let pair_pts: Vec<f64> = pts.chunks_exact(2).map(|c| c[0] + c[1]).collect();
    let penta = stats::pentanomial(&pair_pts);
    let elo = stats::elo(w, d, l);
    let mut reasons = std::collections::BTreeMap::new();
    for g in &games {
        *reasons
            .entry(g.reason.split(':').next().unwrap_or("").to_string())
            .or_insert(0u32) += 1;
    }
    let out = json!({
        "a": {"name": args.a_name, "cmd": args.a, "args": args.a_arg, "options": args.a_opt},
        "b": {"name": args.b_name, "cmd": args.b, "args": args.b_arg, "options": args.b_opt},
        "limits": limit.to_json(),
        "max_plies": args.max_plies,
        "adjudicate_sf": args.adjudicate_sf.as_ref().map(|p| p.to_string_lossy().to_string()),
        "games": games,
        "score": {"a_w": w, "a_d": d, "a_l": l},
        "reasons": reasons,
        "pentanomial": penta,
        "elo": elo,
        "started": started,
        "finished": finished,
    });
    if let Some(p) = &args.out {
        std::fs::write(p, serde_json::to_string_pretty(&out).unwrap())
            .map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if let Some(p) = &args.pgn {
        let text: String = games
            .iter()
            .enumerate()
            .map(|(i, g)| pgn_text(g, i + 1))
            .collect();
        std::fs::write(p, text).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    println!(
        "{} vs {}: +{w} ={d} -{l} ({} games, {:.1}%) elo {:+.1} [{:+.1}, {:+.1}]{} penta {:?} in {:.1}s",
        args.a_name,
        args.b_name,
        games.len(),
        elo.score * 100.0,
        elo.estimate,
        elo.lo95,
        elo.hi95,
        if elo.reliable {
            ""
        } else {
            " (<200 games, unreliable)"
        },
        penta,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tc_parse() {
        match parse_tc("10+0.1").unwrap() {
            Limit::Tc { base_ms, inc_ms } => assert_eq!((base_ms, inc_ms), (10_000, 100)),
            _ => panic!(),
        }
        assert!(parse_tc("x+1").is_err());
    }

    #[test]
    fn pgn_has_castling_san() {
        let g = GameRecord {
            opening: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -".into(),
            white: "A".into(),
            black: "B".into(),
            result: "1/2-1/2".into(),
            reason: "x".into(),
            plies: 7,
            moves_uci: ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };
        let p = pgn_text(&g, 1);
        assert!(
            p.contains("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O 1/2-1/2"),
            "{p}"
        );
    }
}
