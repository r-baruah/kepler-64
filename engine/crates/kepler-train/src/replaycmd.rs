//! `kepler-train replay`: Replay v3 JSON for the website's Watch mode
//! (`03_ENGINE_ARCHITECTURE.md` §6.3).
//!
//! Two sources: a PGN file (every game in it, or one with `--game`), or a fresh
//! self-play game of one generation (`--selfplay`). Every ply carries the full
//! `02_LAWS_SPEC.md` §10 breakdown of the position after the move, and the search
//! result from the position before it. Numbers are rounded to four decimals.

use crate::chess::{self, End, move_to_san, san_to_move};
use crate::selfgame::load_generation;
use crate::util::{self, XorShift};
use clap::Args;
use cozy_chess::{Board, Move, Piece};
use kepler_core::eval::{self, Report};
use kepler_core::params::{Params, SPEC_ID};
use kepler_core::position::Position;
use kepler_core::search::{Limits, Searcher, move_to_uci};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct ReplayArgs {
    /// Generation file (gen-XXX.json)
    #[arg(long = "gen")]
    gen_file: PathBuf,
    /// PGN file to annotate (every game, or one with --game)
    #[arg(long)]
    pgn: Option<PathBuf>,
    /// Play a fresh self-play game instead of reading a PGN
    #[arg(long)]
    selfplay: bool,
    /// Search depth for every ply
    #[arg(long, default_value_t = 4)]
    depth: u32,
    /// Self-play seed (drives the exploration plies)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Maximum plies (self-play default 60; PGN default: the whole game)
    #[arg(long)]
    plies: Option<usize>,
    /// Self-play: first plies chosen at random among moves within 0.15 ku of the best
    #[arg(long = "explore-plies", default_value_t = 6)]
    explore_plies: usize,
    /// Transposition table, MB
    #[arg(long, default_value_t = 16)]
    hash: usize,
    /// PGN: only this game (1-based)
    #[arg(long)]
    game: Option<usize>,
    /// Output file stem (default: derived from the source)
    #[arg(long)]
    name: Option<String>,
    /// Output directory
    #[arg(long)]
    out: PathBuf,
}

// ---------------------------------------------------------------- output types

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchInfo {
    /// White's point of view, in ku.
    score_ku: f32,
    pov: &'static str,
    mate_in: Option<i32>,
    depth: u32,
    nodes: u64,
    /// Kepler's best move from the position before this ply (can differ from the move played).
    best: String,
    pv: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PlyOut {
    ply: usize,
    san: String,
    uci: String,
    fen_after: String,
    breakdown: Report,
    search: SearchInfo,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct CaptureOut {
    ply: usize,
    square: String,
    captor: char,
    victim: char,
    /// Accreted mass the captor gained (`f_acc` times the victim's carried mass).
    mass_gained: f32,
}

#[derive(Serialize)]
struct Heaviest {
    square: String,
    #[serde(rename = "M")]
    m: f32,
    piece: char,
    color: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Highlights {
    /// First ply after which a king has `eta / rho >= 1`.
    first_break_ply: Option<usize>,
    first_break_color: Option<&'static str>,
    /// Ply whose move changed Kepler's white-point-of-view search score the most
    /// (scores clamped to +-50 ku; needs a following ply to compare with).
    biggest_swing_ply: Option<usize>,
    biggest_swing_ku: Option<f32>,
    /// Heaviest piece on the board after the last ply.
    heaviest: Option<Heaviest>,
    captures: Vec<CaptureOut>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GameInfo {
    headers: BTreeMap<String, String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReplayOut {
    schema: u32,
    spec: &'static str,
    generation: u32,
    game: GameInfo,
    initial_fen: String,
    initial_breakdown: Report,
    result: String,
    reason: String,
    highlights: Highlights,
    plies: Vec<PlyOut>,
}

// ------------------------------------------------------------------- rounding

fn r4(x: f32) -> f32 {
    (x * 1e4).round() / 1e4
}

fn round_report(r: &mut Report) {
    let t = &mut r.terms;
    for v in [
        &mut t.material,
        &mut t.bind,
        &mut t.space,
        &mut t.roche,
        &mut t.score_white,
    ] {
        *v = r4(*v);
    }
    for v in r.field.white.iter_mut().chain(r.field.black.iter_mut()) {
        *v = r4(*v);
    }
    for p in &mut r.pieces {
        p.rest_mass = r4(p.rest_mass);
        p.accreted = r4(p.accreted);
        p.carried_mass = r4(p.carried_mass);
        p.emission = r4(p.emission);
        p.energy = r4(p.energy);
    }
    for k in &mut r.kings {
        k.stress = r4(k.stress);
        k.binding = r4(k.binding);
        k.eta = r4(k.eta);
        k.gauge = r4(k.gauge);
        k.break_ratio = r4(k.break_ratio);
    }
}

fn piece_letter(p: Piece) -> char {
    match p {
        Piece::Pawn => 'P',
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
    }
}

// ------------------------------------------------------------------------ PGN

struct PgnGame {
    headers: BTreeMap<String, String>,
    san: Vec<String>,
    result: String,
}

fn parse_pgn(text: &str) -> Vec<PgnGame> {
    let mut games = Vec::new();
    let mut headers = BTreeMap::new();
    let mut movetext = String::new();
    let mut in_moves = false;
    let mut flush = |headers: &mut BTreeMap<String, String>, movetext: &mut String| {
        if headers.is_empty() && movetext.trim().is_empty() {
            return;
        }
        let (san, result) = tokenize_movetext(movetext);
        games.push(PgnGame {
            headers: std::mem::take(headers),
            san,
            result,
        });
        movetext.clear();
    };
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with('[') && l.ends_with(']') {
            if in_moves {
                flush(&mut headers, &mut movetext);
                in_moves = false;
            }
            if let Some((k, v)) = l[1..l.len() - 1].split_once(' ') {
                headers.insert(k.to_string(), v.trim().trim_matches('"').to_string());
            }
        } else if !l.is_empty() {
            in_moves = true;
            movetext.push_str(l);
            movetext.push('\n');
        }
    }
    flush(&mut headers, &mut movetext);
    games
}

/// SAN tokens and the result token, with comments, variations and NAGs removed.
fn tokenize_movetext(text: &str) -> (Vec<String>, String) {
    let mut clean = String::new();
    let (mut brace, mut paren) = (false, 0usize);
    let mut line_comment = false;
    for c in text.chars() {
        match c {
            '\n' if line_comment => {
                line_comment = false;
                clean.push(' ');
            }
            _ if line_comment => {}
            '{' => brace = true,
            '}' => brace = false,
            _ if brace => {}
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            _ if paren > 0 => {}
            ';' => line_comment = true,
            _ => clean.push(c),
        }
    }
    let mut san = Vec::new();
    let mut result = "*".to_string();
    for tok in clean.split_whitespace() {
        if matches!(tok, "1-0" | "0-1" | "1/2-1/2" | "*") {
            result = tok.to_string();
            continue;
        }
        if tok.starts_with('$') {
            continue;
        }
        // Strip a leading move number ("12." or "12...").
        let t = tok.trim_start_matches(|c: char| c.is_ascii_digit());
        let t = if t.len() < tok.len() && t.starts_with('.') {
            t.trim_start_matches('.')
        } else {
            tok
        };
        if !t.is_empty() {
            san.push(t.to_string());
        }
    }
    (san, result)
}

// ------------------------------------------------------------------ the game

struct Job<'a> {
    params: &'a Params,
    generation: u32,
    start: Position,
    /// SAN moves to replay (PGN), or None for self-play.
    scripted: Option<Vec<String>>,
    headers: BTreeMap<String, String>,
    given_result: Option<String>,
    depth: u32,
    seed: u64,
    explore_plies: usize,
    max_plies: usize,
    hash: usize,
}

const SWING_CLAMP: f32 = 50.0;
const EXPLORE_MARGIN: f32 = 0.15;
const MATE_THRESHOLD: f32 = 20000.0;

fn run_job(job: Job) -> Result<ReplayOut, String> {
    let params = job.params;
    let mut searcher = Searcher::new(job.hash);
    let mut rng = XorShift::new(job.seed);
    let mut pos = job.start.clone();
    let mut boards: Vec<Board> = vec![pos.board.clone()];
    let mut hashes: Vec<u64> = Vec::new();
    let mut initial_breakdown = eval::report(&pos, params, Some(job.generation));
    round_report(&mut initial_breakdown);
    let initial_fen = pos.board.to_string();
    let mut plies: Vec<PlyOut> = Vec::new();
    let mut captures = Vec::new();
    let mut white_scores: Vec<f32> = Vec::new();
    let mut ended: Option<(End, &'static str)> = chess::game_end(&pos.board, &boards);

    while ended.is_none() && plies.len() < job.max_plies {
        let idx = plies.len();
        let b = pos.board.clone();
        let limits = Limits {
            depth: Some(job.depth),
            ..Default::default()
        };
        let r = searcher.search(&pos, &hashes, params, &limits, &mut |_| {});
        let sign = if b.side_to_move() == cozy_chess::Color::White {
            1.0
        } else {
            -1.0
        };
        let best = r.best.ok_or("search returned no move")?;

        let mv: Move = match &job.scripted {
            Some(san) => {
                let Some(tok) = san.get(idx) else { break };
                san_to_move(&b, tok).map_err(|e| format!("ply {}: {e}", idx + 1))?
            }
            None if idx < job.explore_plies && r.mate_in.is_none() => explore(
                &mut searcher,
                &pos,
                &hashes,
                params,
                job.depth,
                &mut rng,
                best,
            ),
            None => best,
        };

        let san = move_to_san(&b, mv);
        let uci = move_to_uci(&b, mv);
        // Capture bookkeeping before the move is played.
        let victim = if b.piece_on(mv.from) == Some(Piece::Pawn)
            && mv.from.file() != mv.to.file()
            && b.piece_on(mv.to).is_none()
        {
            Some(Piece::Pawn)
        } else if b.color_on(mv.to) == Some(!b.side_to_move()) {
            b.piece_on(mv.to)
        } else {
            None
        };
        let captor = b.piece_on(mv.from).ok_or("move from an empty square")?;
        let acc_before = pos.acc[mv.from as usize];

        hashes.push(b.hash());
        pos = pos.play(mv, params);
        boards.push(pos.board.clone());
        if let Some(v) = victim {
            captures.push(CaptureOut {
                ply: idx + 1,
                square: mv.to.to_string(),
                captor: piece_letter(captor),
                victim: piece_letter(v),
                mass_gained: r4(pos.acc[mv.to as usize] - acc_before),
            });
        }

        let mut breakdown = eval::report(&pos, params, Some(job.generation));
        round_report(&mut breakdown);
        let ws = r.score_ku * sign;
        white_scores.push(ws);
        plies.push(PlyOut {
            ply: idx + 1,
            san,
            uci,
            fen_after: pos.board.to_string(),
            breakdown,
            search: SearchInfo {
                score_ku: r4(ws),
                pov: "white",
                mate_in: r.mate_in,
                depth: r.depth,
                nodes: r.nodes,
                best: move_to_uci(&b, best),
                pv: pv_uci(&b, &r.pv),
            },
        });
        ended = chess::game_end(&pos.board, &boards);
    }

    let (result, reason) = match ended {
        Some((End::WhiteWins, why)) => ("1-0".to_string(), why.to_string()),
        Some((End::BlackWins, why)) => ("0-1".to_string(), why.to_string()),
        Some((End::Draw, why)) => ("1/2-1/2".to_string(), why.to_string()),
        None => (
            job.given_result.clone().unwrap_or_else(|| "*".to_string()),
            if job.scripted.is_some() {
                "game_record_ends"
            } else {
                "ply_limit"
            }
            .to_string(),
        ),
    };

    // Highlights.
    let mut first_break = None;
    for p in &plies {
        for (i, k) in p.breakdown.kings.iter().enumerate() {
            if k.break_ratio >= 1.0 && first_break.is_none() {
                first_break = Some((p.ply, if i == 0 { "white" } else { "black" }));
            }
        }
        if first_break.is_some() {
            break;
        }
    }
    let mut swing: Option<(usize, f32)> = None;
    for i in 0..white_scores.len().saturating_sub(1) {
        let d = (white_scores[i + 1].clamp(-SWING_CLAMP, SWING_CLAMP)
            - white_scores[i].clamp(-SWING_CLAMP, SWING_CLAMP))
        .abs();
        if swing.is_none_or(|s| d > s.1) {
            swing = Some((i + 1, d));
        }
    }
    let heaviest = plies
        .last()
        .map(|p| &p.breakdown)
        .or(Some(&initial_breakdown))
        .and_then(|r| {
            r.pieces
                .iter()
                .filter(|p| p.piece != 'K')
                .max_by(|a, b| a.carried_mass.total_cmp(&b.carried_mass))
                .map(|p| Heaviest {
                    square: p.square.clone(),
                    m: p.carried_mass,
                    piece: p.piece,
                    color: p.color,
                })
        });

    Ok(ReplayOut {
        schema: 3,
        spec: SPEC_ID,
        generation: job.generation,
        game: GameInfo {
            headers: job.headers,
        },
        initial_fen,
        initial_breakdown,
        result,
        reason,
        highlights: Highlights {
            first_break_ply: first_break.map(|x| x.0),
            first_break_color: first_break.map(|x| x.1),
            biggest_swing_ply: swing.map(|x| x.0),
            biggest_swing_ku: swing.map(|x| r4(x.1)),
            heaviest,
            captures,
        },
        plies,
    })
}

fn pv_uci(root: &Board, pv: &[Move]) -> Vec<String> {
    let mut b = root.clone();
    let mut out = Vec::new();
    for &m in pv {
        if !b.is_legal(m) {
            break;
        }
        out.push(move_to_uci(&b, m));
        b.play_unchecked(m);
    }
    out
}

/// Pick uniformly among the root moves within `EXPLORE_MARGIN` ku of the best,
/// scored by a search of each child at depth - 1 (the same rule as `selfplay`).
fn explore(
    searcher: &mut Searcher,
    pos: &Position,
    hashes: &[u64],
    params: &Params,
    depth: u32,
    rng: &mut XorShift,
    fallback: Move,
) -> Move {
    let moves = chess::legal_moves(&pos.board);
    if moves.len() < 2 {
        return fallback;
    }
    let mut with_root = hashes.to_vec();
    with_root.push(pos.board.hash());
    let lim = Limits {
        depth: Some(depth.saturating_sub(1).max(1)),
        ..Default::default()
    };
    let scored: Vec<(Move, f32)> = moves
        .iter()
        .map(|&m| {
            let child = pos.play(m, params);
            let r = searcher.search(&child, &with_root, params, &lim, &mut |_| {});
            (m, -r.score_ku)
        })
        .collect();
    let top = scored.iter().map(|x| x.1).fold(f32::MIN, f32::max);
    if top.abs() > MATE_THRESHOLD {
        return scored.iter().find(|x| x.1 == top).map_or(fallback, |x| x.0);
    }
    let cands: Vec<Move> = scored
        .iter()
        .filter(|x| x.1 >= top - EXPLORE_MARGIN)
        .map(|x| x.0)
        .collect();
    cands[rng.below(cands.len())]
}

fn write_json(out: &Path, stem: &str, r: &ReplayOut) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let path = out.join(format!("{stem}.json"));
    let s = serde_json::to_string(r).map_err(|e| e.to_string())?;
    std::fs::write(&path, &s).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!(
        "wrote {} ({} plies, {} KB, result {} by {})",
        path.display(),
        r.plies.len(),
        s.len() / 1024,
        r.result,
        r.reason
    );
    Ok(())
}

pub fn run(a: ReplayArgs) -> Result<(), String> {
    let gen_ = load_generation(&a.gen_file)?;
    let params = gen_.constants;
    let gid = gen_.generation;
    match (&a.pgn, a.selfplay) {
        (Some(_), true) | (None, false) => {
            Err("give exactly one of --pgn <file> or --selfplay".into())
        }
        (None, true) => {
            let mut headers = BTreeMap::new();
            let who = format!("Kepler-64 gen-{gid:03} depth {}", a.depth);
            headers.insert("Event".into(), "Kepler-64 self-play".into());
            headers.insert("White".into(), who.clone());
            headers.insert("Black".into(), who);
            headers.insert("Date".into(), util::utc_date());
            headers.insert("Seed".into(), a.seed.to_string());
            let r = run_job(Job {
                params: &params,
                generation: gid,
                start: Position::startpos(),
                scripted: None,
                headers,
                given_result: None,
                depth: a.depth,
                seed: a.seed,
                explore_plies: a.explore_plies,
                max_plies: a.plies.unwrap_or(60),
                hash: a.hash,
            })?;
            let stem = a
                .name
                .unwrap_or_else(|| format!("selfplay-gen{gid:03}-d{}-s{}", a.depth, a.seed));
            write_json(&a.out, &stem, &r)
        }
        (Some(pgn), false) => {
            let text =
                std::fs::read_to_string(pgn).map_err(|e| format!("{}: {e}", pgn.display()))?;
            let games = parse_pgn(&text);
            if games.is_empty() {
                return Err(format!("{}: no games found", pgn.display()));
            }
            let base = a.name.clone().unwrap_or_else(|| {
                pgn.file_stem()
                    .map_or("game".into(), |s| s.to_string_lossy().into_owned())
            });
            let pick: Vec<(usize, &PgnGame)> = match a.game {
                Some(n) => vec![(
                    n,
                    games.get(n.wrapping_sub(1)).ok_or(format!("no game {n}"))?,
                )],
                None => games.iter().enumerate().map(|(i, g)| (i + 1, g)).collect(),
            };
            let many = pick.len() > 1;
            for (n, g) in pick {
                let start = match g.headers.get("FEN") {
                    Some(f) => Position::from_fen(f)?,
                    None => Position::startpos(),
                };
                let mut headers = g.headers.clone();
                headers.insert(
                    "Analysis".into(),
                    format!("Kepler-64 gen-{gid:03} depth {}", a.depth),
                );
                let r = run_job(Job {
                    params: &params,
                    generation: gid,
                    start,
                    scripted: Some(g.san.clone()),
                    headers,
                    given_result: Some(g.result.clone()),
                    depth: a.depth,
                    seed: a.seed,
                    explore_plies: 0,
                    max_plies: a.plies.unwrap_or(usize::MAX),
                    hash: a.hash,
                })?;
                let stem = if many {
                    format!("{base}-{n}")
                } else {
                    base.clone()
                };
                write_json(&a.out, &stem, &r)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pgn_tokenizer_handles_comments_and_numbers() {
        let (san, res) = tokenize_movetext(
            "1. e4 {best by test} e5 2. Nf3 (2. f4 exf4) Nc6 $1 3... Bb5 ; rest of line\n a6 1-0",
        );
        assert_eq!(san, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
        assert_eq!(res, "1-0");
    }

    #[test]
    fn pgn_splits_games_and_headers() {
        let g = parse_pgn(
            "[Event \"A\"]\n[White \"x y\"]\n\n1. e4 e5 *\n\n[Event \"B\"]\n\n1. d4 d5 1/2-1/2\n",
        );
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].headers["White"], "x y");
        assert_eq!(g[1].san, ["d4", "d5"]);
        assert_eq!(g[1].result, "1/2-1/2");
    }
}
