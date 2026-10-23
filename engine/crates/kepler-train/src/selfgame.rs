//! In-process self-play: one game between two `Searcher`s (no UCI), the thread pool
//! used by `selfplay`, `tdleaf` and `sprt`, and PV-leaf evaluation for TDLeaf.

use crate::chess::{self, End, legal_moves};
use crate::epd;
use crate::util::XorShift;
use cozy_chess::{Board, Color, Move};
use kepler_core::params::Generation;
use kepler_core::search::{Limits, Searcher};
use kepler_core::{Dual, NGRAD, Params, Position, evaluate};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// |score| above this (ku) is a forced-mate score (search mates are about +-30000).
pub const MATE_THRESHOLD: f32 = 20000.0;

#[derive(Clone, Debug)]
pub struct Opening {
    pub id: String,
    pub fen4: String,
}

pub fn load_openings(path: &Path) -> Result<Vec<Opening>, String> {
    let mut v = Vec::new();
    for (i, e) in epd::parse_file(path)?.into_iter().enumerate() {
        chess::board_from_fen4(&e.fen4)?;
        let id = e
            .op("id")
            .map(str::to_string)
            .unwrap_or_else(|| format!("line-{}", i + 1));
        v.push(Opening { id, fen4: e.fen4 });
    }
    if v.is_empty() {
        return Err(format!("{}: no openings", path.display()));
    }
    Ok(v)
}

pub fn load_generation(path: &Path) -> Result<Generation, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Generation::from_json(&s).map_err(|e| format!("{}: {e}", path.display()))
}

/// The value of a PV leaf and its exact gradient, from the root mover's point of view.
#[derive(Clone, Copy, Debug)]
pub struct Leaf {
    pub v: f32,
    pub grad: [f32; NGRAD],
}

#[derive(Clone, Debug)]
pub struct GameCfg {
    pub limits: Limits,
    /// Depth of the per-move searches that rank root moves during exploration.
    pub explore_depth: u32,
    pub max_plies: usize,
    pub explore_plies: usize,
    pub explore_margin: f32,
    /// Record the PV leaf of every ply (TDLeaf).
    pub record: bool,
    /// Tanh scale `s` (ku) for leaf values.
    pub scale: f32,
}

pub struct GameOut {
    pub moves: Vec<String>,
    pub result: End,
    pub reason: &'static str,
    /// Ended by the max-plies cap (counted as a draw).
    pub capped: bool,
    pub start_white: bool,
    pub leaves: Vec<Leaf>,
}

impl GameOut {
    /// Result for the side that made ply `t`: +1 / 0 / -1.
    pub fn result_for_mover(&self, t: usize) -> f32 {
        let white = t.is_multiple_of(2) == self.start_white;
        match (self.result, white) {
            (End::Draw, _) => 0.0,
            (End::WhiteWins, true) | (End::BlackWins, false) => 1.0,
            _ => -1.0,
        }
    }
}

pub fn result_str(e: End) -> &'static str {
    match e {
        End::WhiteWins => "1-0",
        End::BlackWins => "0-1",
        End::Draw => "1/2-1/2",
    }
}

/// Replay `pv` from `root` with `Position::play` (so accreted mass is carried) and
/// return the leaf position. Stops early at an illegal move (defensive; a PV from the
/// search is legal).
pub fn pv_leaf(root: &Position, pv: &[Move], params: &Params) -> Position {
    let mut cur = root.clone();
    for &mv in pv {
        if !cur.board.is_legal(mv) {
            break;
        }
        cur = cur.play(mv, params);
    }
    cur
}

/// `v = tanh(eval_white(leaf) / s)` converted to the root mover's view, with the
/// gradient from a single `Dual` evaluation: `dv/dtheta = sign * (1 - t^2) / s * deval/dtheta`.
pub fn leaf_value(leaf: &Position, root_white: bool, params: &Params, s: f32) -> Leaf {
    let sc = evaluate::<Dual>(leaf, params).score_white;
    let t = (sc.v / s).tanh();
    let sign = if root_white { 1.0 } else { -1.0 };
    let k = sign * (1.0 - t * t) / s;
    let mut grad = [0.0; NGRAD];
    for (g, d) in grad.iter_mut().zip(sc.d) {
        *g = d * k;
    }
    Leaf { v: sign * t, grad }
}

/// Play one game. `engines[c]` / `params[c]` play colour `c` (0 = White, 1 = Black).
/// Each side tracks its own `Position`, so differing `f_acc` stays self-consistent.
///
/// Exploration (first `explore_plies` plies after the opening): every legal root move is
/// scored by a depth-(D-1) search of the child (negated); one move is drawn uniformly from
/// those within `explore_margin` ku of the best of those scores. Forced mates are never
/// randomised.
pub fn play_game(
    cfg: &GameCfg,
    opening: &Opening,
    engines: [&mut Searcher; 2],
    params: [&Params; 2],
    seed: u64,
) -> Result<GameOut, String> {
    let mut engines = engines;
    let mut rng = XorShift::new(seed);
    let board = chess::board_from_fen4(&opening.fen4)?;
    let mut poss = [
        Position {
            board: board.clone(),
            acc: [0.0; 64],
        },
        Position {
            board: board.clone(),
            acc: [0.0; 64],
        },
    ];
    let start_white = board.side_to_move() == Color::White;
    let mut boards: Vec<Board> = vec![board];
    let mut hashes: Vec<u64> = Vec::new();
    let mut out = GameOut {
        moves: Vec::new(),
        result: End::Draw,
        reason: "",
        capped: false,
        start_white,
        leaves: Vec::new(),
    };
    let no_info = &mut |_: &kepler_core::search::SearchResult| {};
    loop {
        let b = boards.last().unwrap().clone();
        if let Some((end, why)) = chess::game_end(&b, &boards) {
            out.result = end;
            out.reason = why;
            break;
        }
        if out.moves.len() >= cfg.max_plies {
            out.result = End::Draw;
            out.reason = "max_plies";
            out.capped = true;
            break;
        }
        let ci = b.side_to_move() as usize;
        let ply = out.moves.len();
        let exploring = ply < cfg.explore_plies;
        let pos = &poss[ci];
        let mut chosen: Option<Move> = None;
        let mut mate = false;
        if cfg.record || !exploring {
            let res = engines[ci].search(pos, &hashes, params[ci], &cfg.limits, no_info);
            let best = res.best.ok_or("search returned no move")?;
            chosen = Some(best);
            mate = res.score_ku.abs() > MATE_THRESHOLD;
            if cfg.record {
                out.leaves.push(if mate {
                    Leaf {
                        v: res.score_ku.signum(),
                        grad: [0.0; NGRAD],
                    }
                } else {
                    let leaf = pv_leaf(pos, &res.pv, params[ci]);
                    leaf_value(
                        &leaf,
                        b.side_to_move() == Color::White,
                        params[ci],
                        cfg.scale,
                    )
                });
            }
        }
        let mut mv = chosen;
        if exploring && !mate {
            let moves = legal_moves(&b);
            if moves.len() > 1 {
                let mut with_root = hashes.clone();
                with_root.push(b.hash());
                let lim = Limits {
                    depth: Some(cfg.explore_depth),
                    ..Default::default()
                };
                let scored: Vec<(Move, f32)> = moves
                    .iter()
                    .map(|&m| {
                        let child = pos.play(m, params[ci]);
                        let r = engines[ci].search(&child, &with_root, params[ci], &lim, no_info);
                        (m, -r.score_ku)
                    })
                    .collect();
                let top = scored.iter().map(|x| x.1).fold(f32::MIN, f32::max);
                if top.abs() > MATE_THRESHOLD {
                    mv = scored.iter().find(|x| x.1 == top).map(|x| x.0);
                } else {
                    let cands: Vec<Move> = scored
                        .iter()
                        .filter(|x| x.1 >= top - cfg.explore_margin)
                        .map(|x| x.0)
                        .collect();
                    mv = Some(cands[rng.below(cands.len())]);
                }
            } else {
                mv = moves.first().copied();
            }
        }
        let mv = mv.ok_or("no move")?;
        out.moves.push(chess::move_to_uci(&b, mv));
        hashes.push(b.hash());
        poss[0] = poss[0].play(mv, params[0]);
        poss[1] = poss[1].play(mv, params[1]);
        boards.push(poss[0].board.clone());
    }
    Ok(out)
}

/// Run `n` jobs on `threads` workers pulling from a shared counter; each worker owns a
/// state built by `make`. Results come back in job order.
pub fn par_run<S, T: Send>(
    n: usize,
    threads: usize,
    make: impl Fn() -> S + Sync,
    work: impl Fn(&mut S, usize) -> T + Sync,
) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, T)>> = Mutex::new(Vec::with_capacity(n));
    std::thread::scope(|sc| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            sc.spawn(|| {
                let mut st = make();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = work(&mut st, i);
                    results.lock().unwrap().push((i, r));
                }
            });
        }
    });
    let mut v = results.into_inner().unwrap();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pv_leaf_is_legal_and_carries_mass() {
        let p = Params::default();
        let mut s = Searcher::new(4);
        let root = Position::startpos();
        let lim = Limits {
            depth: Some(4),
            ..Default::default()
        };
        let r = s.search(&root, &[], &p, &lim, &mut |_| {});
        assert!(!r.pv.is_empty());
        // replay checking legality at every step
        let mut cur = root.clone();
        for &mv in &r.pv {
            assert!(cur.board.is_legal(mv));
            cur = cur.play(mv, &p);
        }
        let leaf = pv_leaf(&root, &r.pv, &p);
        assert_eq!(leaf, cur);
        assert_eq!(
            leaf.board.side_to_move() == Color::White,
            r.pv.len().is_multiple_of(2)
        );
        // capture sequence: the capturing queen carries accreted mass to the leaf
        let pos = Position::from_fen("4k3/8/8/3p4/8/8/3Q4/4K3 w - - 0 1").unwrap();
        let cap: Move = "d2d5".parse().unwrap();
        let leaf = pv_leaf(&pos, &[cap], &p);
        assert!(leaf.acc[cozy_chess::Square::D5 as usize] > 0.0);
        // leaf value: sign flips with the root mover, gradient is finite
        let l_w = leaf_value(&leaf, true, &p, 4.0);
        let l_b = leaf_value(&leaf, false, &p, 4.0);
        assert!((l_w.v + l_b.v).abs() < 1e-6);
        assert!(l_w.grad.iter().all(|g| g.is_finite()));
        assert!(
            l_w.grad
                .iter()
                .zip(l_b.grad)
                .all(|(a, b)| (a + b).abs() < 1e-6)
        );
    }

    #[test]
    fn game_plays_to_a_result() {
        let p = Params::default();
        let cfg = GameCfg {
            limits: Limits {
                depth: Some(2),
                ..Default::default()
            },
            explore_depth: 1,
            max_plies: 60,
            explore_plies: 4,
            explore_margin: 0.15,
            record: true,
            scale: 4.0,
        };
        let o = Opening {
            id: "x".into(),
            fen4: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -".into(),
        };
        let (mut a, mut b) = (Searcher::new(2), Searcher::new(2));
        let g = play_game(&cfg, &o, [&mut a, &mut b], [&p, &p], 1).unwrap();
        assert!(!g.moves.is_empty() && g.moves.len() <= 60);
        assert_eq!(g.leaves.len(), g.moves.len());
        assert!(g.leaves.iter().all(|l| l.v.is_finite() && l.v.abs() <= 1.0));
    }
}
