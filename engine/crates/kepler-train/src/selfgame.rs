//! In-process self-play: one game between two `Searcher`s (no UCI), the thread pool
//! used by `selfplay`, `tdleaf` and `sprt`, and PV-leaf evaluation for TDLeaf.

use crate::chess::{self, End, legal_moves};
use crate::epd;
use crate::util::XorShift;
use clap::Args;
use cozy_chess::{Board, Color, Move};
use kepler_core::params::Generation;
use kepler_core::search::{Limits, Searcher};
use kepler_core::{Dual, NGRAD, Params, Position, evaluate};
use serde::Serialize;
use serde_json::json;
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

/// Game adjudication, identical for both sides (see `Adjudicator`).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Adjudication {
    pub enabled: bool,
    pub resign_ku: f32,
    pub resign_plies: usize,
    pub draw_ku: f32,
    pub draw_plies: usize,
    pub draw_after: usize,
}

impl Default for Adjudication {
    fn default() -> Self {
        Adjudication {
            enabled: true,
            resign_ku: 8.0,
            resign_plies: 6,
            draw_ku: 0.15,
            draw_plies: 20,
            draw_after: 80,
        }
    }
}

/// Command-line flags for `Adjudication`, flattened into each self-play subcommand.
#[derive(Args, Debug, Clone)]
pub struct AdjArgs {
    /// Resign threshold (ku): the side to move scores <= -this for --resign-plies plies in a
    /// row while the previous mover scored >= +this
    #[arg(long = "resign-ku", default_value_t = 8.0)]
    pub resign_ku: f32,
    #[arg(long = "resign-plies", default_value_t = 6)]
    pub resign_plies: usize,
    /// Draw threshold (ku): |score| <= this for --draw-plies plies in a row
    #[arg(long = "draw-ku", default_value_t = 0.15)]
    pub draw_ku: f32,
    #[arg(long = "draw-plies", default_value_t = 20)]
    pub draw_plies: usize,
    /// Only plies at or after this ply count towards a draw adjudication
    #[arg(long = "draw-after", default_value_t = 80)]
    pub draw_after: usize,
    /// Play every game to a real end (or the ply cap)
    #[arg(long = "no-adjudicate")]
    pub no_adjudicate: bool,
}

impl AdjArgs {
    pub fn cfg(&self) -> Adjudication {
        Adjudication {
            enabled: !self.no_adjudicate,
            resign_ku: self.resign_ku,
            resign_plies: self.resign_plies,
            draw_ku: self.draw_ku,
            draw_plies: self.draw_plies,
            draw_after: self.draw_after,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The side to move is lost.
    MoverLoses,
    /// The side to move is winning.
    MoverWins,
    Draw,
}

/// Tracks root search scores (ku, side to move's view), one per ply, and decides when to
/// end a game early. A ply without a root search (`None`) resets both streaks.
pub struct Adjudicator {
    adj: Adjudication,
    prev: Option<f32>,
    resign_run: usize,
    draw_run: usize,
}

impl Adjudicator {
    pub fn new(adj: Adjudication) -> Self {
        Adjudicator {
            adj,
            prev: None,
            resign_run: 0,
            draw_run: 0,
        }
    }

    /// `ply` is the index of the ply about to be played, `score` its root search score.
    /// Resign: both this ply's mover and the previous ply's mover see the same side ahead
    /// by at least `resign_ku` (scores are in each mover's own view, so they have opposite
    /// signs: e.g. this mover <= -resign_ku and the previous mover >= +resign_ku), for
    /// `resign_plies` consecutive plies. Plies strictly alternate sides, so "the side to
    /// move scores <= -K and the previous mover >= +K" can only hold on every other ply;
    /// this form is its symmetric extension (the side ahead may be to move too). Draw: ply >= draw_after and |score| <= draw_ku for
    /// `draw_plies` consecutive plies. Mate scores (about +-30000) exceed any threshold.
    pub fn update(&mut self, ply: usize, score: Option<f32>) -> Option<Verdict> {
        if !self.adj.enabled {
            return None;
        }
        let prev = std::mem::replace(&mut self.prev, score);
        let Some(s) = score else {
            self.resign_run = 0;
            self.draw_run = 0;
            return None;
        };
        let k = self.adj.resign_ku;
        if s.abs() >= k && prev.is_some_and(|p| p.abs() >= k && (p > 0.0) != (s > 0.0)) {
            self.resign_run += 1;
        } else {
            self.resign_run = 0;
        }
        if ply >= self.adj.draw_after && s.abs() <= self.adj.draw_ku {
            self.draw_run += 1;
        } else {
            self.draw_run = 0;
        }
        if self.adj.resign_plies > 0 && self.resign_run >= self.adj.resign_plies {
            Some(if s < 0.0 {
                Verdict::MoverLoses
            } else {
                Verdict::MoverWins
            })
        } else if self.adj.draw_plies > 0 && self.draw_run >= self.adj.draw_plies {
            Some(Verdict::Draw)
        } else {
            None
        }
    }
}

/// FNV-1a 64-bit hash.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// All 14 constants as a JSON object (GRAD_NAMES, then `c`, `f_acc`).
pub fn constants_json(p: &Params) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    for (n, v) in Params::GRAD_NAMES.iter().zip(p.grad_values()) {
        m.insert(n.to_string(), json!(v));
    }
    m.insert("c".into(), json!(p.c));
    m.insert("f_acc".into(), json!(p.f_acc));
    m.into()
}

/// `gen_hash`: FNV-1a 64 of the compact JSON text of `constants_json(p)` (object keys in
/// serde_json's sorted order, f32 values in shortest round-trip form), as 16 lowercase
/// hex digits. Depends only on the constants, not on generation number or file layout.
pub fn gen_hash(p: &Params) -> String {
    format!("{:016x}", fnv1a64(constants_json(p).to_string().as_bytes()))
}

/// The value of a PV leaf and its exact gradient, from the root mover's point of view.
#[derive(Clone, Copy, Debug)]
pub struct Leaf {
    pub v: f32,
    pub grad: [f32; NGRAD],
    /// dv/dT, the derivative with respect to the temperature (same view as `v`).
    pub grad_t: f32,
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
    /// Measure the recorded leaf at the end of the quiescence line from the PV leaf
    /// (`Searcher::quiet_leaf`, T5) instead of at the PV leaf itself.
    pub quiet_leaf: bool,
    /// Temperature `T` (ku) in `v = tanh(eval / T)` for leaf values.
    pub temperature: f32,
    pub adj: Adjudication,
}

pub struct GameOut {
    pub moves: Vec<String>,
    pub result: End,
    pub reason: &'static str,
    /// Ended by the max-plies cap (counted as a draw).
    pub capped: bool,
    /// Ended by `Adjudicator` (resign or draw).
    pub adjudicated: bool,
    pub start_white: bool,
    pub leaves: Vec<Leaf>,
    /// Root search score (ku, the last recorded mover's view) of the last recorded ply.
    pub last_score_ku: Option<f32>,
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

/// `v = tanh(eval_white(leaf) / T)` converted to the root mover's view, with the
/// gradient from a single `Dual` evaluation: `dv/dtheta = sign * (1 - t^2) / T * deval/dtheta`
/// and `dv/dT = -sign * (1 - t^2) * eval / T^2`.
pub fn leaf_value(leaf: &Position, root_white: bool, params: &Params, temp: f32) -> Leaf {
    let sc = evaluate::<Dual>(leaf, params).score_white;
    let t = libm::tanhf(sc.v / temp);
    let sign = if root_white { 1.0 } else { -1.0 };
    let one_m = 1.0 - t * t;
    let k = sign * one_m / temp;
    let mut grad = [0.0; NGRAD];
    for (g, d) in grad.iter_mut().zip(sc.d) {
        *g = d * k;
    }
    Leaf {
        v: sign * t,
        grad,
        grad_t: -sign * one_m * sc.v / (temp * temp),
    }
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
        adjudicated: false,
        start_white,
        leaves: Vec::new(),
        last_score_ku: None,
    };
    let mut judge = Adjudicator::new(cfg.adj);
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
            // before the leaf is recorded, so leaves.len() == moves.len() at the end
            if let Some(v) = judge.update(ply, Some(res.score_ku)) {
                let mover_white = b.side_to_move() == Color::White;
                out.adjudicated = true;
                match v {
                    Verdict::Draw => {
                        out.result = End::Draw;
                        out.reason = "adjudicated_draw";
                    }
                    Verdict::MoverLoses | Verdict::MoverWins => {
                        out.result = if (v == Verdict::MoverWins) == mover_white {
                            End::WhiteWins
                        } else {
                            End::BlackWins
                        };
                        out.reason = "adjudicated_resign";
                    }
                }
                break;
            }
            if cfg.record {
                out.last_score_ku = Some(res.score_ku);
                out.leaves.push(if mate {
                    Leaf {
                        v: res.score_ku.signum(),
                        grad: [0.0; NGRAD],
                        grad_t: 0.0,
                    }
                } else {
                    let mut leaf = pv_leaf(pos, &res.pv, params[ci]);
                    if cfg.quiet_leaf {
                        leaf = engines[ci].quiet_leaf(&leaf, params[ci]);
                    }
                    leaf_value(
                        &leaf,
                        b.side_to_move() == Color::White,
                        params[ci],
                        cfg.temperature,
                    )
                });
            }
        }
        if chosen.is_none() {
            judge.update(ply, None);
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
    fn adjudication_on_synthetic_scores() {
        let adj = Adjudication::default();
        // resign: movers alternate +9 / -9 (both see the same side ahead). The first ply has
        // no predecessor, so the 6th qualifying ply is the 7th score.
        let mut j = Adjudicator::new(adj);
        for ply in 0..10 {
            assert_eq!(j.update(ply, Some(0.5)), None);
        }
        let mut fired = None;
        for i in 0..8 {
            let s = if i % 2 == 0 { -9.0 } else { 9.0 };
            if let Some(v) = j.update(10 + i, Some(s)) {
                fired = Some((i, v));
                break;
            }
        }
        // i=0 has previous 0.5 (no), i=1..=6 qualify; ply 16 (i=6) is the 6th, mover (even i) -9
        assert_eq!(fired, Some((6, Verdict::MoverLoses)));
        // the side ahead may be to move
        let mut j = Adjudicator::new(adj);
        let mut fired = None;
        for i in 0..8 {
            let s = if i % 2 == 0 { 9.0 } else { -9.0 };
            if let Some(v) = j.update(i, Some(s)) {
                fired = Some((i, v));
                break;
            }
        }
        assert_eq!(fired, Some((6, Verdict::MoverWins)));
        // a quiet ply, or one side disagreeing, resets the run
        let mut j = Adjudicator::new(adj);
        for i in 0..40 {
            let s = match i % 7 {
                6 => 0.0,
                x if x % 2 == 0 => -9.0,
                _ => 9.0,
            };
            assert_eq!(j.update(i, Some(s)), None, "ply {i}");
        }
        let mut j = Adjudicator::new(adj);
        for i in 0..40 {
            // both movers say -9: they disagree about who is ahead
            assert_eq!(j.update(i, Some(-9.0)), None);
        }
        // a ply without a search (None) resets
        let mut j = Adjudicator::new(adj);
        for i in 0..5 {
            j.update(i, Some(if i % 2 == 0 { -9.0 } else { 9.0 }));
        }
        j.update(5, None);
        assert_eq!(j.update(6, Some(-9.0)), None);
        // mate scores count as beyond the threshold
        let mut cfg = adj;
        cfg.resign_plies = 1;
        let mut j = Adjudicator::new(cfg);
        assert_eq!(j.update(0, Some(30000.0)), None);
        assert_eq!(j.update(1, Some(-30000.0)), Some(Verdict::MoverLoses));
        // draw: only after ply 80, 20 consecutive small scores
        let mut j = Adjudicator::new(adj);
        for ply in 0..80 {
            assert_eq!(j.update(ply, Some(0.0)), None);
        }
        for ply in 80..99 {
            assert_eq!(j.update(ply, Some(0.1)), None, "ply {ply}");
        }
        assert_eq!(j.update(99, Some(-0.15)), Some(Verdict::Draw));
        // one larger score resets the draw streak
        let mut j = Adjudicator::new(adj);
        for ply in 80..99 {
            j.update(ply, Some(0.0));
        }
        assert_eq!(j.update(99, Some(0.3)), None);
        assert_eq!(j.update(100, Some(0.0)), None);
        // disabled never fires
        let mut off = adj;
        off.enabled = false;
        let mut j = Adjudicator::new(off);
        for ply in 80..200 {
            assert_eq!(j.update(ply, Some(0.0)), None);
        }
    }

    #[test]
    fn gen_hash_is_stable_and_sensitive() {
        let p = Params::default();
        assert_eq!(gen_hash(&p), gen_hash(&p));
        assert_eq!(gen_hash(&p).len(), 16);
        let mut q = p;
        q.alpha -= 0.01;
        assert_ne!(gen_hash(&p), gen_hash(&q));
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
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
            quiet_leaf: true,
            temperature: 4.0,
            adj: Adjudication::default(),
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
