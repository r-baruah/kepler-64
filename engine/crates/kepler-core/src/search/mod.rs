//! Search (`03_ENGINE_ARCHITECTURE.md` §5): iterative deepening, PVS, quiescence,
//! transposition table, null move, LMR, check extension. Single-threaded.
//! The evaluation is only `eval_f32`; nothing here scores chess knowledge.

mod tt;

use crate::eval::eval_f32;
use crate::params::Params;
use crate::position::Position;
use cozy_chess::{Board, Color, File, Move, Piece, Rank, Square};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tt::{Bound, Tt};

pub const MATE: f32 = 30000.0;
const INF: f32 = 32000.0;
const MATE_BOUND: f32 = MATE - 1000.0;
const MAX_PLY: usize = 128;

#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime_ms: Option<u64>,
    pub wtime_ms: Option<u64>,
    pub btime_ms: Option<u64>,
    pub winc_ms: Option<u64>,
    pub binc_ms: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best: Option<Move>,
    pub score_ku: f32,
    pub mate_in: Option<i32>,
    pub depth: u32,
    pub seldepth: u32,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub elapsed_ms: u64,
}

/// Is `mv` a castling move in cozy-chess encoding (king takes own rook)?
fn is_castle(board: &Board, mv: Move) -> bool {
    board.piece_on(mv.from) == Some(Piece::King)
        && board.color_on(mv.to) == Some(board.side_to_move())
}

fn is_capture(board: &Board, mv: Move) -> bool {
    let us = board.side_to_move();
    board.color_on(mv.to) == Some(!us)
        || (board.piece_on(mv.from) == Some(Piece::Pawn)
            && mv.from.file() != mv.to.file()
            && board.piece_on(mv.to).is_none())
}

fn promo_char(p: Piece) -> char {
    match p {
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        Piece::Pawn => 'p',
        Piece::King => 'k',
    }
}

/// Move in standard UCI notation (castling as e1g1 / e1c1).
pub fn move_to_uci(board: &Board, mv: Move) -> String {
    let mut to = mv.to;
    if is_castle(board, mv) {
        let rank = Rank::First.relative_to(board.side_to_move());
        let file = if mv.to.file() > mv.from.file() {
            File::G
        } else {
            File::C
        };
        to = Square::new(file, rank);
    }
    let mut s = format!("{}{}", mv.from, to);
    if let Some(p) = mv.promotion {
        s.push(promo_char(p));
    }
    s
}

/// Parse a UCI move; accepts both e1g1 and e1h1 castling forms.
pub fn move_from_uci(board: &Board, s: &str) -> Option<Move> {
    let s = s.trim().to_ascii_lowercase();
    let mut found = None;
    board.generate_moves(|pm| {
        for mv in pm {
            let raw = match mv.promotion {
                Some(p) => format!("{}{}{}", mv.from, mv.to, promo_char(p)),
                None => format!("{}{}", mv.from, mv.to),
            };
            if raw == s || move_to_uci(board, mv) == s {
                found = Some(mv);
                return true;
            }
        }
        false
    });
    found
}

fn legal_moves(board: &Board) -> Vec<Move> {
    let mut v = Vec::with_capacity(48);
    board.generate_moves(|pm| {
        v.extend(pm);
        false
    });
    v
}

fn insufficient_material(b: &Board) -> bool {
    if !(b.pieces(Piece::Pawn) | b.pieces(Piece::Rook) | b.pieces(Piece::Queen)).is_empty() {
        return false;
    }
    (b.pieces(Piece::Knight) | b.pieces(Piece::Bishop)).len() <= 1
}

fn score_to_tt(s: f32, ply: usize) -> f32 {
    if s > MATE_BOUND {
        s + ply as f32
    } else if s < -MATE_BOUND {
        s - ply as f32
    } else {
        s
    }
}

fn score_from_tt(s: f32, ply: usize) -> f32 {
    if s > MATE_BOUND {
        s - ply as f32
    } else if s < -MATE_BOUND {
        s + ply as f32
    } else {
        s
    }
}

pub struct Searcher {
    tt: Tt,
    killers: [[Option<Move>; 2]; MAX_PLY + 2],
    history: Vec<i32>,
    stop: Arc<AtomicBool>,
    // per-search state
    stack: Vec<u64>,
    pv: Vec<Vec<Move>>,
    nodes: u64,
    seldepth: u32,
    start: Instant,
    hard_ms: Option<u64>,
    node_limit: Option<u64>,
    aborted: bool,
    can_abort: bool,
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Self {
        Searcher {
            tt: Tt::new(hash_mb),
            killers: [[None; 2]; MAX_PLY + 2],
            history: vec![0; 2 * 64 * 64],
            stop: Arc::new(AtomicBool::new(false)),
            stack: Vec::new(),
            pv: vec![Vec::new(); MAX_PLY + 2],
            nodes: 0,
            seldepth: 0,
            start: Instant::now(),
            hard_ms: None,
            node_limit: None,
            aborted: false,
            can_abort: false,
        }
    }

    /// Reallocate the transposition table with a new size.
    pub fn set_hash(&mut self, hash_mb: usize) {
        self.tt = Tt::new(hash_mb);
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.killers = [[None; 2]; MAX_PLY + 2];
        self.history.fill(0);
    }

    /// Handle that makes a running search stop when set.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// `history` = Zobrist hashes of previous game positions, oldest first, excluding root.
    pub fn search(
        &mut self,
        root: &Position,
        history: &[u64],
        params: &Params,
        limits: &Limits,
        info: &mut dyn FnMut(&SearchResult),
    ) -> SearchResult {
        self.start = Instant::now();
        self.stop.store(false, Ordering::Relaxed);
        self.tt.age = self.tt.age.wrapping_add(1);
        self.nodes = 0;
        self.seldepth = 0;
        self.aborted = false;
        self.can_abort = false;
        self.node_limit = limits.nodes;
        self.stack.clear();
        self.stack.extend_from_slice(history);
        self.stack.push(root.board.hash());
        for k in self.killers.iter_mut() {
            *k = [None; 2];
        }

        let us = root.board.side_to_move();
        let (time, inc) = match us {
            Color::White => (limits.wtime_ms, limits.winc_ms),
            Color::Black => (limits.btime_ms, limits.binc_ms),
        };
        let mut soft_ms: Option<u64> = None;
        self.hard_ms = None;
        if let Some(mt) = limits.movetime_ms {
            soft_ms = Some(mt);
            self.hard_ms = Some(mt);
        } else if time.is_some() || inc.is_some() {
            let t = time.unwrap_or(0) as f64;
            let soft = t / 30.0 + inc.unwrap_or(0) as f64 * 0.8;
            let hard = (soft * 3.0).min(t * 0.8).max(1.0);
            soft_ms = Some(soft.max(1.0) as u64);
            self.hard_ms = Some(hard as u64);
        }
        let max_depth = limits.depth.unwrap_or(64).clamp(1, 64);

        let moves = legal_moves(&root.board);
        let mut result = SearchResult {
            best: moves.first().copied(),
            score_ku: 0.0,
            mate_in: None,
            depth: 0,
            seldepth: 0,
            nodes: 0,
            pv: Vec::new(),
            elapsed_ms: 0,
        };
        if moves.is_empty() {
            result.score_ku = if root.board.checkers().is_empty() {
                0.0
            } else {
                -MATE
            };
            if !root.board.checkers().is_empty() {
                result.mate_in = Some(0);
            }
            return result;
        }
        if let Some(e) = self.tt.probe(root.board.hash())
            && let Some(m) = e.mv
            && moves.contains(&m)
        {
            result.best = Some(m);
        }

        for depth in 1..=max_depth {
            self.can_abort = depth > 1;
            self.pv[0].clear();
            let score = self.negamax(root, params, depth as i32, -INF, INF, 0, true, true);
            if self.aborted {
                break;
            }
            result.depth = depth;
            result.score_ku = score;
            result.seldepth = self.seldepth;
            result.nodes = self.nodes;
            result.elapsed_ms = self.start.elapsed().as_millis() as u64;
            if !self.pv[0].is_empty() {
                result.pv = self.pv[0].clone();
                result.best = Some(result.pv[0]);
            }
            result.mate_in = mate_in(score);
            info(&result);
            if let Some(s) = soft_ms
                && self.start.elapsed().as_millis() as u64 >= s
            {
                break;
            }
            if let Some(n) = self.node_limit
                && self.nodes >= n
            {
                break;
            }
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            // A found forced mate cannot be improved by searching deeper.
            if result.mate_in.is_some_and(|m| m > 0) && depth >= 2 {
                break;
            }
        }
        result.nodes = self.nodes;
        result.seldepth = result.seldepth.max(self.seldepth);
        result.elapsed_ms = self.start.elapsed().as_millis() as u64;
        result
    }

    #[inline]
    fn check_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if !self.can_abort {
            return false;
        }
        if let Some(n) = self.node_limit
            && self.nodes >= n
        {
            self.aborted = true;
            return true;
        }
        if self.nodes & 2047 == 0 {
            let over_time = self
                .hard_ms
                .is_some_and(|h| self.start.elapsed().as_millis() as u64 >= h);
            if over_time || self.stop.load(Ordering::Relaxed) {
                self.aborted = true;
                return true;
            }
        }
        false
    }

    fn is_repetition(&self, hm: usize) -> bool {
        let len = self.stack.len();
        let cur = self.stack[len - 1];
        let mut back = 2;
        while back <= hm && back < len {
            if self.stack[len - 1 - back] == cur {
                return true;
            }
            back += 2;
        }
        false
    }

    fn order_moves(
        &self,
        pos: &Position,
        params: &Params,
        moves: Vec<Move>,
        tt_move: Option<Move>,
        ply: usize,
        captures_only_scores: bool,
    ) -> Vec<(Move, i32)> {
        let b = &pos.board;
        let us = b.side_to_move() as usize;
        let mut out: Vec<(Move, i32)> = moves
            .into_iter()
            .map(|mv| {
                let score = if Some(mv) == tt_move {
                    2_000_000_000
                } else if is_capture(b, mv) {
                    let victim = b.piece_on(mv.to).unwrap_or(Piece::Pawn);
                    let attacker = b.piece_on(mv.from).unwrap_or(Piece::Pawn);
                    let promo = mv.promotion.map_or(0.0, |p| params.rest_mass(p));
                    1_000_000_000
                        + ((params.rest_mass(victim) + promo) * 1000.0 - params.rest_mass(attacker))
                            as i32
                } else if let Some(p) = mv.promotion {
                    900_000_000 + (params.rest_mass(p) * 1000.0) as i32
                } else if !captures_only_scores && self.killers[ply][0] == Some(mv) {
                    800_000_000
                } else if !captures_only_scores && self.killers[ply][1] == Some(mv) {
                    790_000_000
                } else {
                    self.history[(us * 64 + mv.from as usize) * 64 + mv.to as usize]
                };
                (mv, score)
            })
            .collect();
        out.sort_by_key(|&(_, s)| std::cmp::Reverse(s));
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &Position,
        params: &Params,
        mut depth: i32,
        mut alpha: f32,
        mut beta: f32,
        ply: usize,
        allow_null: bool,
        is_pv: bool,
    ) -> f32 {
        let board = &pos.board;
        let in_check = !board.checkers().is_empty();
        if ply >= MAX_PLY {
            return eval_f32(pos, params);
        }
        if in_check {
            depth += 1;
        }
        if is_pv {
            self.pv[ply].clear();
        }
        if depth <= 0 {
            return self.qsearch(pos, params, alpha, beta, ply);
        }
        self.nodes += 1;
        if self.check_abort() {
            return 0.0;
        }
        self.seldepth = self.seldepth.max(ply as u32);
        let hm = board.halfmove_clock() as usize;
        if ply > 0 {
            if self.is_repetition(hm) || insufficient_material(board) {
                return 0.0;
            }
            // Mate distance pruning.
            alpha = alpha.max(-MATE + ply as f32);
            beta = beta.min(MATE - ply as f32 - 1.0);
            if alpha >= beta {
                return alpha;
            }
        }

        let key = board.hash();
        let mut tt_move = None;
        if let Some(e) = self.tt.probe(key) {
            tt_move = e.mv;
            if ply > 0 && !is_pv && e.depth as i32 >= depth {
                let s = score_from_tt(e.score, ply);
                match e.bound {
                    Bound::Exact => return s,
                    Bound::Lower if s >= beta => return s,
                    Bound::Upper if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let moves = legal_moves(board);
        if moves.is_empty() {
            return if in_check { -MATE + ply as f32 } else { 0.0 };
        }
        if hm >= 100 {
            return 0.0;
        }

        // Null move: R = 2, not in check, not in pawn-only positions.
        if allow_null && !is_pv && !in_check && depth >= 3 {
            let us = board.side_to_move();
            let has_pieces = !(board.colors(us)
                & !(board.pieces(Piece::Pawn) | board.pieces(Piece::King)))
            .is_empty();
            if has_pieces
                && eval_f32(pos, params) >= beta
                && let Some(nb) = board.null_move()
            {
                let child = Position {
                    board: nb,
                    acc: pos.acc,
                };
                self.stack.push(child.board.hash());
                let s = -self.negamax(
                    &child,
                    params,
                    depth - 1 - 2,
                    -beta,
                    -beta + 1.0,
                    ply + 1,
                    false,
                    false,
                );
                self.stack.pop();
                if self.aborted {
                    return 0.0;
                }
                if s >= beta {
                    return if s > MATE_BOUND { beta } else { s };
                }
            }
        }

        let ordered = self.order_moves(pos, params, moves, tt_move, ply, false);
        let orig_alpha = alpha;
        let mut best = -INF;
        let mut best_move = None;
        let us = board.side_to_move() as usize;

        for (i, &(mv, _)) in ordered.iter().enumerate() {
            let quiet = !is_capture(board, mv) && mv.promotion.is_none();
            let child = pos.play(mv, params);
            self.stack.push(child.board.hash());
            let gives_check = !child.board.checkers().is_empty();
            let mut score;
            if i == 0 {
                score = -self.negamax(
                    &child,
                    params,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    true,
                    is_pv,
                );
            } else {
                let mut reduction = 0;
                if depth >= 3 && ply > 0 && i >= 4 && quiet && !in_check && !gives_check {
                    reduction = 1 + (i >= 10) as i32 + (depth >= 6) as i32;
                }
                score = -self.negamax(
                    &child,
                    params,
                    depth - 1 - reduction,
                    -alpha - 1.0,
                    -alpha,
                    ply + 1,
                    true,
                    false,
                );
                if score > alpha && reduction > 0 {
                    score = -self.negamax(
                        &child,
                        params,
                        depth - 1,
                        -alpha - 1.0,
                        -alpha,
                        ply + 1,
                        true,
                        false,
                    );
                }
                if score > alpha && score < beta && is_pv {
                    score = -self.negamax(
                        &child,
                        params,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        true,
                        true,
                    );
                }
            }
            self.stack.pop();
            if self.aborted {
                return 0.0;
            }
            if score > best {
                best = score;
                best_move = Some(mv);
                if score > alpha {
                    alpha = score;
                    if is_pv {
                        let mut line = Vec::with_capacity(self.pv[ply + 1].len() + 1);
                        line.push(mv);
                        line.extend_from_slice(&self.pv[ply + 1]);
                        self.pv[ply] = line;
                    }
                    if alpha >= beta {
                        if quiet {
                            if self.killers[ply][0] != Some(mv) {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = Some(mv);
                            }
                            let idx = (us * 64 + mv.from as usize) * 64 + mv.to as usize;
                            let h = &mut self.history[idx];
                            *h = (*h + depth * depth).min(500_000_000 / 4);
                        }
                        break;
                    }
                }
            }
            // Quiet moves that did not cut are not penalised (plain history).
        }

        let bound = if best >= beta {
            Bound::Lower
        } else if best > orig_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt
            .store(key, depth as i16, score_to_tt(best, ply), bound, best_move);
        best
    }

    fn qsearch(
        &mut self,
        pos: &Position,
        params: &Params,
        mut alpha: f32,
        beta: f32,
        ply: usize,
    ) -> f32 {
        self.nodes += 1;
        if self.check_abort() {
            return 0.0;
        }
        self.seldepth = self.seldepth.max(ply as u32);
        let board = &pos.board;
        if insufficient_material(board) {
            return 0.0;
        }
        let in_check = !board.checkers().is_empty();
        if ply >= MAX_PLY {
            return eval_f32(pos, params);
        }
        let moves = legal_moves(board);
        if moves.is_empty() {
            return if in_check { -MATE + ply as f32 } else { 0.0 };
        }
        if board.halfmove_clock() >= 100 {
            return 0.0;
        }
        let stand = if in_check {
            -INF
        } else {
            eval_f32(pos, params)
        };
        if !in_check {
            if stand >= beta {
                return stand;
            }
            alpha = alpha.max(stand);
        }
        let candidates: Vec<Move> = if in_check {
            moves
        } else {
            moves
                .into_iter()
                .filter(|&m| is_capture(board, m) || m.promotion.is_some())
                .collect()
        };
        let ordered = self.order_moves(pos, params, candidates, None, ply, true);
        let mut best = stand;
        for &(mv, _) in &ordered {
            if !in_check {
                // Delta pruning with rest masses: even winning the victim (and a
                // promotion) plus a pawn of slack cannot reach alpha.
                let victim = if is_capture(board, mv) {
                    board.piece_on(mv.to).unwrap_or(Piece::Pawn)
                } else {
                    Piece::King
                };
                let promo = mv.promotion.map_or(0.0, |p| params.rest_mass(p) - 1.0);
                if stand + params.rest_mass(victim) + promo + 1.0 < alpha {
                    continue;
                }
            }
            let child = pos.play(mv, params);
            let score = -self.qsearch(&child, params, -beta, -alpha, ply + 1);
            if self.aborted {
                return 0.0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }
}

fn mate_in(score: f32) -> Option<i32> {
    if score > MATE_BOUND {
        Some(((MATE - score + 1.0) / 2.0) as i32)
    } else if score < -MATE_BOUND {
        Some(-(((MATE + score) / 2.0) as i32))
    } else {
        None
    }
}
