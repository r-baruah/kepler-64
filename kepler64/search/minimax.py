"""Alpha-beta search over the board, using the pure-JAX-friendly FastBoard.

chess:  negamax/alpha-beta with Principal-Variation Search, a Zobrist-style
        transposition table, iterative deepening, null-move pruning, killer
        moves, and a history heuristic. The evaluation is the Roche Engine.
physics: candidate batches are statically padded to the theoretical maximum
         (218) and evaluated with one consistent vmap shape. Captures accrete
         mass (2C) before evaluation. The transposition key is a Zobrist hash
         of the position plus a compact checksum of the accreted mass state,
         so cached scores are never reused across different field states.

Search-horizon notes (why this file looks the way it is):
  * Every depth is reachable. The old null-move probe ran with a degenerate
    (-INF, -INF+1) window at the root and returned +INF, which silently broke
    best_move for depth >= 4. Null-move is now gated on a finite beta.
  * Iterative deepening with time budgeting lets a fixed wall-clock budget
    reach depth 5-7 instead of the previous depth 2-3, and the PV move from
    the previous iteration seeds root ordering.
  * Quiescence resolves captures AND (when in check) evasions, so hanging
    pieces and king escapes are not read at a volatile horizon.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..core.fastboard import FastBoard, _MASS_LUT
from ..core.constants import Constants
from ..core.gravity import _DIST2, _DIST
from ..core.evaluate import score_white, batch_score, multiverse_score_white
from ..core.transitions import child_mass_vector

INF = float("inf")
MATE = 100000.0
_MAX_MOVES = 218

# ── static geometry for cheap numpy move ordering (no JAX device round-trip) ─
_DIST_NP = np.asarray(_DIST)
_DIST2_NP = np.asarray(_DIST2)

# Transposition-table flags.
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

_TT_SIZE = 1 << 19  # 524k slots (~10 MB)

# Zobrist randoms for the position key (fixed seed: hashes are only compared
# inside one process, determinism is more valuable than randomness here).
# RandomState.randint is int32-limited, so two 31-bit draws are combined to a
# full 64-bit word.
_ZRNG = np.random.RandomState(20260817)


def _rand64(n):
    lo = _ZRNG.randint(1, 2**31 - 1, size=n).astype(np.uint64)
    hi = _ZRNG.randint(1, 2**31 - 1, size=n).astype(np.uint64)
    return (hi << 32) | lo


_ZOB = (_rand64(64 * 13)).reshape(64, 13)
_ZOB[:, 6] = 0  # empty square contributes nothing
_ZOB_TURN = np.array([0, 0x9E3779B97F4A7C15], dtype=np.uint64)
_ZOB_CASTLE = _rand64(16)
_ZOB_EP = _rand64(65)
# Per-square random multipliers for the mass-state checksum.
_MASS_RAND = _rand64(64)


def _mass_checksum(m) -> int:
    """Exact 64-bit checksum of a (64,) float32 mass vector.

    Any change in any float32 bit (accretion boost, Lorentz factor, moved mass)
    changes the checksum. O(64) numpy ops — the old blake2b-over-1.5KB hash
    cost ~50x more per TT probe and kept the table effectively unreachable.
    """
    bits = np.asarray(m, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return int((bits * _MASS_RAND).sum()) & 0xFFFFFFFFFFFFFFFF


class TT:
    """Open-addressing transposition table.

    Keyed by a Zobrist hash of the position (pieces, turn, castling, ep) XOR a
    checksum of the child AND parent mass vectors: two lines that reach the
    same pieces with different accreted masses (or different parent fields,
    which the move-sensitivity terms depend on) are DIFFERENT physical states
    and must not share a cached score.

    Slot replacement: keep the entry with the larger depth (ties keep the old
    entry). No verification beyond the stored 64-bit hash (1/2^64 collision
    odds are far below the engine's other failure modes).
    """

    def __init__(self, size: int = _TT_SIZE):
        self.size = size
        self.mask = size - 1
        self.keys = np.zeros(size, dtype=np.uint64)
        self.depth = np.zeros(size, dtype=np.int8)
        self.flag = np.zeros(size, dtype=np.int8)
        self.score = np.zeros(size, dtype=np.float32)
        self.move = np.full(size, -1, dtype=np.int32)
        self.occupied = np.zeros(size, dtype=np.bool_)

    @staticmethod
    def _hash(board, child_m, parent_m) -> int:
        pieces = board.pieces
        sq = np.arange(64, dtype=np.int64)
        idx = pieces.astype(np.int64) + 6  # 0=empty, 1..12 pieces
        h = int(np.bitwise_xor.reduce(np.where(idx != 6, _ZOB[sq, idx], 0)))
        h ^= int(_ZOB_CASTLE[int(board.castling) & 0xF])
        h ^= int(_ZOB_EP[int(board.ep) + 1])  # ep in [-1,63] -> [0,64]
        h ^= int(_ZOB_TURN[int(board.turn)])
        h = (h ^ _mass_checksum(child_m) ^ _mass_checksum(parent_m)) & 0xFFFFFFFFFFFFFFFF
        return h

    def lookup(self, board, child_m, parent_m):
        h = self._hash(board, child_m, parent_m)
        slot = h & self.mask
        k = self.keys[slot]
        if self.occupied[slot] and k == h:
            return (int(self.depth[slot]), int(self.flag[slot]),
                    float(self.score[slot]), int(self.move[slot]))
        return None

    def store(self, board, child_m, parent_m, depth: int, flag: int,
              score: float, move):
        h = self._hash(board, child_m, parent_m)
        slot = h & self.mask
        if self.occupied[slot] and self.depth[slot] > depth and self.keys[slot] != h:
            return  # keep deeper entry
        self.occupied[slot] = True
        self.keys[slot] = h
        self.depth[slot] = np.int8(min(depth, 127))
        self.flag[slot] = np.int8(flag)
        self.score[slot] = np.float32(score)
        if move is None:
            packed = -1
        else:
            f, t, promo = move
            packed = (int(f) << 16) | (int(t) << 8) | int(promo or 0)
        self.move[slot] = np.int32(packed)


def _score_position(masses, constants: Constants, turn: int,
                     use_multiverse: bool = False, key=None, K: int = 8,
                     sigma: float = 0.1, parent=None) -> float:
    if use_multiverse and key is not None:
        s = float(multiverse_score_white(masses, constants, key, K=K, sigma=sigma,
                                         parent=parent))
    else:
        s = float(score_white(masses, constants, parent=parent))
    return s if turn == 0 else -s


def _terminal(board: FastBoard) -> float:
    if board.is_checkmate():
        return -MATE  # side to move is mated
    return 0.0


def _mass_capture_value(board: FastBoard, m) -> float:
    """MVV-LVA-ish capture value from raw masses (cheap, no device traffic)."""
    f, t, _ = m
    moving = float(_MASS_LUT[abs(int(board.pieces[f]))])
    captured_piece = int(board.pieces[t])
    captured = float(_MASS_LUT[abs(captured_piece)])
    if captured_piece == 0 and abs(int(board.pieces[f])) == 1 and t == board.ep:
        captured = 1.0  # en-passant
    return 10.0 * captured - moving / 10.0


def _ordered_moves(board: FastBoard, constants: Constants,
                   tt_move=None, killers=None, history=None, ply: int = 0,
                   root_pv=None):
    """Legal moves ordered for alpha-beta efficiency.

    Order (descending priority): TT move, MVV-LVA captures, previous-iteration
    PV move (at the root), killer moves, then history heuristic. The
    gravitational potential well is only a STABLE tiebreak at zero score — the
    previous implementation re-sorted the WHOLE ordered list by the well,
    which destroyed the TT/MVV-LVA ordering and killed alpha-beta efficiency.
    """
    moves = board.legal_moves()
    if len(moves) <= 1:
        return moves

    U = None  # potential well, computed lazily for the tiebreak only
    scored = []
    for m in moves:
        sc = 0.0
        if tt_move is not None and m == tt_move:
            sc += 1_000_000.0
        elif board.is_capture(m):
            sc += 100_000.0 + _mass_capture_value(board, m)
        if root_pv and m == root_pv:
            sc += 50_000.0
        if killers:
            if m in killers[ply]:
                sc += 20_000.0
        if history is not None:
            sc += float(history[m[0] * 64 + m[1]])
        if sc == 0.0:
            # Quiet, unhinted move: the physics tiebreak decides its slot among
            # the other unhinted quiets (moving into a deep friendly well =
            # well-supported square first), without ever outranking a capture.
            if U is None:
                try:
                    masses = np.abs(_MASS_LUT[np.abs(board.pieces).astype(np.int8)]).astype(np.float32)
                    gate = 1.0 / (1.0 + np.exp(-(float(constants.c) - _DIST_NP)))
                    r = np.sqrt(_DIST2_NP + float(constants.eps) ** 2)
                    U = -(gate / r) @ masses  # (64,) potential at every square
                except Exception:
                    U = np.zeros(64)
            sc = float(U[int(m[1])])
        scored.append((m, sc))

    scored.sort(key=lambda t: -t[1])
    return [m for m, _ in scored]


def negamax(engine, board: FastBoard, depth: int, alpha: float, beta: float,
            masses_override=None, allow_null: bool = True, parent_masses=None,
            stats=None, ctx=None, ply: int = 0):
    """Negamax with PVS, null-move pruning, TT, killers and history.

    Returns the score from the side-to-move's perspective. `ctx` carries the
    per-search transposition table, killer slots, and history table; when None
    a throwaway context is used (correct, but no cross-node caching).
    """
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1
    consts = engine.constants

    current_mv = board.mass_vector() if masses_override is None else masses_override
    parent = board.mass_vector() if parent_masses is None else parent_masses

    # TT probe: full physics state keyed.
    tt_move = None
    if ctx is not None:
        hit = ctx.tt.lookup(board, current_mv, parent)
        if hit is not None:
            h_depth, h_flag, h_score, h_move = hit
            tt_move = None if h_move < 0 else (
                (h_move >> 16) & 0x3F, (h_move >> 8) & 0x3F, h_move & 0xF)
            if h_depth >= depth:
                if h_flag == _TT_EXACT:
                    return h_score
                if h_flag == _TT_LOWER and h_score >= beta:
                    return h_score
                if h_flag == _TT_UPPER and h_score <= alpha:
                    return h_score

    # Check extension: a side in check gets one extra ply so escape/capture
    # replies are resolved before the horizon.
    if board.in_check(board.turn == 0):
        depth += 1

    if depth <= 1:
        return _quiesce(engine, board, alpha, beta, current_mv, qdepth=3,
                        parent_masses=parent, stats=stats, ctx=ctx)

    consts = engine.constants

    # Null-move pruning ("gravitational field probe"): if passing the turn
    # still leaves us at or above beta, no opponent reply can save it — prune.
    # The window (-beta, -beta+1) is only well-formed for a FINITE beta; with
    # beta = +INF it degenerates to (alpha=-INF, beta=-INF) and every stand-pat
    # cutoffs to -INF, which poisoned every search at depth >= 3 (the old
    # engine returned +INF at the root and best_move collapsed to None).
    if allow_null and depth >= 3 and beta < INF and not board.in_check(board.turn == 0):
        null_board = FastBoard(board.pieces.copy(), 1 - board.turn,
                               board.castling, -1)
        null_val = -negamax(engine, null_board, depth - 3, -beta, -beta + 1,
                            current_mv, allow_null=False,
                            parent_masses=current_mv, stats=stats, ctx=ctx,
                            ply=ply)
        if null_val >= beta:
            # Fail-soft: return the bound, not beta. Returning the flat `beta`
            # here is harmless for null move, but the same fail-hard pattern in
            # quiescence was producing false mates (a zero-window probe against
            # a mate-inflated alpha returned beta, which negated to +MATE).
            return null_val

    killers = ctx.killers if ctx is not None else None
    history = ctx.history if ctx is not None else None
    ordered = _ordered_moves(board, consts, tt_move, killers, history,
                             ply, root_pv=None)
    if not ordered:
        # No legal moves: mate (side in check) or stalemate — decided without a
        # second move generation (the list is already the terminal probe).
        return -MATE if board.in_check(board.turn == 0) else 0.0

    best = -INF
    best_move = None
    first = True
    for mi, m in enumerate(ordered):
        child = board.apply(m)
        mv = child_mass_vector(board, m, current_mv, child_board=child)
        is_capture = board.is_capture(m)
        gives_check = child.in_check(child.turn == 0)
        if first:
            val = -negamax(engine, child, depth - 1, -beta, -alpha, mv,
                           parent_masses=current_mv, stats=stats, ctx=ctx,
                           ply=ply + 1)
            first = False
        else:
            # LMR (Late Move Reduction): late, quiet, non-checking moves are
            # probed at reduced depth first. Only a reduced score that climbs
            # inside the window earns a full-depth re-search. Captures and
            # checks are always searched at full width (they are forcing).
            reduce = (depth >= 3 and mi >= 3 and not is_capture and not gives_check)
            r = 1 + (mi - 3) // 6 if reduce else 0
            reduced_depth = max(depth - 1 - r, 1)
            # PVS: zero-window probe (possibly at reduced depth), re-search
            # only when the score climbs inside the window. (Gate must be
            # alpha < val < beta; the old -beta < val < -alpha form let
            # fail-soft zero-window values leak through as exact.)
            val = -negamax(engine, child, reduced_depth, -alpha - 1, -alpha, mv,
                           parent_masses=current_mv, stats=stats, ctx=ctx,
                           ply=ply + 1)
            if alpha < val < beta:
                val = -negamax(engine, child, depth - 1, -beta, -alpha, mv,
                               parent_masses=current_mv, stats=stats, ctx=ctx,
                               ply=ply + 1)
        if val > best:
            best = val
            best_move = m
        if best > alpha:
            alpha = best
        if alpha >= beta:
            if stats is not None:
                stats["cutoffs"] = stats.get("cutoffs", 0) + 1
            # Record cut-off move in killers + history for future ordering.
            if ctx is not None and best_move is not None and not board.is_capture(best_move):
                k1, k2 = ctx.killers[ply]
                ctx.killers[ply] = [best_move, k1]
                ctx.history[best_move[0] * 64 + best_move[1]] += depth * depth
            break
        # A forced mate is found: with flat mate scoring nothing scores higher,
        # and continuing would push alpha to the mate ceiling, corrupting the
        # remaining zero-window probes into false +MATE values. Stop the node.
        if best >= MATE - 1:
            break

    # TT store (side-to-move values, full-state key). Fail-soft flagging:
    #   best <= alpha  -> upper bound (fail-low)
    #   best >= beta   -> lower bound (fail-high)
    #   else           -> exact value
    if ctx is not None:
        if best <= alpha:
            flag = _TT_UPPER
        elif best >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        ctx.tt.store(board, current_mv, parent, depth,
                     flag, best, best_move)
    return best


def _quiesce(engine, board: FastBoard, alpha: float, beta: float,
             masses_override=None, qdepth: int = 4, parent_masses=None,
             stats=None, ctx=None):
    """Quiescence search: resolve forcing lines so the eval isn't read at a
    volatile horizon.

    Standard stand-pat with capture expansion — and, when the side to move is
    in check, ALL legal evasions are considered (quiet escapes included), so
    hanging-piece saves and king escapes are not pushed past the horizon.
    Capture children of a node are scored in ONE `batch_score` vmap (the
    218-pad sweep), so the cost per node is a single XLA call rather than one
    score_white per capture. Capped to avoid blow-ups.
    """
    in_chk = board.in_check(board.turn == 0)
    moves = board.legal_moves()
    if not moves:
        # Mate or stalemate — decided from the existing generation.
        return -MATE if in_chk else 0.0
    if stats is not None:
        stats["qnodes"] = stats.get("qnodes", 0) + 1
    current_mv = masses_override if masses_override is not None else board.mass_vector()
    parent = board.mass_vector() if parent_masses is None else parent_masses
    stand = _score_position(current_mv, engine.constants, board.turn,
                            parent=parent)
    if stand >= beta:
        # Fail-soft: return the exact stand-pat value, not the flat beta. A
        # fail-hard `return beta` here, inside a zero-window probe whose beta
        # is -MATE (alpha was already pushed to a mate), negates to a FALSE
        # +MATE at the parent — the "every quiet move scores mate" bug.
        return stand
    if stand > alpha:
        alpha = stand

    if in_chk:
        # In check: every legal evasion matters (quiet moves can save a piece
        # or flee the attack). Evasions include captures, so this subsumes the
        # capture set below.
        evasions = moves
        evasions.sort(key=lambda m: (0 if board.is_capture(m) else 1,
                                     -_mass_capture_value(board, m)))
        moves = evasions[:14]
    else:
        # Stand-pat quiescence: expand captures AND quiet checking moves. A
        # quiet move that gives check is forcing — ignoring it at the horizon
        # is the classic way a search hangs pieces to a discovered check or a
        # mate it never saw.
        caps = []
        checks = []
        for m in moves:
            if board.is_capture(m):
                caps.append(m)
            else:
                nb = board.apply(m)
                if nb.in_check(nb.turn == 0):
                    checks.append(m)
        if not caps and not checks:
            return alpha
        caps.sort(key=lambda m: -_mass_capture_value(board, m))
        moves = caps[:10] + checks[:4]

    # Score all candidate children in ONE vmap sweep (side-to-move perspective),
    # with parent threading so the move-sensitivity (delta) terms are active.
    child_m = []
    parents = []
    for m in moves:
        child = board.apply(m)
        mv = child_mass_vector(board, m, current_mv, child_board=child)
        child_m.append(mv)
        parents.append(parent)
    pad = 16 if len(moves) <= 16 else 32
    scores = batch_score(child_m, [board.turn] * len(child_m), engine.constants,
                         pad=pad, parents=jnp.stack(parents))
    order = sorted(range(len(moves)), key=lambda i: -float(scores[i]))
    for i in order:
        if qdepth <= 1:
            val = float(scores[i])
        else:
            child = board.apply(moves[i])
            val = -_quiesce(engine, child, -beta, -alpha, child_m[i],
                            qdepth - 1, parent_masses=current_mv, stats=stats,
                            ctx=ctx)
        if val >= beta:
            return val
        if val > alpha:
            alpha = val
    return alpha


class _SearchCtx:
    __slots__ = ("tt", "killers", "history")

    def __init__(self):
        self.tt = TT()
        self.killers = [[None, None] for _ in range(64)]
        self.history = [0] * 4096


def _root_static_order(board: FastBoard, engine, parent_mv):
    """Order root candidates by their TRUE static child scores (parent-threaded
    batch), not by the potential-well tiebreak — so with near-equal search
    values the physically best-looking quiet move gets first crack at setting
    alpha (ties are broken by ordering in a flat landscape)."""
    moves = board.legal_moves()
    if len(moves) <= 1:
        return moves
    masses = [child_mass_vector(board, m, parent_mv) for m in moves]
    turns = [board.turn] * len(moves)
    scores = batch_score(masses, turns, engine.constants,
                         parents=jnp.stack([parent_mv] * len(masses)))
    order = sorted(range(len(moves)), key=lambda i: -float(scores[i]))
    return [moves[i] for i in order]


def _search_root_iteration(engine, board, parent_mv, ordered, d, alpha, beta,
                           seen, stats, ctx, t0, time_ms):
    """Search one root depth with the given window.

    Returns (iter_best, iter_score, scored, timed_out). A timed-out iteration
    returns the moves scored so far and flags ``timed_out`` so the caller can
    keep the last completed depth rather than a partial one.
    """
    iter_best, iter_score = None, -INF
    first = True
    scored = []
    for m in ordered:
        if time_ms is not None and (time.perf_counter() - t0) * 1000.0 > time_ms:
            return iter_best, iter_score, scored, True
        child = board.apply(m)
        mv = child_mass_vector(board, m, parent_mv, child_board=child)
        if seen is not None and child_in_seen(child, seen):
            # Closed orbit: the same physical state already occurred on the
            # line. A cycle exchanges no net mass flux — a draw by physics. Do
            # not pick it over any real progress; search-time reuse is wasted
            # on it (score it as 0 and move on).
            val = 0.0 if d > 1 else -INF
        elif first:
            val = -negamax(engine, child, d - 1, -beta, -alpha, mv,
                           parent_masses=parent_mv, stats=stats, ctx=ctx, ply=1)
            first = False
        else:
            val = -negamax(engine, child, d - 1, -alpha - 1, -alpha, mv,
                           parent_masses=parent_mv, stats=stats, ctx=ctx, ply=1)
            if alpha < val < beta:
                val = -negamax(engine, child, d - 1, -beta, -alpha, mv,
                               parent_masses=parent_mv, stats=stats, ctx=ctx,
                               ply=1)
        if val > iter_score:
            iter_score, iter_best = val, m
        if val > alpha:
            alpha = val
        scored.append((m, val))
        # A forced mate is found at the root: nothing scores higher with flat
        # mate scoring, and continuing would let the mate-inflated alpha corrupt
        # the remaining zero-window probes into false +MATE values.
        if iter_score >= MATE - 1:
            break
    return iter_best, iter_score, scored, False


def iterative_search(engine, board: FastBoard, max_depth: int = 4,
                     time_ms: float | None = None, stats=None,
                     seen=None, use_multiverse: bool = False,
                     multiverse_seed: int = 20260808):
    """Iterative-deepening search with PVS at the root.

    Returns (best_move, best_score) or (None, None) when no legal move exists.
    When `time_ms` is given, deepening stops as soon as the budget is
    exhausted; the best move of the last COMPLETED iteration is returned
    (deeper-but-unfinished searches are never returned). Without a budget the
    search runs to `max_depth` (which may still be cut short by TT cutoffs).

    `seen` is the set of FENs already visited on the game line (the "closed
    orbit" history). A root candidate whose child repeats a seen position is
    a zero-net-momentum closed cycle — there is no real progress in it, so it
    is scored as the equilibrium (draw) value. This is the physics-native
    cure for the endless Rb1-Ra1-Rb1 loops.

    `use_multiverse` runs the Layer-2 posterior-average evaluation (the
    multiverse) on a small re-score head: the top candidates are re-ranked
    under the K-realization mean when the ordinary search cannot separate
    them (they lie within `_TIE_EPS`). This is deliberately root-only —
    K× scoring on every leaf would cost 8× per node for no tactical win.
    """
    moves = board.legal_moves()
    if not moves:
        return None, None
    ctx = _SearchCtx()
    parent_mv = board.mass_vector()
    root_pv = None
    ordered = None
    best, best_score = None, None
    t0 = time.perf_counter()
    scored = []
    prev_score = None

    for d in range(1, max_depth + 1):
        if d == 1:
            ordered = _root_static_order(board, engine, parent_mv)
        else:
            ordered = _ordered_moves(board, engine.constants, tt_move=None,
                                     killers=None, history=None, ply=0,
                                     root_pv=root_pv)

        # Aspiration window: from depth 3 onward assume the previous iteration's
        # score holds and search a narrow window first. A fail-low/high re-search
        # full-width below, so the returned value is identical to a full-width
        # search — the window is a pure time-saver, never a change in verdict.
        if prev_score is not None and d >= 3:
            delta = 60.0 + 20.0 * d
            alpha, beta = prev_score - delta, prev_score + delta
        else:
            alpha, beta = -INF, INF

        iter_best, iter_score, iter_scored, timed_out = _search_root_iteration(
            engine, board, parent_mv, ordered, d, alpha, beta, seen, stats,
            ctx, t0, time_ms)

        if timed_out:
            # Budget spent mid-iteration: keep the last complete result. Never
            # fall back to a searcher-blind move (first generated): the current
            # iteration's ordering already encodes the static verdict when no
            # move has been committed yet.
            if best is None:
                best = ordered[0]
            return best, (best_score if best_score is not None else 0.0)

        # Aspiration fail: the true score left the window — re-search full-width.
        if prev_score is not None and d >= 3 and (
                iter_score <= alpha or iter_score >= beta):
            iter_best, iter_score, iter_scored, timed_out = _search_root_iteration(
                engine, board, parent_mv, ordered, d, -INF, INF, seen, stats,
                ctx, t0, time_ms)
            if timed_out:
                if best is None:
                    best = ordered[0]
                return best, (best_score if best_score is not None else 0.0)

        scored = iter_scored
        best, best_score = iter_best, iter_score
        root_pv = iter_best
        prev_score = best_score
        if time_ms is not None and (time.perf_counter() - t0) * 1000.0 > time_ms:
            break

    # Multiverse verification head. The multiverse must arbitrate ties the
    # SEARCH could not separate: candidates whose searched values are within
    # `eps` of the searched best. (Static near-ties would let a statically
    # flat move override the search verdict — the exact failure mode this
    # head is meant to cure.) Deterministic seed: the posterior draws must
    # not add random-play variance to the game.
    if use_multiverse and best is not None and best_score is not None:
        k1 = jax.random.PRNGKey(multiverse_seed)
        near = [(m, val) for m, val in scored if abs(val - best_score) <= 1.2]
        near.sort(key=lambda t: -t[1])
        near = near[:4]
        if len(near) > 1:
            winner, wscore = near[0][0], None
            for m, _ in near:
                child = board.apply(m)
                mv = child_mass_vector(board, m, parent_mv, child_board=child)
                s = _score_position(mv, engine.constants, board.turn,
                                    use_multiverse=True, key=k1, K=8,
                                    sigma=0.1, parent=parent_mv)
                if wscore is None or s > wscore:
                    winner, wscore = m, s
            best = winner
    return best, best_score


def child_in_seen(child, seen) -> bool:
    """True if the child position is a position seen earlier on the line."""
    try:
        fen = child.to_chess().fen()
        return " ".join(fen.split()[:4]) in seen
    except Exception:
        return False


def best_move(engine, board: FastBoard, depth: int = 3):
    """Iterative-deepening search to a fixed depth; returns the best move."""
    mv, _ = iterative_search(engine, board, max_depth=depth)
    return mv


def best_move_time(engine, board: FastBoard, time_ms: float,
                   max_depth: int = 8):
    """Time-budgeted search; returns the best move of the deepest completed
    iteration within `time_ms`."""
    mv, _ = iterative_search(engine, board, max_depth=max_depth, time_ms=time_ms)
    return mv


def root_sweep(engine, board: FastBoard):
    """The headline 218-pad vmap sweep at the root; returns the best move.

    Parent masses are threaded so the move-sensitivity (delta) terms are active
    at the root too — each child is scored against the root position."""
    moves = board.legal_moves()
    if not moves:
        return None
    parent_mv = board.mass_vector()
    masses = [child_mass_vector(board, m, parent_mv) for m in moves]
    parents = jnp.stack([parent_mv] * len(masses))
    turns = [board.turn] * len(masses)
    scores = batch_score(masses, turns, engine.constants, parents=parents)
    return moves[int(jnp.argmax(scores))]
