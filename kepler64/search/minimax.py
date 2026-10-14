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
# Mate-distance scoring: a mate delivered `p` plies from the root scores
# +/-(MATE - p), so faster mates win and slower mates are delayed. Any score
# with |s| >= _MATE_BOUND is "mate-ish" (search never reaches 512 plies).
_MATE_BOUND = MATE - 512.0

# Multiverse re-score head: candidates whose searched values lie within this
# many Kepler units of the searched best are arbitrated by the K-realization mean.
_TIE_EPS = 1.2

# Check the wall-clock deadline every this many nodes (negamax + quiescence).
_TIME_CHECK_NODES = 16

# History scores are capped well below the killer (20k) and capture (100k)
# bonuses so a quiet move's history can never outrank them.
_HISTORY_MAX = 10_000

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
_SQ_64 = np.arange(64, dtype=np.int64)


def _mass_checksum(m) -> int:
    """Exact 64-bit checksum of a (64,) float32 mass vector.

    Any change in any float32 bit (accretion boost, Lorentz factor, moved mass)
    changes the checksum. O(64) numpy ops — the old blake2b-over-1.5KB hash
    cost ~50x more per TT probe and kept the table effectively unreachable.
    """
    bits = np.asarray(m, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return int((bits * _MASS_RAND).sum()) & 0xFFFFFFFFFFFFFFFF


def _tt_score_in(score: float, ply: int) -> float:
    """Root-relative mate score -> node-relative, for storing in the TT."""
    if score >= _MATE_BOUND:
        return score + ply
    if score <= -_MATE_BOUND:
        return score - ply
    return score


def _tt_score_out(score: float, ply: int) -> float:
    """Node-relative TT mate score -> root-relative at the probing ply."""
    if score >= _MATE_BOUND:
        return score - ply
    if score <= -_MATE_BOUND:
        return score + ply
    return score


class _SearchTimeout(Exception):
    """Raised inside the tree when the wall-clock deadline passes."""


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
        idx = pieces.astype(np.int64) + 6  # 0=empty, 1..12 pieces
        h = int(np.bitwise_xor.reduce(np.where(idx != 6, _ZOB[_SQ_64, idx], 0)))
        h ^= int(_ZOB_CASTLE[int(board.castling) & 0xF])
        h ^= int(_ZOB_EP[int(board.ep) + 1])  # ep in [-1,63] -> [0,64]
        h ^= int(_ZOB_TURN[int(board.turn)])
        h = (h ^ _mass_checksum(child_m) ^ _mass_checksum(parent_m)) & 0xFFFFFFFFFFFFFFFF
        return h

    def lookup(self, board, child_m, parent_m):
        return self.probe(self._hash(board, child_m, parent_m))

    def probe(self, h: int):
        """Look up by a precomputed hash (see `_hash`)."""
        slot = h & self.mask
        k = self.keys[slot]
        if self.occupied[slot] and k == h:
            return (int(self.depth[slot]), int(self.flag[slot]),
                    float(self.score[slot]), int(self.move[slot]))
        return None

    def store(self, board, child_m, parent_m, depth: int, flag: int,
              score: float, move):
        self.put(self._hash(board, child_m, parent_m), depth, flag, score, move)

    def put(self, h: int, depth: int, flag: int, score: float, move):
        """Store by a precomputed hash (see `_hash`)."""
        slot = h & self.mask
        # Depth-preferred replacement: a DEEPER stored entry always survives,
        # whether the incoming key differs (different position, same slot) or
        # matches (revisiting the same state at reduced depth). The previous
        # `and self.keys[slot] != h` clause exempted same-key hits, so a
        # shallow revisit silently destroyed the deeper cached entry.
        if self.occupied[slot] and self.depth[slot] >= depth:
            return  # keep the deeper (or equal-depth, older) entry
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
        cap = board.is_capture(m)
        if tt_move is not None and m == tt_move:
            sc += 1_000_000.0
        elif cap:
            sc += 100_000.0 + _mass_capture_value(board, m)
        if root_pv and m == root_pv:
            sc += 50_000.0
        if killers:
            if m in killers[ply]:
                sc += 20_000.0
        if history is not None and not cap:
            # Quiet moves only: history is a quiet-move heuristic and is capped
            # (_HISTORY_MAX) so it never outranks killers or captures.
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
    if ctx is not None:
        ctx.tick()
    consts = engine.constants

    current_mv = board.mass_vector(float(consts.c)) if masses_override is None else masses_override
    parent = board.mass_vector(float(consts.c)) if parent_masses is None else parent_masses

    # TT probe: full physics state keyed. The hash is computed ONCE per node
    # and reused for the store at the end.
    tt_move = None
    tt_hash = None
    if ctx is not None:
        tt_hash = ctx.tt._hash(board, current_mv, parent)
        hit = ctx.tt.probe(tt_hash)
        if hit is not None:
            h_depth, h_flag, h_score, h_move = hit
            # Mate scores are stored node-relative; make them root-relative.
            h_score = _tt_score_out(h_score, ply)
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
                        parent_masses=parent, stats=stats, ctx=ctx, ply=ply)

    consts = engine.constants

    # (Zugzwang guard: no null move when the side to move has only king and
    # pawns, where passing can genuinely be the best move.)
    # Null-move pruning ("gravitational field probe"): if passing the turn
    # still leaves us at or above beta, no opponent reply can save it — prune.
    # The window (-beta, -beta+1) is only well-formed for a FINITE beta; with
    # beta = +INF it degenerates to (alpha=-INF, beta=-INF) and every stand-pat
    # cutoffs to -INF, which poisoned every search at depth >= 3 (the old
    # engine returned +INF at the root and best_move collapsed to None).
    if (allow_null and depth >= 3 and beta < INF and _has_non_pawn_material(board)
            and not board.in_check(board.turn == 0)):
        # Velocity passes through so the probe's Lorentz-boosted masses match
        # the real line's physical state (a null probe without momentum reads
        # a different universe than the position it stands in for).
        null_board = FastBoard(board.pieces.copy(), 1 - board.turn,
                               board.castling, -1, velocity=board.velocity)
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
        return -(MATE - ply) if board.in_check(board.turn == 0) else 0.0

    best = -INF
    best_move = None
    first = True
    for mi, m in enumerate(ordered):
        child = board.apply(m)
        mv = child_mass_vector(board, m, current_mv, child_board=child,
                               c_lorentz=float(consts.c))
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
                hi = best_move[0] * 64 + best_move[1]
                ctx.history[hi] = min(ctx.history[hi] + depth * depth,
                                      _HISTORY_MAX)
            break
        # Mating at the very next ply is the best a node at `ply` can score
        # (MATE - ply - 1); nothing later can beat it, so stop the node.
        if best >= MATE - ply - 1:
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
        # Mate scores are stored node-relative (see _tt_score_in).
        ctx.tt.put(tt_hash, depth, flag, _tt_score_in(best, ply), best_move)
    return best


def _has_non_pawn_material(board: FastBoard) -> bool:
    """True if the side to move owns any piece besides king and pawns (a
    rule-level fact read off the board; used as the null-move zugzwang guard)."""
    own = board.pieces if board.turn == 0 else -board.pieces
    return bool(np.any((own >= 2) & (own <= 5)))


def _pad_for(n: int) -> int:
    """Static batch width: reuse the 16/32 compiled shapes, then the 218 pad."""
    return 16 if n <= 16 else 32 if n <= 32 else 218


def _quiesce(engine, board: FastBoard, alpha: float, beta: float,
             masses_override=None, qdepth: int = 4, parent_masses=None,
             stats=None, ctx=None, ply: int = 0, known_stand=None):
    """Quiescence search: resolve forcing lines so the eval isn't read at a
    volatile horizon.

    Stand-pat with capture expansion — and, when the side to move is in check,
    NO stand-pat (the side must answer the check; standing pat there is
    unsound) and ALL legal evasions are searched, quiet escapes included.
    Children of a node are scored in ONE `batch_score` vmap against their TRUE
    parent (this node's mass vector, exactly as the recursive call and
    training/loss.py score them), so the cost per node is a single XLA call.
    The batch score of a child doubles as that child's stand-pat (`known_stand`,
    negated into the child's perspective), so no position is scored twice.
    Fail-soft throughout. Mate scores carry the distance `ply`.
    """
    if ctx is not None:
        ctx.tick()
    in_chk = board.in_check(board.turn == 0)
    moves = board.legal_moves()
    if not moves:
        # Mate or stalemate — decided from the existing generation.
        return -(MATE - ply) if in_chk else 0.0
    if stats is not None:
        stats["qnodes"] = stats.get("qnodes", 0) + 1
    current_mv = masses_override if masses_override is not None else board.mass_vector(float(engine.constants.c))
    parent = board.mass_vector(float(engine.constants.c)) if parent_masses is None else parent_masses

    if in_chk:
        best = -INF  # no stand-pat while in check
        # Every legal evasion matters; captures first.
        moves.sort(key=lambda m: (0 if board.is_capture(m) else 1,
                                  -_mass_capture_value(board, m)))
    else:
        if known_stand is not None:
            stand = known_stand
        else:
            stand = _score_position(current_mv, engine.constants, board.turn,
                                    parent=parent)
        if stand >= beta:
            # Fail-soft: return the exact stand-pat value, not the flat beta. A
            # fail-hard `return beta` here, inside a zero-window probe whose
            # beta is -MATE (alpha was already pushed to a mate), negates to a
            # FALSE +MATE at the parent — the "every quiet move scores mate" bug.
            return stand
        best = stand
        if stand > alpha:
            alpha = stand
        # Expand captures AND quiet checking moves. A quiet move that gives
        # check is forcing — ignoring it at the horizon is the classic way a
        # search hangs pieces to a discovered check or a mate it never saw.
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
            return stand
        caps.sort(key=lambda m: -_mass_capture_value(board, m))
        moves = caps[:10] + checks[:4]

    # Score all candidate children in ONE vmap sweep (side-to-move perspective),
    # against this node as their parent so the move-sensitivity (delta) terms
    # match the recursive call and the training objective.
    child_m = []
    for m in moves:
        child = board.apply(m)
        child_m.append(child_mass_vector(board, m, current_mv, child_board=child,
                                         c_lorentz=float(engine.constants.c)))
    scores = batch_score(child_m, [board.turn] * len(child_m), engine.constants,
                         pad=_pad_for(len(moves)),
                         parents=jnp.stack([current_mv] * len(child_m)))
    sc = [float(scores[i]) for i in range(len(moves))]
    order = sorted(range(len(moves)), key=lambda i: -sc[i])
    for i in order:
        if qdepth <= 1:
            val = sc[i]
        else:
            child = board.apply(moves[i])
            val = -_quiesce(engine, child, -beta, -alpha, child_m[i],
                            qdepth - 1, parent_masses=current_mv, stats=stats,
                            ctx=ctx, ply=ply + 1, known_stand=-sc[i])
        if val > best:
            best = val
        if val >= beta:
            return val
        if val > alpha:
            alpha = val
    return best


class _SearchCtx:
    __slots__ = ("tt", "killers", "history", "deadline", "_ticks")

    def __init__(self, deadline: float | None = None):
        self.tt = TT()
        self.killers = [[None, None] for _ in range(64)]
        self.history = [0] * 4096
        self.deadline = deadline  # absolute time.perf_counter() value, or None
        self._ticks = 0

    def tick(self):
        """Count a node; every _TIME_CHECK_NODES check the deadline and abort
        the iteration (caught in iterative_search) once it has passed."""
        self._ticks += 1
        if (self.deadline is not None and self._ticks % _TIME_CHECK_NODES == 0
                and time.perf_counter() > self.deadline):
            raise _SearchTimeout

    def age_history(self):
        """Halve history between iterations so stale cutoffs fade."""
        self.history = [h >> 1 for h in self.history]


def _root_static_order(board: FastBoard, engine, parent_mv):
    """Order root candidates by their TRUE static child scores (parent-threaded
    batch), not by the potential-well tiebreak — so with near-equal search
    values the physically best-looking quiet move gets first crack at setting
    alpha (ties are broken by ordering in a flat landscape)."""
    moves = board.legal_moves()
    if len(moves) <= 1:
        return moves
    c_lor = float(engine.constants.c)
    masses = [child_mass_vector(board, m, parent_mv, c_lorentz=c_lor) for m in moves]
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
        mv = child_mass_vector(board, m, parent_mv, child_board=child,
                               c_lorentz=float(engine.constants.c))
        if seen is not None and child_in_seen(child, seen):
            # Closed orbit: the same physical state already occurred on the
            # line. A cycle exchanges no net mass flux — a draw by physics. Do
            # not pick it over any real progress; search-time reuse is wasted
            # on it (score it as 0 and move on).
            val = 0.0
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
        # Mate in 1 at the root is the highest possible score (MATE - 1); a
        # slower mate can still be improved on, so only stop for this one.
        if iter_score >= MATE - 1:  # mate in 1: the best possible score
            break
    return iter_best, iter_score, scored, False


def iterative_search(engine, board: FastBoard, max_depth: int = 4,
                     time_ms: float | None = None, stats=None,
                     seen=None, use_multiverse: bool = False,
                     multiverse_seed: int = 20260808, root_masses=None):
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
    them (they lie within `_TIE_EPS` (named constant, 1.2)). This is deliberately root-only —
    K× scoring on every leaf would cost 8× per node for no tactical win.

    `root_masses` is the game's carried mass vector (accretion + Lorentz from
    the real line, see core/universe.GameState). Without it the root is
    rebuilt from piece identities and earlier captures are forgotten.
    """
    moves = board.legal_moves()
    if not moves:
        return None, None
    t0 = time.perf_counter()
    ctx = _SearchCtx(deadline=None if time_ms is None else t0 + time_ms / 1000.0)
    parent_mv = (board.mass_vector(float(engine.constants.c)) if root_masses is None
                 else jnp.asarray(root_masses, dtype=jnp.float32))
    root_pv = None
    ordered = None
    best, best_score = None, None
    scored = []
    prev_score = None

    for d in range(1, max_depth + 1):
        if d == 1:
            ordered = _root_static_order(board, engine, parent_mv)
        else:
            ordered = _ordered_moves(board, engine.constants, tt_move=None,
                                     killers=None, history=None, ply=0,
                                     root_pv=root_pv)
            # Previous completed iteration's scores, best first; moves it did
            # not score (cut short by a mate) keep the fallback order after.
            if scored:
                rank = {m: -v for m, v in scored}
                ordered = sorted(ordered, key=lambda m: rank.get(m, INF))

        # Aspiration window: from depth 3 onward assume the previous iteration's
        # score holds and search a narrow window first. A fail-low/high re-search
        # full-width below, so the returned value is identical to a full-width
        # search — the window is a pure time-saver, never a change in verdict.
        if prev_score is not None and d >= 3:
            delta = 60.0 + 20.0 * d
            alpha, beta = prev_score - delta, prev_score + delta
        else:
            alpha, beta = -INF, INF

        try:
            iter_best, iter_score, iter_scored, timed_out = _search_root_iteration(
                engine, board, parent_mv, ordered, d, alpha, beta, seen, stats,
                ctx, t0, time_ms)
        except _SearchTimeout:
            iter_best, iter_score, iter_scored, timed_out = None, -INF, [], True

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
            try:
                iter_best, iter_score, iter_scored, timed_out = _search_root_iteration(
                    engine, board, parent_mv, ordered, d, -INF, INF, seen, stats,
                    ctx, t0, time_ms)
            except _SearchTimeout:
                timed_out = True
            if timed_out:
                if best is None:
                    best = ordered[0]
                return best, (best_score if best_score is not None else 0.0)

        scored = iter_scored
        best, best_score = iter_best, iter_score
        if stats is not None:
            stats["depth"] = d  # deepest COMPLETED iteration
        ctx.age_history()
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
        near = [(m, val) for m, val in scored if abs(val - best_score) <= _TIE_EPS]
        near.sort(key=lambda t: -t[1])
        near = near[:4]
        if len(near) > 1:
            winner, wscore = near[0][0], None
            for m, _ in near:
                child = board.apply(m)
                mv = child_mass_vector(board, m, parent_mv, child_board=child,
                                       c_lorentz=float(engine.constants.c))
                s = _score_position(mv, engine.constants, board.turn,
                                    use_multiverse=True, key=k1, K=8,
                                    sigma=0.1, parent=parent_mv)
                if wscore is None or s > wscore:
                    winner, wscore = m, s
            best = winner
    return best, best_score


def seen_positions(board):
    """Closed-orbit history of a chess.Board: the set of board identities
    (pieces/turn/castling/ep — NOT the halfmove clocks, which change every ply)
    of every position already on the line, current one included. None when the
    board has no move history."""
    if not board.move_stack:
        return None
    b = board.copy()
    seen = {" ".join(b.fen().split()[:4])}
    while b.move_stack:
        b.pop()
        seen.add(" ".join(b.fen().split()[:4]))
    return seen


def child_in_seen(child, seen) -> bool:
    """True if the child position is a position seen earlier on the line."""
    try:
        fen = child.to_chess().fen()
        return " ".join(fen.split()[:4]) in seen
    except Exception:
        return False
