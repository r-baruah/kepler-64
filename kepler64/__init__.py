"""Kepler-64 — the Roche Engine.

A chess engine whose evaluation is a differentiable N-body gravitational
potential. It does not understand chess; it understands orbital mechanics and
structural collapse.

  chess:  a position is evaluated by physics, not by heuristics.
  physics: pieces are masses; the enemy King "collapses" past its Roche/tidal
           limit when the tidal Jacobian at its coordinate exceeds a learned
           binding threshold.
"""

from .core.constants import Constants
from .core.board import Board
from .core.evaluate import evaluate
from .multiverse import (
    apply_capture,
    uncertainty_field,
    stokes_flow,
    observe_update,
    multiverse_score_white,
)

__all__ = [
    "RocheEngine",
    "Constants",
    "Board",
    "evaluate",
    "apply_capture",
    "uncertainty_field",
    "stokes_flow",
    "observe_update",
    "multiverse_score_white",
]


class RocheEngine:
    """Instantiate a universe.

    Holds the (learnable) physical constants of this chess universe and exposes
    the evaluation interface used by the search tree.
    """

    def __init__(self, constants: Constants | None = None, seed_image=None,
                 load_trained: bool = True):
        # Constants resolution order:
        #   1. an explicitly supplied Constants (tests, ablations, matches)
        #   2. the persisted trained artifact (trained_constants.json) if it
        #      exists — the trained universe the pipeline last accepted
        #   3. the pristine hand-set universe
        if constants is None and load_trained:
            from .core.constants import load_constants
            constants = load_constants() or Constants()
        self.constants = constants or Constants()
        if seed_image is not None:
            from .core.image_seed import seed_from_image

            self.constants = seed_from_image(seed_image, self.constants)

    def evaluate(self, board: Board) -> float:
        """Score a position from the perspective of the side to move.

        Positive = good for the side to move; negative = its King is being
        tidally disrupted.
        """
        return evaluate(board, self.constants)

    def play(self, board, depth: int = 3, search_time_ms: float | None = None,
             max_depth: int = 8, use_multiverse: bool = True,
             multiverse_seed: int | None = None, observe: bool = False):
        """Search + return the best move (as a python-chess Move).

        `board` may be a chess.Board, a FastBoard, or a `GameState`
        (core/universe.py). Only a GameState carries the game's history into
        the physics — accreted mass from earlier captures and the velocity
        behind the Lorentz boost; the other two start the root at rest mass.

        `depth` caps iterative deepening (default 3). Pass `search_time_ms` to
        search with a wall-clock budget instead of a fixed depth: the engine
        deepens until the budget is exhausted and returns the best move of the
        last COMPLETED iteration (never an unfinished deeper search).

        `use_multiverse` (default True) runs the Layer-2 posterior mean on the
        root's near-tie candidates, so the multiverse breaks ties the search
        cannot; it is deterministic (fixed posterior seed). Pass
        `multiverse_seed` to reseed the posterior draws — training uses this
        for exploration noise on self-play games without changing the physics.

        `observe` (default False) enables the Layer-2 Observer: after the move
        is chosen, `self.constants` shifts one tiny KL-anchored step toward
        the physics that explains the resulting position, and STAYS shifted on
        this engine instance — later calls in the same game play under the
        evolved laws. Keep it False for reproducible matches and benchmarks.
        """
        import chess

        from .core.fastboard import FastBoard
        from .search.minimax import iterative_search, seen_positions

        from .core.universe import GameState

        root_masses = None
        if isinstance(board, GameState):
            # Carried universe: accretion + velocity from the real game line.
            state = board
            board = state.board
            fb = state.fb
            root_masses = state.masses
        else:
            fb = board if isinstance(board, FastBoard) else FastBoard.from_chess(board)
        mv_seed = 20260808 if multiverse_seed is None else multiverse_seed

        # Closed-orbit history: every position already on the line (board
        # identity only — pieces/turn/castling/ep — NOT the halfmove clocks,
        # which change on every ply and would defeat the comparison). A root
        # move that lands back on a seen board is a cycle with no net
        # momentum: the laws assign it the draw value.
        seen = seen_positions(board) if isinstance(board, chess.Board) else None

        if search_time_ms is not None:
            mv, _ = iterative_search(self, fb, max_depth=max_depth,
                                     time_ms=search_time_ms, seen=seen,
                                     use_multiverse=use_multiverse,
                                     multiverse_seed=mv_seed,
                                     root_masses=root_masses)
        else:
            mv, _ = iterative_search(self, fb, max_depth=depth, seen=seen,
                                     use_multiverse=use_multiverse,
                                     multiverse_seed=mv_seed,
                                     root_masses=root_masses)
        if mv is None:
            return None
        f, t, promo = mv

        # Layer-2 Observer: one KL-anchored belief shift after the move, so the
        # laws this engine plays under co-evolve with the game (opt-in).
        if observe:
            from .core.transitions import child_mass_vector
            from .multiverse.observer import observe_update

            fb_child = fb.apply(mv)
            parent_mv = (fb.mass_vector(float(self.constants.c))
                         if root_masses is None else root_masses)
            child_masses = child_mass_vector(
                fb, mv, parent_mv, child_board=fb_child,
                c_lorentz=float(self.constants.c))
            self.constants = observe_update(self.constants, child_masses)

        return chess.Move(f, t, chess.PieceType(promo) if promo else None)

    def uncertainty_field(self, eval_samples):
        """Compute the spatial variance field across Multiverse evaluation samples (64,)."""
        return uncertainty_field(eval_samples)

    def stokes_flow(self, eval_samples=None, uncertainty=None, mu: float = 1.0):
        """Compute the 8x8 Stokes velocity vector field (64, 2) from Multiverse variance."""
        if uncertainty is None:
            if eval_samples is None:
                raise ValueError("Either eval_samples or uncertainty must be provided.")
            uncertainty = uncertainty_field(eval_samples)
        return stokes_flow(uncertainty, mu=mu)

    def accrete_capture(self, masses, captor_sq: int, captured_sq: int,
                        eta_acc: float = 0.8, Rg_old: float = 1.0):
        """Apply 2C accretion on capture: captor absorbs captured mass, expanding Rg."""
        return apply_capture(masses, captor_sq, captured_sq, eta_acc=eta_acc, Rg_old=Rg_old)

    def warmup(self):
        """Pre-warm JAX JIT compilation for single-board and batch scoring.

        Traces and compiles both `score_white` and `batch_score` kernels on dummy
        tensors so subsequent searches and matches execute with zero JIT latency.
        """
        import jax.numpy as jnp
        from .core.evaluate import score_white, batch_score

        dummy_mv = jnp.zeros(64, dtype=jnp.float32)
        _ = score_white(dummy_mv, self.constants, parent=dummy_mv).block_until_ready()

        dummy_batch16 = [dummy_mv] * 16
        dummy_turns16 = [0] * 16
        dummy_parents16 = jnp.zeros((16, 64), dtype=jnp.float32)
        _ = batch_score(dummy_batch16, dummy_turns16, self.constants, pad=16,
                        parents=dummy_parents16).block_until_ready()
        return self

