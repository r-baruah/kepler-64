"""Training loop - gradient descent through the physics engine.

physics: optimize G, eps, c, roche, bonus, kgain, gamma, Rg with Adam so the
         gravitational "weights" are genuinely learned from real play, not
         hand-picked.  Adam adapts the LR per-parameter, which is critical here
         because the 8 physics params span very different scales (bonus≈50 vs
         eps≈0.5).  Gradient clipping (max_norm=1.0) prevents noisy mini-batches
         from causing destructive updates to roche or eps.
         G is frozen (fix_G=True) because it cancels out of the tidal index and
         training otherwise collapses it toward 0, killing the force terms.
chess:  the result is an engine whose evaluation mimics real, structurally sound
        positions / expert moves without ever being told a chess rule.  Training
        data is REAL (puzzles + your own games) - no invented heuristics.

Hardware notes (Ryzen 5 5500U, 8 GB RAM, Radeon iGPU, no CUDA):
  * JAX runs CPU-only here.  We force XLA to use all 12 logical threads via
    --xla_force_host_platform_device_count (set in _configure_jax below) — by
    default it only uses the physical core count, leaving half the laptop idle.
  * 8 GB is the hard limit.  The biggest tensor is moves_m (N,K,64) float32:
    5000 puzzles * 64 * 64 * 4 bytes ~= 85 MB, plus the per-step vmap over 64
    children.  We keep batch_size=128 so a single step's working set stays well
    under ~1 GB and we never hit the swap file.
  * optax (pure-python, tiny) gives Adam.  Without it we fall back to a
    gradient-norm-clipped SGD, which is ~3x slower to converge.
"""

import os

import numpy as np

# ── JAX CPU tuning for this laptop ─────────────────────────────────────────
# Force XLA to see all 12 logical threads (Ryzen 5500U: 6 cores / 12 threads).
# Without this JAX pins to physical-core count and wastes half the machine.
_n_logical = os.cpu_count() or 4
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_n_logical}"
# Keep JAX from hoarding threads for its own BLAS fork and starving the trainer.
os.environ.setdefault("OMP_NUM_THREADS", str(_n_logical))

import jax
import jax.numpy as jnp

try:
    import optax
    _HAS_OPTAX = True
except ImportError:  # pragma: no cover
    optax = None     # type: ignore[assignment]
    _HAS_OPTAX = False
    import warnings
    warnings.warn(
        "optax not found — falling back to plain SGD (slow).  "
        "Run `pip install optax` to enable the Adam optimiser.",
        stacklevel=1,
    )

from ..core.constants import Constants
from .loss import loss as loss_fn
from .data import to_arrays


def _to_arr(c: Constants) -> "jnp.ndarray":
    # 9 trainable physical leaves: the 8 original + mat_gain (unfrozen scale).
    return jnp.array([c.G, c.eps, c.c, c.roche, c.bonus, c.kgain, c.gamma, c.Rg,
                     c.mat_gain], dtype=jnp.float32)


def _from_arr(a) -> Constants:
    return Constants(
        G=float(a[0]),
        eps=float(jnp.clip(a[1], 0.01, 20.0)),
        c=float(jnp.clip(a[2], 1.0, 10.0)),
        roche=float(jnp.clip(a[3], 0.05, 20.0)),
        bonus=float(jnp.clip(a[4], 0.01, 500.0)),
        kgain=float(jnp.clip(a[5], 0.01, 50.0)),
        gamma=float(jnp.clip(a[6], 0.0, 50.0)),
        Rg=float(jnp.clip(a[7], 0.1, 10.0)),
        mat_gain=float(jnp.clip(a[8], 0.0, 5.0)),  # material scale stays modest
    )


# Physical bounds for projection after each Adam step.  Index 8 = mat_gain.
_LO = jnp.array([0.01, 0.01, 1.0,  0.05, 0.01, 0.01, 0.0, 0.1, 0.0 ], dtype=jnp.float32)
_HI = jnp.array([50.0, 20.0, 10.0, 20.0, 500., 50.0, 50., 10.0, 5.0 ], dtype=jnp.float32)


def train(base: Constants, M, Y, turns=None, moves_m=None, mask=None, expert_idx=None,
           steps: int = 200, lr: float = 3e-3, fix_G: bool = False,
           batch_size: int = 256, seed: int = 0, tau: float = 2.0,
           margin: float = 0.0) -> Constants:
    """Mini-batch Adam over (M, Y, turns, moves_m, mask, expert_idx).

    M (N,64) mass vectors; Y (N,) outcomes {-1,0,+1} White-view; turns (N,)
    0=White/1=Black; moves_m (N,K,64) child mass vectors; mask (N,K);
    expert_idx (N,).  Outcome-only training: pass moves_m=None.

    Adam (lr≈3e-3) + gradient clipping (max_norm=1.0) + projected params back
    into physical bounds after every step.
    """
    M = jnp.asarray(M, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    N = int(M.shape[0])
    has_policy = moves_m is not None and expert_idx is not None

    if has_policy:
        moves_m = jnp.asarray(moves_m, dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.float32)
        expert_idx = jnp.asarray(expert_idx, dtype=jnp.int32)
        turns = jnp.asarray(turns, dtype=jnp.float32) if turns is not None \
            else jnp.zeros((N,), dtype=jnp.float32)
        has_policy = jnp.array(1.0)
    else:
        moves_m = jnp.zeros((N, 1, 64), dtype=jnp.float32)
        expert_idx = jnp.zeros((N,), dtype=jnp.int32)
        mask = jnp.ones((N, 1), dtype=jnp.float32)
        turns = jnp.zeros((N,), dtype=jnp.float32)
        has_policy = jnp.array(0.0)

    rng = np.random.default_rng(seed)
    arr = _to_arr(base)
    if fix_G:
        arr = arr.at[0].set(1.0)

    if _HAS_OPTAX:
        # ── Adam + gradient clipping (preferred) ─────────────────────────────
        # Clipping max_norm=1.0 prevents a single noisy batch from launching
        # roche or eps to an extreme value in one step.
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
        )
        opt_state = opt.init(arr)

        @jax.jit
        def _step(a, state, bM, bY, bMM, bEI, bMK, bT):
            val, g = jax.value_and_grad(loss_fn)(a, bM, bY, bMM, bEI, has_policy, bMK, bT, tau, margin)
            if fix_G:
                g = g.at[0].set(0.0)
            updates, new_state = opt.update(g, state)
            new_a = optax.apply_updates(a, updates)
            new_a = jnp.clip(new_a, _LO, _HI)
            if fix_G:
                new_a = new_a.at[0].set(1.0)
            return val, new_a, new_state

        for _ in range(steps):
            perm = rng.permutation(N)
            for s in range(0, N, batch_size):
                idx = jnp.asarray(perm[s:s + batch_size], dtype=jnp.int32)
                _, arr, opt_state = _step(
                    arr, opt_state,
                    M[idx], Y[idx], moves_m[idx],
                    expert_idx[idx], mask[idx], turns[idx],
                )

    else:
        # ── SGD fallback (no optax) ───────────────────────────────────────────
        # Manual gradient-norm clip + projected SGD.
        @jax.jit
        def _step_sgd(a, bM, bY, bMM, bEI, bMK, bT):
            val, g = jax.value_and_grad(loss_fn)(a, bM, bY, bMM, bEI, has_policy, bMK, bT, tau, margin)
            if fix_G:
                g = g.at[0].set(0.0)
            # clip gradient norm to 1.0
            gnorm = jnp.sqrt(jnp.sum(g ** 2)) + 1e-9
            g = jnp.where(gnorm > 1.0, g / gnorm, g)
            return val, g

        for _ in range(steps):
            perm = rng.permutation(N)
            for s in range(0, N, batch_size):
                idx = jnp.asarray(perm[s:s + batch_size], dtype=jnp.int32)
                _, g = _step_sgd(
                    arr,
                    M[idx], Y[idx], moves_m[idx],
                    expert_idx[idx], mask[idx], turns[idx],
                )
                arr = arr - lr * g
                arr = jnp.clip(arr, _LO, _HI)
                if fix_G:
                    arr = arr.at[0].set(1.0)

    return _from_arr(arr)



def train_examples(base: Constants, examples, steps: int = 200, lr: float = 3e-3,
                   fix_G: bool = False, batch_size: int = 128, seed: int = 0,
                   val_frac: float = 0.2, verbose: bool = True,
                   tau: float = 2.0, margin: float = 0.0):
    """Convenience: build arrays from examples (list of dicts), split train/val,
    train, and report validation ranking metrics so we can see real progress
    (not just overfitting on the training set).

    Laptop defaults: batch_size=128 (keeps a step's working set < ~1 GB of the
    8 GB total), val_frac=0.2 (held-out check).  tau=2.0 softens the policy
    softmax; margin>0 adds pairwise margin-ranking (easier than sharp CE).
    """
    if not examples:
        raise ValueError(
            "train_examples received an empty example list. "
            "Check that puzzle_examples / game_examples loaded data correctly "
            "(CSV path, column names, move format)."
        )
    M, Y, turns, moves_m, mask, expert_idx = to_arrays(examples)
    n = int(M.shape[0])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int((1.0 - val_frac) * n)
    tr, va = perm[:split], perm[split:]

    trained = train(
        base, M[tr], Y[tr], turns[tr], moves_m[tr], mask[tr], expert_idx[tr],
        steps, lr, fix_G, batch_size, seed, tau=tau, margin=margin,
    )

    if verbose:
        from .loss import policy_metrics
        Mv, Yv, tv, mmv, mkv, eiv = (
            M[va], Y[va], turns[va], moves_m[va], mask[va], expert_idx[va])
        b = policy_metrics(base, Mv, Yv, tv, mmv, mkv, eiv)
        t = policy_metrics(trained, Mv, Yv, tv, mmv, mkv, eiv)
        print(f"[train] N={n}  train={split} val={n - split}  "
              f"steps={steps} lr={lr} tau={tau} margin={margin}")
        print(f"[train] baseline  top1={b['top1']:.3f}  mrr={b['mrr']:.3f}")
        print(f"[train] trained   top1={t['top1']:.3f}  mrr={t['mrr']:.3f}")
        print(f"[train] delta mrr              : {t['mrr'] - b['mrr']:+.3f}")
    return trained


def train_from_data(base: Constants, data_dir: str, steps: int = 200,
                    lr: float = 3e-3, fix_G: bool = True,
                    puzzle_limit: int = 5000, game_limit: int = 30,
                    game_positions: int = 8000, seed: int = 0, verbose: bool = True):
    """One-call pipeline for this laptop: load real data, train, report.

    Defaults chosen for the Ryzen 5 5500U / 8 GB machine:
      * puzzle_limit=5000 (the stale 200-puzzle run was too small to learn)
      * game_limit=30, game_positions=8000 (your own games as behavioural clone)
      * fix_G=True (G is non-identifiable in the tidal index)
    """
    from .data import puzzle_examples, game_examples
    puz = puzzle_examples(os.path.join(data_dir, "puzzles_50k.csv"),
                          limit=puzzle_limit)
    if verbose:
        print(f"[train] loaded {len(puz)} puzzle examples")
    games = game_examples(os.path.join(data_dir, "games", "Ripu01.pgn"),
                          limit_games=game_limit, limit_positions=game_positions)
    if verbose:
        print(f"[train] loaded {len(games)} game-position examples")
    examples = puz + games
    if not examples:
        raise ValueError("No training examples loaded — check data_dir path.")
    return train_examples(base, examples, steps, lr, fix_G,
                          batch_size=128, seed=seed, verbose=verbose)


