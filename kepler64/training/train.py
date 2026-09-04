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

import json
import os
import shutil
import time
from pathlib import Path

import jax.random as jrandom

import numpy as np

# ── JAX CPU tuning (laptops without a GPU) ─────────────────────────────────
# Force XLA to see all logical threads. Skipped when a GPU is present
# (nvidia-smi) or KEPLER64_NO_XLA_TUNE is set: the device-count flag would
# pin JAX to CPU and strangle cloud-GPU runs.
_n_logical = os.cpu_count() or 4
if ("XLA_FLAGS" not in os.environ
        and not os.environ.get("KEPLER64_NO_XLA_TUNE")
        and shutil.which("nvidia-smi") is None):
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_n_logical}"
# Keep JAX from hoarding threads for its own BLAS fork and starving the trainer.
os.environ.setdefault("OMP_NUM_THREADS", str(_n_logical))

# Persistent XLA compilation cache: avoid recompiling kernels on process restarts.
# Opt out via KEPLER64_NO_CACHE=1; custom path via JAX_COMPILATION_CACHE_DIR.
if not os.environ.get("KEPLER64_NO_CACHE"):
    _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR",
                                os.path.expanduser("~/.cache/kepler64_jax_cache"))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)

import jax
import jax.numpy as jnp

if not os.environ.get("KEPLER64_NO_CACHE"):
    try:
        from jax.experimental.compilation_cache import compilation_cache as _cc
        _cc.set_cache_dir(os.environ["JAX_COMPILATION_CACHE_DIR"])
    except Exception:
        pass

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

from ..core.constants import (
    Constants, leaves_to_array, array_to_leaves, LEAF_LO_F, LEAF_HI_F,
)
from .loss import loss as loss_fn
from .data import to_arrays

# Single source of truth: the leaf array layout and the physical bounds now
# live in core/constants.py.  The trainer, the loss, and JSON persistence all
# read the SAME tuple, so a new leaf or a bound change only ever happens once.


def _to_arr(c: Constants) -> "jnp.ndarray":
    return leaves_to_array(c)


def _from_arr(a) -> Constants:
    return array_to_leaves(a)


# Physical bounds for projection after each Adam step (shared with loss).
_LO = LEAF_LO_F
_HI = LEAF_HI_F


def atomic_save_npz(path: str | Path, **arrays):
    """Write an npz file atomically via temporary file and os.replace."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f"{p.stem}.tmp.{os.getpid()}{p.suffix}")
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, p)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_checkpoint(ckpt_path: str | Path, rerun_cmd: str = "python scripts/credibility_gate.py --only train"):
    """Load an npz checkpoint, validating required keys and catching corruption."""
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    try:
        z = np.load(p, allow_pickle=False)
        files = set(z.files)
    except Exception as e:
        raise RuntimeError(
            f"Checkpoint file '{p}' is corrupt or unreadable: {e}. "
            f"To restart cleanly, rerun: {rerun_cmd}"
        ) from e

    if "arr" not in files or "step" not in files:
        missing = {"arr", "step"} - files
        raise RuntimeError(
            f"Checkpoint file '{p}' is corrupt (missing required keys: {missing}). "
            f"To restart cleanly, rerun: {rerun_cmd}"
        )

    arr = z["arr"]
    step = int(z["step"])
    rng_state = None
    if "rng_state" in files:
        try:
            rng_state = json.loads(str(z["rng_state"]))
        except Exception:
            pass

    opt_state_leaves = None
    if "opt_state_num_leaves" in files:
        num = int(z["opt_state_num_leaves"])
        leaves = [z[f"opt_state_{i}"] for i in range(num) if f"opt_state_{i}" in files]
        if len(leaves) == num:
            opt_state_leaves = leaves
    elif "opt_state_0" in files:
        leaves = []
        i = 0
        while f"opt_state_{i}" in files:
            leaves.append(z[f"opt_state_{i}"])
            i += 1
        opt_state_leaves = leaves

    return {
        "arr": arr,
        "step": step,
        "rng_state": rng_state,
        "opt_state_leaves": opt_state_leaves,
    }


def train(base: Constants, M, Y, turns=None, moves_m=None, mask=None, expert_idx=None,
          steps: int = 200, lr: float = 3e-3, fix_G: bool = False,
          batch_size: int = 256, seed: int = 0, tau: float = 2.0,
          margin: float = 0.0, key=None, use_multiverse: bool = False,
          K: int = 8, sigma: float = 0.1, log_every: int = 0,
          ckpt_every: int = 0, ckpt_path: str | None = None,
          init_arr=None, init_step: int = 0,
          init_opt_state=None, init_rng_state=None) -> Constants:
    """Mini-batch Adam over (M, Y, turns, moves_m, mask, expert_idx).

    M (N,64) mass vectors; Y (N,) outcomes {-1,0,+1} White-view; turns (N,)
    0=White/1=Black; moves_m (N,K,64) child mass vectors; mask (N,K);
    expert_idx (N,).  Outcome-only training: pass moves_m=None.

    Adam (lr≈3e-3) + gradient clipping (max_norm=1.0) + projected params back
    into physical bounds after every step.

    use_multiverse: score each child under the Layer-2 Bayesian average over K
    posterior realizations of the physics (the "Multiverse"). Requires `key`.

    Resilience: `log_every` prints step/loss/ETA (flushed); `ckpt_every` +
    `ckpt_path` save an atomic `{arr, step, opt_state, rng_state}` npz a killed
    run resumes from via `init_arr`/`init_step`/`init_opt_state`/`init_rng_state`.
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
    if init_rng_state is not None:
        if isinstance(init_rng_state, (str, bytes)):
            rng.bit_generator.state = json.loads(init_rng_state)
        elif isinstance(init_rng_state, dict):
            rng.bit_generator.state = init_rng_state

    base_key = jrandom.PRNGKey(seed) if key is None else key
    arr = _to_arr(base) if init_arr is None else jnp.asarray(init_arr, dtype=jnp.float32)
    if fix_G:
        arr = arr.at[0].set(1.0)
    t_start = time.perf_counter()

    if _HAS_OPTAX:
        # ── Adam + gradient clipping (preferred) ─────────────────────────────
        # Clipping max_norm=1.0 prevents a single noisy batch from launching
        # roche or eps to an extreme value in one step.
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
        )
        if init_opt_state is not None:
            if isinstance(init_opt_state, list):
                _, treedef = jax.tree_util.tree_flatten(opt.init(arr))
                opt_state = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in init_opt_state])
            else:
                opt_state = init_opt_state
        else:
            opt_state = opt.init(arr)

        @jax.jit
        def _step(a, state, bM, bY, bMM, bEI, bMK, bT, step_key):
            val, g = jax.value_and_grad(loss_fn)(
                a, bM, bY, bMM, bEI, has_policy, bMK, bT, tau, margin,
                step_key, use_multiverse, K, sigma)
            if fix_G:
                g = g.at[0].set(0.0)
            updates, new_state = opt.update(g, state)
            new_a = optax.apply_updates(a, updates)
            new_a = jnp.clip(new_a, _LO, _HI)
            if fix_G:
                new_a = new_a.at[0].set(1.0)
            return val, new_a, new_state
        for step in range(init_step, steps):
            perm = rng.permutation(N)
            step_key = jrandom.fold_in(base_key, step) if use_multiverse else None
            for s in range(0, N, batch_size):
                idx = jnp.asarray(perm[s:s + batch_size], dtype=jnp.int32)
                val, arr, opt_state = _step(
                    arr, opt_state,
                    M[idx], Y[idx], moves_m[idx],
                    expert_idx[idx], mask[idx], turns[idx], step_key,
                )
            if log_every and (step + 1) % log_every == 0:
                el = time.perf_counter() - t_start
                done, total = step + 1 - init_step, steps - init_step
                print(f"[train] step {step + 1}/{steps}  loss={float(val):.4f}  "
                      f"elapsed={el:.0f}s eta={el / max(1, done) * (total - done):.0f}s",
                      flush=True)
            if ckpt_every and ckpt_path and (step + 1) % ckpt_every == 0:
                ckpt_dict = {
                    "arr": np.asarray(arr),
                    "step": np.int64(step + 1),
                    "rng_state": np.array(json.dumps(rng.bit_generator.state)),
                }
                leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(opt_state)]
                ckpt_dict["opt_state_num_leaves"] = np.int64(len(leaves))
                for i, l in enumerate(leaves):
                    ckpt_dict[f"opt_state_{i}"] = l
                atomic_save_npz(ckpt_path, **ckpt_dict)

    else:
        # ── SGD fallback (no optax) ───────────────────────────────────────────
        # Manual gradient-norm clip + projected SGD.
        @jax.jit
        def _step_sgd(a, bM, bY, bMM, bEI, bMK, bT, step_key):
            val, g = jax.value_and_grad(loss_fn)(
                a, bM, bY, bMM, bEI, has_policy, bMK, bT, tau, margin,
                step_key, use_multiverse, K, sigma)
            if fix_G:
                g = g.at[0].set(0.0)
            # clip gradient norm to 1.0
            gnorm = jnp.sqrt(jnp.sum(g ** 2)) + 1e-9
            g = jnp.where(gnorm > 1.0, g / gnorm, g)
            return val, g

        for step in range(init_step, steps):
            perm = rng.permutation(N)
            step_key = jrandom.fold_in(base_key, step) if use_multiverse else None
            for s in range(0, N, batch_size):
                idx = jnp.asarray(perm[s:s + batch_size], dtype=jnp.int32)
                val, g = _step_sgd(
                    arr,
                    M[idx], Y[idx], moves_m[idx],
                    expert_idx[idx], mask[idx], turns[idx], step_key,
                )
                arr = arr - lr * g
                arr = jnp.clip(arr, _LO, _HI)
                if fix_G:
                    arr = arr.at[0].set(1.0)
            if log_every and (step + 1) % log_every == 0:
                el = time.perf_counter() - t_start
                done, total = step + 1 - init_step, steps - init_step
                print(f"[train] step {step + 1}/{steps}  loss={float(val):.4f}  "
                      f"elapsed={el:.0f}s eta={el / max(1, done) * (total - done):.0f}s",
                      flush=True)
            if ckpt_every and ckpt_path and (step + 1) % ckpt_every == 0:
                atomic_save_npz(ckpt_path,
                                arr=np.asarray(arr),
                                step=np.int64(step + 1),
                                rng_state=np.array(json.dumps(rng.bit_generator.state)))

    return _from_arr(arr)

def train_examples(base: Constants, examples, steps: int = 200, lr: float = 3e-3,
                   fix_G: bool = False, batch_size: int = 128, seed: int = 0,
                   val_frac: float = 0.2, verbose: bool = True,
                   tau: float = 2.0, margin: float = 0.0,
                   key=None, use_multiverse: bool = False,
                   K: int = 8, sigma: float = 0.1, policy: bool = True,
                   return_metrics: bool = False, log_every: int = 0,
                   ckpt_every: int = 0, ckpt_path: str | None = None,
                   init_arr=None, init_step: int = 0,
                   init_opt_state=None, init_rng_state=None):

    """Convenience: build arrays from examples (list of dicts), split train/val,
    train, and report validation ranking metrics so we can see real progress
    (not just overfitting on the training set).

    Laptop defaults: batch_size=128 (keeps a step's working set < ~1 GB of the
    8 GB total), val_frac=0.2 (held-out check).  tau=2.0 softens the policy
    softmax; margin>0 adds pairwise margin-ranking (easier than sharp CE).
    use_multiverse: train the Layer-2 Bayesian-average score.
    return_metrics: also return {"baseline": ..., "trained": ...} policy
    metrics on the held-out validation split (used by the self-play loop as an
    accept/reject gate).
    """
    if not examples:
        raise ValueError(
            "train_examples received an empty example list. "
            "Check that puzzle_examples / game_examples loaded data correctly "
            "(CSV path, column names, move format)."
        )
    M, Y, turns, moves_m, mask, expert_idx, expert_cap = to_arrays(examples)
    n = int(M.shape[0])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int((1.0 - val_frac) * n)
    tr, va = perm[:split], perm[split:]

    train_moves = moves_m[tr] if policy else None
    train_mask = mask[tr] if policy else None
    train_expert = expert_idx[tr] if policy else None
    trained = train(
        base, M[tr], Y[tr], turns[tr], train_moves, train_mask, train_expert,
        steps, lr, fix_G, batch_size, seed, tau=tau, margin=margin,
        key=key, use_multiverse=use_multiverse, K=K, sigma=sigma,
        log_every=log_every, ckpt_every=ckpt_every, ckpt_path=ckpt_path,
        init_arr=init_arr, init_step=init_step,
        init_opt_state=init_opt_state, init_rng_state=init_rng_state,
    )

    metrics = None
    if policy:
        from .loss import policy_metrics
        Mv, Yv, tv, mmv, mkv, eiv, capv = (
            M[va], Y[va], turns[va], moves_m[va], mask[va], expert_idx[va],
            expert_cap[va])
        b = policy_metrics(base, Mv, Yv, tv, mmv, mkv, eiv, capv)
        t = policy_metrics(trained, Mv, Yv, tv, mmv, mkv, eiv, capv)
        metrics = {"baseline": b, "trained": t}
        if verbose:
            print(f"[train] N={n}  train={split} val={n - split}  "
                  f"steps={steps} lr={lr} tau={tau} margin={margin}")
            print(f"[train] baseline  top1={b['top1']:.3f}  mrr={b['mrr']:.3f} "
                  f"cap={b['mrr_capture']:.3f} quiet={b['mrr_quiet']:.3f}")
            print(f"[train] trained   top1={t['top1']:.3f}  mrr={t['mrr']:.3f} "
                  f"cap={t['mrr_capture']:.3f} quiet={t['mrr_quiet']:.3f}")
            print(f"[train] delta mrr              : {t['mrr'] - b['mrr']:+.3f}")
    elif verbose:
        print(f"[train] N={n}  train={split} val={n - split}  "
              f"steps={steps} lr={lr} outcome-only")
    if return_metrics:
        return trained, metrics
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

