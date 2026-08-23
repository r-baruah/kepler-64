# Chapter 8 — The 218-Pad Trick and the Sub-Millisecond Sweep

> **Read me first (no background needed).** This chapter is pure speed engineering, and its hero idea is easy to picture. A compiler like XLA is a factory that builds a custom machine for each *shape* of problem you feed it. Chess ruins that: one position has 30 legal moves, the next has 5, so the factory would rebuild the machine after every move. The fix: always deliver exactly 218 problems — chess's theoretical maximum number of legal moves — padding empty slots with harmless dummies. The factory builds once; every position rides the same machine. Empty seats cost almost nothing.

## 8.1 Intuition: why variable batch sizes kill compilers

JAX/XLA compiles a function into very fast machine code — but only after it has seen the **exact shape** of the tensors involved. Chess positions have a *variable* number of legal moves (anywhere from 0 to 218). If you re-compile for every different move count, the "sub-millisecond" promise evaporates into constant recompilation (the "silent recompilation trap," `Kepler-64.md` §2).

The trick: **always pad to the theoretical maximum of 218 legal moves**, filling unused slots with **zero-mass dummy pieces**. XLA then traces one fixed shape `(218, 64, 2)` exactly once, and every later sweep reuses that single compiled kernel.

> **Intuition box:** Like loading a bus that seats 218 even when only 30 passengers show up — the bus is always the same size, so the depot never rebuilds it. Empty seats (dummy masses) cost almost nothing.

## 8.2 The math: batching and masking

For a position with $n \le 218$ legal moves, build a buffer $B \in \mathbb{R}^{218\times 64}$: the first $n$ rows are the child mass vectors, the remaining $218-n$ rows are zeros. Apply `score_white` to all 218 rows via `jax.vmap` (vectorized map), then **mask** the dummy rows to `−∞` so they can never win a node (`core/evaluate.py:137-151`):

```python
buf = jnp.zeros((pad, 64), ...)
buf = buf.at[:n].set(jnp.stack([child mass vectors]))
white = jax.vmap(score_white, in_axes=(0, None))(buf, constants)  # (218,)
side  = jnp.where(turns == 0, white, -white)
mask  = jnp.concatenate([ones(n), zeros(pad-n)])
return jnp.where(mask > 0.5, side, -jnp.inf)
```

The dummy rows score *something* (all-zero masses → near-zero field) but are forced to `−inf`, so `argmax` over the 218 never picks them.

**Why 218?** 218 is the proven theoretical maximum number of legal moves in any chess position (a specific configuration with many pawn promotions). Padding to 218 guarantees the buffer always fits (`README.md` "218-pad trick", `core/evaluate.py:32`).

## 8.3 The sub-millisecond claim — what it honestly means

The claim is: **one `vmap` sweep over all 218 candidate moves runs in sub-millisecond** (on XLA-compiled JAX, CPU or GPU). This is *true and reproducible* via `bench/sweep_time.py` (`Kepler-64.md` §2, `Kepler-64 Scaffold.md` "rigor"). What it does **not** mean is that *end-to-end move latency* is sub-millisecond — that includes move generation and board updates, which (in the v1 adapter) cross into python-chess (`Kepler-64 Audit` counter-audit §D). The pure-JAX `FastBoard` (Chapter 12) removes that tax.

> **⚠ [ISSUE: SUBMS-SCOPE] (P2, Audit counter-audit §1):** Do **not** say "the engine thinks in sub-millisecond." Say "the evaluation *sweep* is sub-millisecond; end-to-end latency is dominated by move generation at MVP and removed by the pure-JAX board." The benchmark `sweep_time.py` times only `batch_score`, which is the honest scope. Report p50/p99, not just a best-case number.

## 8.4 Efficiency notes on the batch path

- ~~**`batch_score` loop (BUG 9)**~~ **(Resolved):** children are stacked once into a single buffer, then padded — no per-move intermediate arrays.
- ~~**double potential (BUG 10)**~~ **(Partially open):** `_score_terms_body` still computes force and potential fields separately; they could share one `r2`/`sqrt_r2`. Minor redundancy.
- **root sweep (the headline):** `root_sweep` (`search/minimax.py`) builds all child mass vectors via `child_mass_vector` and calls `batch_score` once — this is the showcased 218-pad sweep.

## 8.5 Forward link

The sweep feeds an alpha-beta search. The next chapter covers the board/search layer (how moves are generated and how the score is used in negamax + quiescence), including the pure-JAX board that makes the claim fully true.

**Cross-references:** Mass vector → §7. Search & alpha-beta → §10. Pure-JAX `FastBoard` → §10.2. Benchmark → `bench/sweep_time.py`.
