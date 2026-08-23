# Chapter 10 — The Board and the Search Tree

## 10.1 Intuition: physics evaluates, chess provides the moves

The gravitational pipeline only *scores* a position. Something else must *generate the legal moves* and *decide which to play*. That "something" is the board + search layer. Two jobs:

1. **Move generation** — enumerate legal moves (python-chess rules: castling, en passant, check evasion).
2. **Search** — explore the tree of moves a few plies deep (alpha-beta / negamax) and pick the best, using the Roche score as the leaf value.

> **Intuition box:** The physics is the judge scoring each snapshot; the search is the player trying moves and asking the judge which line ends up best.

## 10.2 The board: two implementations

| | `Board` (`core/board.py`) | `FastBoard` (`core/fastboard.py`) |
|---|---|---|
| Backing | wraps `python-chess` | pure NumPy `int8[64]` |
| Move gen | delegates to python-chess | **own** generator, verified vs python-chess |
| Speed | "v1 tax" (Python/C++ boundary) | no python-chess in hot path |
| Role | adapter / training data | search hot path |

`FastBoard` is the key engineering win: it generates fully legal moves in pure NumPy (pseudo-legal generation + a king-safety filter, `core/fastboard.py:164-280`), castling, en passant, promotion — and is **verified against python-chess** via `tests/test_board.py` (`Kepler-64 Scaffold.md` "why brilliant"). This is what makes the sub-ms-sweep claim fully true end-to-end (`Kepler-64 Audit` counter-audit §D).

> **⚠ [ISSUE: IMPORT-PATHS] (P0, Code Review v1 BUG 9 & BUG 24):** Earlier `multiverse/`, `training/`, and `search/` files used single-dot relative imports (`from .core.constants`) where `core` is a *sibling* package, not a parent. Correct is `from ..core.constants`. This broke every import outside the package. The current files use `..core` throughout — verified clean by a repo-wide sweep on 2026-08-23 (`multiverse/posterior.py:18`, `search/minimax.py:29`, `training/loss.py:27`).

> **⚠ [ISSUE: FRAGILE-IMPORT] (P2, Code Review v1 BUG 10 / v2 ISSUE 23):** `board.py:84` uses `__import__("chess").WHITE` inside `_color_at` instead of the already-imported `chess` module. It works but is slow/fragile. `mass_vector()` (the hot path) avoids it by precomputing colors. Low priority but clean it up.

## 10.3 The math: alpha-beta negamax

The search uses **negamax** with **alpha-beta pruning** — the standard, optimal-play tree search. In negamax, the score is always from the perspective of the side to move:

$$\text{value}(s) = \max_{m} \big(-\text{value}(\text{child}(s,m))\big)$$

Alpha-beta maintains two bounds: `alpha` (best score the maximizing side can guarantee) and `beta` (best the minimizing side can force). When `alpha ≥ beta`, the rest of the branch is pruned — it can't improve the result. This cuts the explored tree dramatically without changing the answer (`core/search/minimax.py:44-69`).

**Terminal:** actual checkmate returns `−MATE` for the side to move. Stalemate/end returns 0.

**Depth enhancements (2026-08).** The search now uses:
- **Late move reductions (LMR)** — late quiet, non-checking moves are probed at reduced depth first and only re-searched full-width if they climb into the window;
- **Root aspiration windows** — from depth 3 the previous iteration's score seeds a narrow window, with a full-width re-search on fail (a pure time-saver, never a change in verdict);
- **Fail-soft cutoffs** — quiescence and null-move return the actual value rather than the flat `alpha`/`beta` bound;
- **Break-on-mate** — a node stops as soon as a forced mate is found, since flat mate scoring means nothing scores higher.

The fail-soft + break-on-mate pair eliminated a **false-mate bug**: once a real mate pushed `alpha` to the mate ceiling, a zero-window probe's `beta` became `−MATE`, so every later quiet move "failed high" at `−MATE` and negated to a false `+MATE` — the multiverse head could then pick a queen-hanging move as if it were mate.

## 10.4 Quiescence search (resolving captures and checks)

A fixed-depth search can stop at a volatile moment (just before a capture), misreading the position. **Quiescence** extends the search along forcing lines so the eval is read at a "quiet" position (`core/search/minimax.py:72-114`). It expands **captures AND quiet checking moves** — a quiet move that gives check is forcing, and ignoring it at the horizon is the classic way a search hangs a piece to a discovered check. When the side to move is in check, all legal evasions are considered. Crucially, **all capture children of a node are scored in ONE `batch_score` vmap** (the 218-pad sweep, §8) — so the cost per node is a single XLA call, not one score per capture. This is what keeps the search fast.

## 10.5 Accretion wired into search (Layer 2 preview)

When a capture occurs, the search absorbs the captured piece's *original* mass onto the captor via `child_mass_vector` (`core/transitions.py`) **before** scoring. This matters because the captured mass is already 0 in the child board, so it must be read from the **parent**. This is the correct fix for Code Review v2 BUG 2 (earlier code used the post-move — already-zero — mass). The accretion physics itself lives in Chapter 13.4.

## 10.6 Move ordering and bottlenecks

`negamax` sorts moves captures-first (`core/search/minimax.py:55`). Move ordering is native FastBoard: TT move first, then MVV-LVA captures from raw masses, PV/killer/history hints, and — only as a tiebreak among quiet, unhinted moves — the gravitational potential well (moving into a deep friendly well first). No python-chess call sits anywhere in the ordering path.

> **(Resolved — removed)** The former [ISSUE: OPENING-BOOK-MISSING]: the `search/openings.py` stub was **deleted outright** (2026-08-23), not implemented. This is a deliberate vision decision, not deferred work: Kepler hands the universe only the laws of physics and the rules of the game — a lookup table of pre-chosen openings would violate the core premise ("let the universe figure out how to play"). The engine opens from the primordial position, always. If a "quasi-equilibrium" idea returns, it must be *learned* (sampled from low-energy states the universe itself discovers), never hardcoded.

## 10.7 Forward link

Next: the Glass Box visualizer (Chapter 11), which turns the physics into the GIF that proves the math works — the project's highest-ROI artifact.

**Cross-references:** 218-pad sweep used here → §8. Accretion physics → §13.4. Mass vector → §7.
