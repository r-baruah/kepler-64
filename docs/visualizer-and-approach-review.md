# Kepler-64 — Visualizer & Approach Review

> Consolidated analysis, ideas, and prioritized recommendations for the visualizer
> (GIF banner + interactive surfaces) and the overall engine approach.
> Companion to `docs/visualizer_design_philosophy.md` and the `Kepler-64_Book/`.

**Scope.** This doc is advisory. It summarizes (1) what the project is today, (2)
the tension between the two visualization surfaces and the design-philosophy doc,
(3) how to make the "natural physics shading" idea stronger and more meaningful,
(4) the evolved/missed innovation parts worth adding, (5) a visual-design
elevation, and (6) prioritized "better / best other methods" for the engine
approach. No code changes are described as done; every recommendation is marked
with concrete inputs already present in the repo.

---

## 1. What the project actually is

A real, legal-chess engine whose **static evaluator is a differentiable N-body
gravitational field in JAX**:

- Pieces carry **mass** (pawn 1 … queen 9, **king 1000**).
- Positions feed Plummer-softened Newtonian gravity over the fixed 64-square
  distance matrix → a potential field `U` and force field `F`.
- King safety = the **tidal (Hessian) eigenvalue λ₁** at the king's square.
- λ₁ is scaled into a **dimensionless Roche disruption index** `η = Rg³·λ₁/m_ref²`.
- Score = `η_b − η_w + force-sigmoid tactical terms + field-energy edge + material`.
- ~9–13 constants (`G, ε, c, ρ_roche, Rg, bonus, kgain, gamma, mat_gain…`) are
  learnable leaves optimized through JAX autodiff.
- Search = conventional alpha-beta + quiescence; board = pure-NumPy `FastBoard`
  (no python-chess in the hot path); all candidates scored in one **218-pad
  `vmap`** sweep.

**The project's biggest asset:** its honesty discipline. The `Kepler-64_Book/`
(esp. `14_Review.md`) logs its own bugs, gaps, and the "not real astrophysics"
boundary. That is exactly the kind of self-aware framing HN/Reddit reward.

---

## 2. The current visualization landscape (and a key discovery)

There are **three** visualization surfaces in the repo, and they do **not** agree:

| Surface | What it is | Real data? |
|---|---|---|
| `kepler64/viz/glassbox.py` + `replay.py` | Python → matplotlib two-panel: board + **potential heatmap with geometric tidal ellipses / red λ₁ arrows** → GIF / HTML / PNG | ✅ Real engine output |
| `viz-web/` (React/Vite) | Mock: hardcoded FEN, a **static `overlay.png`**, and **fake metrics** (`+0.4`, `King Danger: High`, `G 1.0`, `c 4.0`) | ❌ Placeholder — not wired to the engine |
| `design-concept/` | Static HTML/JS instrument fed by **real replay data** via `scripts/build_data.py` | ✅ Real (frontend-only) |

**Important realization:** `docs/visualizer_design_philosophy.md` is effectively a
*critique of the geometric visualizer*. It argues the tidal ellipses, arrows, and
last-move arrows are "harsh geometric overlays," and that danger should instead be
expressed as **natural physics-driven shading** — a dark "king's shadow" that
radiates smoothly and lets the two kings' fields resolve "conflict zones" by the
real gravitational math. With one refinement (below), this is the better idea and
is *more* on-theme.

Two problems:

1. **The React app meant to realize this philosophy is a static mock** with no
   engine connection. For a *technical* audience this reads as "a fake engine with
   a UI" — a credibility risk. Wire it to the real engine (either server-side
   compute or precomputed replay via `design-concept`).
2. **It contradicts your own `DESIGN.md`** ("Observatory Ephemeris": light mineral
   field, ultramarine plates, one decisive trajectory-orange, explicitly *no*
   neon-on-black HUD, no glows). The `viz-web` `gif-canvas` is dark (`#0f172a`)
   with neon-ish accents.

---

## 3. Making the philosophy idea stronger and more meaningful

### 3.1 Render danger by **η**, not by raw potential
The philosophy doc maps shading to the *field* (potential depth). But the number
that actually drives engine decisions / checkmate is **η / ρ_roche** (tidal
disruption vs the learned Roche threshold). Drive the "king's shadow" from **η**,
not raw `U`. Three wins:

- **Fixes a real bug.** The Book flags `C12 / VIZ-ETA-MISMATCH`: the visualizer's
  η uses the old denominator `G·M_king²` while the evaluator uses `m_ref²`, so the
  red "danger" can **contradict the engine's own score**. Unify, and visual ==
  objective.
- **Makes the visual meaningful to every audience:** founders see "the king
  shadows as it's attacked"; chess players see "a danger/territory map"; technical
  readers see "the visual *is* the loss surface."
- **Grounds the "conflict zone" you want.** Add a **battlefront line**: the
  saddle/ridge where ∇(net potential) = 0, separating White's basin from Black's
  basin (who controls which territory), resolved automatically by the real field —
  exactly your "the physics resolves the conflict zones" requirement.

### 3.2 Render direction as *flow*, not as an arrow
Replace the harsh red λ₁ "line of failure" with a **line-integral-convolution
(LIC)-style oriented texture** — a streaked field aligned to the tidal
eigenvector, whose intensity/color maps η. Same information, but reads as a
natural liquid/plasma flow rather than a geometric glyph.

### 3.3 The "mood map" formula (concrete)
Compose the frame from physics channels, not from UI glyphs:

```
channel   signal                 visual  ->   meaning
U         potential depth        bathymetry   concentration / wells
η/ρ_roche disruption index       king shadow  actual danger (aligns to score)
λ₁ eigvec direction of stretch   LIC flow     which axis the king is being pulled
battlefront ∇U=0 ridge           neutral line who controls the center
```

---

## 4. Evolved + missed innovation parts (cheap, high-value, on-theme)

These map to mechanisms that already exist in code but are either unwired or not
made visible. The visualizer is the ideal place to surface them.

1. **★ Light-cone / retarded potential → make the speed of light visible.**
   The `c`-gate `σ(c−d)` means gravity does *not* arrive instantly. Animate an
   expanding wavefront rate `c` squares/ply after a move — distant pieces "haven't
   arrived yet." This single animation tells the entire "speed of light in
   squares/ply" story, which today is a dead number in the UI. **Nobody has seen a
   chess engine's evaluation ripple.** High shareability on its own.
2. **Verlet future-orbit (dη/dt).** `core/verlet.py` already leapfrogs the king
   forward; docs promise "impending collapse" via dη/dt but it's unwired
   (C11 / C17). Draw a faint dotted **orbit** of where the king *would* drift a few
   steps + a tiny η-trend glyph. Projects movement instead of a snapshot.
3. **★ Multiverse fan (Layer 2).** Render the *same* move under K=8 sampled
   constants (`multiverse/posterior.py`) as a faint **fan of slightly-differing
   fields / an uncertainty band**. Very on-theme ("the engine doesn't know which
   universe it's in, so it hedges") and uniquely interesting to technical readers.
   Be honest that posterior sampling isn't the default play path.
4. **Accretion.** On a capture, show the captured **mass merging into the captor**
   and — once `C13` is fixed — the captor growing larger *and more fragile*.
   Completes the cause-and-effect visual story.
5. **Candidate decision-ribbon** (from `analysis/recorder.py` + replay): show which
   candidate moves the engine considered and how they ranked — the "proof it's a
   real engine" element. Keep it; the `design-concept` instrument already does this.

---

## 5. Elevating the visual design

- **Adopt the light "Observatory Ephemeris" language, not the dark HUD.** It is
  more distinctive, more legible against HN/Reddit white backgrounds, and already
  what DESIGN.md commits to. Drop the `#0f172a` dark `gif-canvas`.
- **Keep 1200×630** (correct social-share ratio) and `loop=0` GIF.
- **Layering:** legible board → low-opacity field → faint η "shadow" → **one orange
  accent for the decision** (the selected move's path). One global accent, not many.
- **Correctness of labels:** native scores shown first as native Kepler values, not
  centipawns; matching the reviewer-facing honesty in PRODUCT.md.
- **Accessibility:** preserve reduced-motion + GIF-behind-disclosure (already in
  `replay.py` and DESIGN.md).
- **Banner narrative (from the earlier discussion):** one continuous shot, three
  beats — (1) familiar board + a real move (2s), (2) board → gravitational field
  reveal (the hero moment), (3) the "it's not a metaphor" proof: learned constants +
  a king under tide / candidate ranking. ~8–12s loop. Pick **one** move with a strong
  king-attack story so the η shadow actually flares.

---

## 6. Better / best *other* methods for the overall approach

**Bottleneck = credibility, not physics, and not visuals.**

### Phase 1 — the credibility gate (do this before anything else)
1. **Ship the ablation table (C1).** Train learned-constants vs `G=1` fixed and
   publish Elo/MRR delta. Converts "clever ironized chess" into "demonstrated."
   `Δ < 50` Elo ⇒ the physics isn't doing the work — better to know now.
2. **Scale the data (C9).** 5k+ puzzles / 10k+ positions, two-phase training,
   validation split, Adam + cosine LR. Current run overfit 378 examples and drove
   `roche`/`gamma` to harmful extremes.
3. **Stop overclaiming G (C10).** G is largely non-identifiable (cancels in η).
   Report which constants actually moved. Note: `trained_constants.json` shows
   `gamma=0.0`, `roche` at floor `0.05`, `mat_gain=0.3` → today the engine is
   "material + small physics." Tell that story honestly and fix it mechanically.

### Phase 2 — physics refinements (better methods)
4. **Use the 2D Green's function** (`∝log(r)` / `1/r`), not 3D `1/r²`. The Book
   itself admits 2D space wants `1/r`; try it as an ablation (more honest + possibly
   better conditioned).
5. **Make retardation real**: replace the static `σ(c−d)` gate with a retarded
   Green's function over a history buffer via `jax.lax.scan` — faithful *and* makes
   the light-cone animation true.
6. **Peters–Mathews gravitational-wave energy loss** (∝ G⁴m₁²m₂²/r⁵) as an
   anti-shuffling/repetition penalty. **And/or** a Schwarzschild-radius clustering
   penalty ("don't pile heavy pieces into one event horizon").
7. **Finish dynamic `Rg` (C13, Eddington fragility)**: accreted kings grow larger
   *and easier to tear*; completes the physics cause-and-effect and gives the visual
   a narrative.

### Phase 3 — best / long-run architecture
8. **Neural-physics hybrid, physics-first**: keep η/tidal/material terms as the
   interpretable feature vector (the glass box) and let a small value head fit only
   the *residual*. Preserves PRODUCT.md's "no opaque policy silently replacing
   physics" while making the engine competitive.
9. **AlphaZero-style self-play + TD bootstrapping through the differentiable
   physics** (outcome + policy + TD value targets) — how to actually learn strong
   constants.
10. **Efficient many-body** (hierarchical/multipole) to scale the 218-pad sweep and
    deeper search; proper benchmark with p50/p99 and honest scope ("evaluation
    sweep," not end-to-end); **BayesElo ladder** vs reference engines.

---

## 7. Priority order (TL;DR)

1. Fix **C12**: unify the visualizer's η with the evaluator's; drive visual danger
   from the real score.
2. Replace geometric glyphs with **natural field shading + battlefront line**;
   render λ₁ direction as flow (LIC).
3. Animate the **light-cone** (retarded potential), the **Verlet future-orbit**,
   **accretion**, and the **Multiverse fan** — the evolved/missed visuals.
4. Reconcile the design language to the light **"Observatory Ephemeris"** (drop the
   dark HUD); keep 1200×630, single orange accent, reduced-motion.
5. Gate the *engine* thesis with the **ablation** + **bigger data** before any of
   the above matters for credibility.

---

## 8. Concrete follow-ups available in this repo

If implementation is wanted, these are the smallest high-value changes, all within
the existing Python pipeline (`kepler64/viz/`):

- **C12 η unification**: expose the evaluator's `η` (with `m_ref`) from
  `core/evaluate.py` / `core/tidal.py` and use it in `viz/glassbox.py`'s
  `_eta_from_U64` and disruption coloring.
- **Danger-by-η overlay + battlefront line**: add a field channel computed from η
  and the ∇U=0 ridge, rendered as additive shading in `render_field`.
- **Light-cone animation pass**: a small script that renders N frames with an
  expanding `c`-rate wavefront around the moved piece, composited into a GIF.

> Nothing in this section is implemented yet; it is a proposal for follow-up work.


