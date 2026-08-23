# Kepler-64 Backend Audit & Next-Phase Roadmap

**Date:** 2026-08-23 · **Scope:** full `kepler64/` backend audited against the Kepler-64 Book and the core vision: *"Let the universe figure out how to play."*
**State at audit:** tests passing (spot-verified: constants + tidal suites green), engine stable at commit `df7f825`.

---

## Part 1 — Architecture Map (what exists)

```
kepler64/
├── core/
│   ├── constants.py     14 trainable leaves (G, eps, c, roche, bonus, kgain,
│   │                    gamma, Rg, mat_gain, 4 delta gains, lambda_drift) +
│   │                    bounds, packing, JSON persistence. PIECE_MASSES.
│   ├── board.py         v1 adapter around python-chess -> mass vector
│   ├── fastboard.py     pure-NumPy legal move gen (search hot path),
│   │                    per-piece velocity accumulator (Lorentz boost)
│   ├── gravity.py       static (64,64) geometry; Plummer force/potential
│   │                    with sigmoid(c - d) reach gate
│   ├── tidal.py         edge-padded FD Hessian at king, closed-form 2x2
│   │                    eigenvalues, eta = Rg^3·λ1/mref²
│   ├── lorentz.py       γ = 1/√(1-u²), u=v/(v+c)  (wired via FastBoard)
│   ├── verlet.py        standalone leapfrog rollout  ← DEAD CODE (see A7)
│   ├── transitions.py   child_mass_vector: accretion + unboost/reboost
│   ├── image_seed.py    photo → FFT → initial constants
│   └── evaluate.py      THE LAWS: η pair, army-sourced disruption sigmoid,
│                        binding γ, material edge, 4 delta terms, Verlet
│                        dη/dt drift, Shannon entropy, batch_score (218-pad
│                        vmap), Layer-2 multiverse posterior mean
├── search/minimax.py    PVS negamax, iterative deepening, aspiration, TT
│                        (Zobrist ⊕ mass checksum), null-move, LMR, killers,
│                        history, quiescence (captures+checks+evasions),
│                        closed-orbit draw rule, root multiverse tie-break
├── search/openings.py   stub returning None  ← conflicts with vision (A11)
├── training/            loss (outcome CE + policy CE/margin + c-prior),
│                        Adam loop, puzzle/PGN loaders, self-play + Elo gate,
│                        ablation harness
├── multiverse/          accretion (2C), posterior re-export, observer (2B),
│                        fluid (conditional)  ← observer & fluid UNWIRED (A8)
├── bench/, match/, viz/, analysis/    harnesses, UCI, glass-box, replay
```

Data flow per leaf node: `FastBoard → mass_vector() [+Lorentz] → child_mass_vector (accrete, unboost/reboost) → score_white (gravity → tidal → η → gauges → deltas) → negamax`.

---

## Part 2 — Physics Audit (law by law)

### What is genuinely correct and rigorous ✅
1. **Plummer gravity + einsum kernel** (`gravity.py`): correct F ∝ m·r̂/(r²+ε²)^{3/2}; static geometry computed once; batch-safe; |m| enforced at entry (no sign-repulsion bug).
2. **Reach gate as retarded potential**: sigmoid(c − d) is a clean, differentiable stand-in for finite propagation speed; c prior-bounded [1,10] with a *differentiable* penalty in the loss (old `max()` gradient bug fixed).
3. **Tidal tensor**: edge-padded second differences fix the ~25%-of-squares boundary bug; file/rank axes correctly ordered; closed-form symmetric-2×2 eigenvalues are exact.
4. **η scaling**: G correctly cancels (Hessian already ∝ G); mref fixed unit scale keeps η ∈ [0.05 … 1+]; king cbrt floor prevents NaN gradients on padded rows.
5. **King-as-detector discipline**: kings excluded from attacking fields everywhere — the old ±150 saturation bug is genuinely dead.
6. **Army-sourced force gauge**: sigmoid(kgain·(‖F_army‖−roche)) sits near zero in openings and rises only under real attacks.
7. **Delta terms**: all difference-form (child − parent), mass sums without denominators — the free-rook/+240-advance bug class is structurally eliminated.
8. **Mass conservation through captures**: accretion absorbs rather than deletes; unboost-before-reboost kills the compounding double-Lorentz bug; castling carries rook accretion; en passant handled.
9. **Layer-2 multiverse**: plain equal-weight mean over K perturbed realizations = honest Bayesian model average (softmax-hedging bug fixed); deterministic seed keeps play reproducible.

### Findings

| ID | Sev | Finding |
|----|-----|---------|
| **A1** | High | **Two speeds of light.** The Lorentz boost hardcoded `u = v/(v+6.0)` in `fastboard.mass_vector` and twice in `transitions.child_mass_vector`. The learned leaf `c` did **not** govern relativistic mass — two different light speeds in one universe. ✅ RESOLVED 2026-08-23: c threaded through both functions at every search call site (`LORENTZ_C_DEFAULT = 6.0` remains only as fallback for ad-hoc callers). |
| **A2** | High | **Verlet rollout ignored the reach gate.** `_eta_drift` integrated forces **without** sigmoid(c − d) while static η terms used gated fields — dynamics simulated a different universe than the score. ✅ RESOLVED 2026-08-23: gate applied; drift → 0 for distant attackers at small c (tested). |
| **A3** | Med | ~~**Velocity never decays, ignores distance.~~** CORRECTION: decay already existed (×0.985/ply); what was missing was **distance weighting** (`+1.0` per hop regardless of slide length). ✅ RESOLVED 2026-08-23: velocity now gains true lattice distance; decay named `VELOCITY_DECAY`; leaf-ification of the decay deferred (renumbers trainable leaves — do with next training-phase change). |
| **A4** | Med | **Promotion destroys accreted mass.** Promotion resets mass to `_MASS_LUT[promo] + accretion*captured`, discarding the pawn's previously accreted surplus. ✅ RESOLVED 2026-08-23 (surplus carried forward, tested). |
| **A5** | Med | **TT depth-preference broken for identical keys.** `store()` skipped overwrite only when `keys[slot] != h`; a same-key deeper entry got overwritten by a shallower one. ✅ RESOLVED 2026-08-23 (condition now depth-only; tested incl. tie-keeps-old). |
| **A6** | Low | **Null-move probe drops velocity state**, so null-probe masses differ from the real line's boosted masses. ✅ RESOLVED 2026-08-23 (velocity passed through). |
| **A7** | Low | **Dead/duplicated rollout.** `core/verlet.rollout` imported by nothing; `_eta_drift` reimplements leapfrog inline. Delete or deduplicate. |
| **A8** | High (narrative) | **Observer & fluid layers unwired.** `observe_update`, `stokes_flow` called by nothing. Book Ch.13.3 reads as if physics co-evolves during play — it does not. Latent bug: `observe_update` reconstructs Constants with only G/eps/c/roche, silently resetting the other 10 leaves to defaults. Wire behind a flag (fixing reconstruction first) or demote in the book with a STATUS block. |
| **A9** | Med (honesty) | **Captor fragility applies only to kings.** `accretion.apply_capture` returns `Rg_new` but no caller consumes it; eval's `Rg_eff` grows only from *king* mass. "Hoarding makes you fragile" is king-only today. Implement per-piece extent or rescope the book claim. |
| **A10** | Low (framing) | **Material = chess values in physics clothing.** Masses 1/3/3/5/9/1000 + mat_gain·Δmass are conventional piece values with gravitational accent. Fine, but say plainly: material is the universe's baryon budget; scale learned, ladder inherited from chess. Current framing overstates how alien the universe is. |
| **A11** | Vision | **`search/openings.py` conflicts with the core vision.** ✅ RESOLVED 2026-08-23: module deleted; decision recorded in book Ch.10.6 — *"The universe starts from the primordial position; no book."* A quasi-equilibrium idea may return later as a learned sampler, never a lookup table. |

---

## Part 3 — Book ↔ Code Sync Ledger

| # | Book claim | Code reality | Action |
|---|-----------|--------------|--------|
| S1 | Ch.13.5: "γ uses u = v/(v+c)" | Hardcoded 6.0 (A1) | ✅ Code fixed 2026-08-23 — text is now true |
| S2 | Ch.13.3: Observer as live mechanism | Never called; resets leaves (A8) | Wire or add STATUS block |
| S3 | Ch.13.4: "hoarding makes you fragile" | King-only (A9) | Scope the sentence |
| S4 | Ch.12.5 artifact shows `mat_gain=0.3` era | Bounds now floor mat_gain at 1.0 | Refresh after next training run |
| S5 | Ch.14 C18 "opening book open" | Stub still present | ✅ Resolved 2026-08-23 — module deleted, C18 + §10.6 updated |
| S6 | Ch.14 C20 Observer-accretion feedback open | Moot until A8 decision | Fold into A8 |
| S7 | Ch.14 C21 `batch_score` Python-loop note | Now stacks once + pads | Mark resolved |
| S8 | Ch.10.6: "ordering calls python-chess" | FastBoard native now | Update prose |
| S9 | README "38 tests passing" badge | Suite grew since | Count in CI, link badge to workflow |
| S10 | Index promises file:line refs throughout | Several drifted after refactors | Sweep during sync pass |

---

## Part 4 — Missed Innovations (bigger levers not yet pulled)

1. **True causal gravity (the big one).** Today `c` gates a *static* field. Make influence actually propagate: a per-square "field age" spreading one Chebyshev ring per ply from every moved mass; squares beyond the c-frontier feel the *old* field. Then c decides whether your attack has physically *arrived*. The most on-vision upgrade possible.
2. **Gravitational-wave energy loss (Peters–Mathews)**: dE/dt ∝ G⁴m₁²m₂²/r⁵ pairwise term — principled anti-shuffle pressure replacing hand-tuned velocity tricks. Flagged in book Ch.14, still unimplemented.
3. **Schwarzschild-radius mechanic**: r_s = 2Gm/c² per piece; soft penalty inside each other's r_s — "don't pile supermassive bodies." Cheap, thematic.
4. **Per-piece extent vector (Rg per square)** — fixes A9 properly and makes "over-extended captor torn by a pawn" a real motif the engine fears.
5. **Time-dilation search scheduling**: spend more search where local |Φ|-curvature is high — physics-native replacement for generic extensions.
6. **Learned dt/steps for the Verlet horizon** (currently hardcoded 0.1×4) so training chooses how far ahead "impending collapse" looks.
7. **Posterior distillation**: fit posterior variance per leaf from self-play runs and report *which constants games actually constrained* — turns the Multiverse into measurable epistemics.
8. **UCI promotion of the identity**: `match/uci_harness.py` exists; shipping a runnable UCI binary makes "the universe plays chess" testable in any GUI.

---

## Part 5 — Improvement Plan (next days)

### Phase 0 — Hygiene (half day) — ✅ DONE 2026-08-23
- [x] A5 TT store condition fix (+ unit test: deeper entry survives).
- [x] A6 null-move velocity passthrough.
- [x] A7 delete `core/verlet.py` (dead code; `_eta_drift` in evaluate.py is the single rollout implementation).
- [x] A11 delete `search/openings.py`; vision decision note added to book Ch.10.6, Ch.14 C18 marked resolved.
- [x] Full pytest run: all suites green (22 transitions/tidal/board/audit-fixes, 17 evaluate/gravity/constants, all 6 search-strength verified individually).

### Phase 1 — One universe, one set of laws (1 day) — ✅ DONE 2026-08-23
- [x] A1 single speed of light: `FastBoard.mass_vector(c_lorentz=...)` and `child_mass_vector(..., c_lorentz=...)` now take the universe's c; every search call site passes `Constants.c`. Fallback constant `fastboard.LORENTZ_C_DEFAULT = 6.0` only for ad-hoc callers. Regression tests: boost monotonic in c; ply-consistency invariant `child_mass_vector == child.mass_vector(c)` for arbitrary c.
- [x] A2 gate rollout forces with sigmoid(c − d); property test: drift → <1e-6 at tiny c for a distant attacker.
- [x] A3 distance-weighted velocity gain (`v += true lattice distance`, incl. castling rooks 2/3). Decay already existed (×0.985/ply), now named `VELOCITY_DECAY`. Leaf-ification of the decay deferred to a later training-phase change (would renumber TRAINABLE_LEAVES).
- [x] A4 promotion preserves pawn accretion surplus (test: a7xe8=Q with surplus 2.4 carries through).
- Note: full Elo regression self-match still pending (Phase 3 harness); behavior-sensitive search tests (mate-in-one, false-mate, reference-parity, king-advance, flank-pawn, depth-4 reach) all pass.

### Phase 2 — Honesty & sync (1 day, writing-heavy) — ✅ DONE 2026-08-23
- [x] A8 WIRED: `observe_update` now shifts ALL 14 leaves via `dataclasses.replace` (leaf-reset bug fixed); `RocheEngine.play(..., observe=True)` persists evolved constants across a game; default off keeps matches reproducible. Tests: leaf preservation, G-direction on attack, engine-level evolution. Book §13.3 rewritten with layperson intro + STATUS block; C20 updated.
- [x] A9 RESCOPED: book §13.4 now says captor fragility is king-only in code; per-piece extent tracked as Phase-4 innovation #4.
- [x] A10: "Honest framing" paragraph added to book Ch.7 — *chess values are the units, gravity is the law, learning sets the coupling.*
- [x] S6 folded into A8 (C20 row updated); S7 marked resolved (batch_score); S8 prose fixed (native FastBoard ordering).
- [x] S1/S5 closed earlier with Phase 0–1.
- Remaining open ledger: S4 (refresh trained-constants reality check) and S9/S10 — both land with Phase 3's training run + CI.

### Phase 3 — The credibility gate (1–2 days) — ✅ PILOT DONE 2026-08-23
- [x] `scripts/credibility_gate.py`: fully automated self-contained ablation (self-play → train → validate → match → report). No external data needed.
- [x] Pilot executed end-to-end (531s wall): harvest 84 examples → 300 Adam steps → learned beats frozen **8/0/0** (+798 Elo proxy, n=8) while held-out MRR *regressed* (small-data overfitting, as Ch.14 C9 predicts).
- [x] Results published with an honest interpretation section (`docs/credibility_gate_results.md`); pilot table added to README with explicit caveats; book §12.5 refreshed (S4 partially closed — full refresh needs the scaled run).
- [ ] SCALING (next session): ≥5k examples + ≥200-game matches before any public Elo claim. The gate script already supports it via flags.

### Phase 4 — Innovation spike — ◐ PARTIAL 2026-08-23
- [x] Innovation #2 Peters–Mathews gravitational-wave energy loss IMPLEMENTED as leaf #15 (`lambda_gw`, init 0.0 → behavior-neutral until training switches it on; bounds [0,10]). Full plumbing: eval body, multiverse posterior, loss unpack, persistence. Tests: neutrality at 0, huddle-vs-spread radiation sign, 15-leaf packing. (Found & fixed a multiply-instead-of-divide kernel bug via these tests.)
- [x] Innovation #1 causal gravity: full design note at `docs/causal_gravity_design.md` (state, gate math, TT-key integration costs, acceptance criteria). Implementation deferred — it touches every field call site plus TT keys and deserves its own gated run.
- [ ] Innovation #4 per-piece Rg extent: deferred with rationale — it changes source softening semantics globally; implement together with the causal-gravity pass and re-gate.
- [ ] Innovations #5–#8 unchanged (time-dilation scheduling, learned rollout horizon, posterior distillation, UCI packaging).
- Note: the GW leaf renumbers TRAINABLE_LEAVES (14→15); old `trained_constants.json` artifacts load cleanly (missing key falls back to default 0.0).

### Standing rules for every change
1. No new chess knowledge in the evaluator — physics terms only; search craft stays in `minimax.py`, labeled instrumentation.
2. Every new knob becomes a trainable leaf with bounds in `constants.py`, or an explicitly documented fixed hyperparameter.
3. Every behavioral change ships with a test + one self-match line vs the previous build.
4. Book updated in the same PR as the code it describes (S-ledger stays empty).

### Post-audit addendum (2026-08-23, same day)
- [x] CI workflow `.github/workflows/tests.yml`: full pytest suite + engine smoke on every push/PR; README badge updated to the real count (62) and tied to CI (S9 closed).
- [x] S10 partially closed: every concept chapter (0–13) carries a layperson door-opener; deleted-module references in Ch.6 rewritten to the live implementation. A full line-number sweep repeats cheaply whenever chapters are touched.
- Remaining for future sessions (no local compute needed for the first two):
  1. Scaled credibility gate (`--games 40 --steps 2000 --match-games 200` on a stronger machine or overnight).
  2. Causal gravity implementation per `docs/causal_gravity_design.md`.
  3. Per-piece Rg extent, bundled with causal gravity + re-gate.

---

## Bottom Line

The physics core is genuinely rigorous, and recent bug archaeology (sacrifice bug, false mates, double-Lorentz, saturation) shows the pipeline works. Two systemic risks remain: **(1) internal inconsistency between parallel laws** (two light speeds; gated statics vs ungated dynamics), which undermines the "one universe" premise; **(2) narrative debt** — the book describes mechanisms (Observer, captor fragility, opening book) that are dormant or scoped differently in code. Phases 1–2 eliminate both cheaply. The credibility gate (Phase 3) remains the highest-stakes deliverable; causal gravity (Phase 4) is the biggest missed innovation and the most on-vision thing left to build.
