# Chapter 14 — Independent Review: Gaps, Bugs, Misassumptions & Missed Innovations

> **Part D — Critique & Strategy.** This is the most important chapter. A reviewer remembers the critique more than the architecture. Every item below is cross-referenced to the concept chapter where the flaw first appears, and every fix is concrete. Items are ranked P0 (must fix) → P3 (awareness). Items labeled **(new)** were not in the existing code-review docs.

---

## 14.1 Summary table (all issues ranked)

*Status: **✅ resolved** = fixed in the current code (2026-08); **open** = still to do.*

| Pri | ID | Issue | Source | Where | Status |
|---|---|---|---|---|---|
| **P0** | C1 | Ablation table ($G$ learned vs $G{=}1$) not in README | Audit | §12.4 | ✅ gate built + pilot table in README (`docs/credibility_gate_results.md`); scaled run pending before thesis claims |
| **P0** | C2 | King not found → silent eval at a1 (`_king_idx` fallback) | v1 BUG4/5, v2 BUG4 | §7.4 | ✅ resolved |
| **P0** | C3 | Import paths (`.core` vs `..core`) across packages | v1 BUG9/24 | §10.2 | ✅ resolved — verified: no single-dot sibling imports remain |
| **P0** | C4 | Tactical penalty sign inverted (older revision) | v2 BUG1 | §7.5 | ✅ resolved |
| **P1** | C5 | $c$-prior used Python `max` (broke gradient) | v1 BUG3, v2 BUG5/6 | §5.3 | ✅ resolved |
| **P1** | C6 | `c` not traced (plain float dataclass) | v1 BUG8 | §5.5 | ✅ resolved |
| **P1** | C7 | Boundary Hessian wrong for ~25% King positions | v2 BUG3 | §3.4 | ✅ resolved |
| **P1** | C8 | King-mass fragile match (`int==1000`) under accretion | v1 BUG5 | §7.4 | ✅ resolved |
| **P1** | C9 | Small-data collapse: `roche`/`gamma`→harmful | verification | §12.5 | open |
| **P1** | C10 | $G$ non-identifiable; "learned $G$" overclaim | Audit/train | §9.4 | open (framing) |
| **P1** | C11 | Verlet rollout not in live eval / not differentiating $d\eta/dt$ | — | §6.5/§4.6 | ✅ resolved |
| **P2** | C12 | Visualizer η uses old $G\,M^2$ denominator (mismatch) | **(new)** | §11.4 | ✅ resolved — `viz/glassbox.py` uses $R_g^3\lambda_1/m_{\text{ref}}^2$, matching the evaluator |
| **P2** | C13 | Accretion doesn't update $R_g$ (fragility missing) | v1 ISS17/v2 ISS16 | §13.4 | ✅ resolved |
| **P2** | C14 | Lorentz mass function unwired into play | **(new)** | §13.5 | ✅ resolved |
| **P2** | C15 | Image FFT of compressed bytes (dishonest claim) | v2 ISS18 | §13.6 | ✅ resolved |
| **P2** | C16 | Image constant normalization collapses to clamps | v1 BUG11 | §13.6 | ✅ resolved |
| **P2** | C17 | `dη/dt` documented, not implemented | v1 BUG12/v2 ISS15 | §4.6 | ✅ resolved |
| **P2** | C18 | Opening book stub returns `None` | P II/Scaffold | §10.6 | ✅ resolved — module deleted per no-books vision |
| **P2** | C19 | Multiverse mislabeled as BMA vs hedging | v2 ISS19 | §13.2 | ✅ resolved |
| **P2** | C20 | Observer doesn't update accretion efficiency | v2 ISS20 | §13.3 | ✅ observer WIRED (opt-in `observe=True`); accretion-feedback remains a documented scope limit |
| **P3** | C21 | `batch_score` Python loop / double potential | v2 BUG9/10 | §8.4 | ✅ resolved — single stack + pad, one vmap |
| **P3** | C22 | `tests` missing `__init__`, weak gravity tests | v1 ISS14/19 | §14.3 | open |
| **P3** | C23 | Naming: "Kapler-64" vs package `kepler64` | v1 ISS15 | §14.3 | open |
| **P3** | C24 | `verlet` `int(king_sq)` under vmap | v2 BUG8 | §6.3 | ✅ resolved |

---

## 14.2 P0 — Must fix before showing to a reviewer

### C1 — The ablation gate is missing from the published artifacts
**What:** The single most important honesty check — train with learned constants vs $G{=}1$ fixed and report the Elo/accuracy delta — is specified as a "first-class README artifact" (`Kepler-64 Audit` Honest/Dishonest table, `Kepler-64 Scaffold.md`) but is **not present** in README.md.
**Why it matters:** The entire thesis is *"gravity, learned via gradient descent, does real work."* Without the delta, a reviewer (or an ML audience) will assume the physics is cosmetic window-dressing over the material term — exactly the failure mode the audit warns about ("if the delta is <50 Elo, the thesis collapses").
**How the fix works:** Run `training/ablation.py` (or a manual `fix_G=True` vs `fix_G=False` train) on the full 5k-puzzle / 10k-position dataset; publish a table: `G=1 fixed` → Elo/MRR vs `G learned` → Elo/MRR, with the delta. Put it in README under the benchmarks. See §12.4.
**See:** §12.4, §9.4.

### C2 — Silent wrong King detection — ✅ resolved
**What:** `_king_idx` used `argmax(mask)` which returns **square 0 (a1)** when no square matches `|m|≈1000` with the right sign — e.g. during training on a corrupted board, or after accretion shifts King mass away from exactly 1000.
**Why it matters:** The evaluation would compute tidal disruption at a1 instead of the real King — a **silently wrong score with no error**, the worst kind of bug: invisible.
**Resolution:** `_king_idx` now matches with `jnp.isclose(..., atol=5.0)` so accretion-shifted Kings are still found, and `_king_found` neutralizes the disruption/force terms when either King is missing — no silent a1 fallback. See §7.4.

### C3 — Broken relative imports across packages
**What:** `search/`, `multiverse/`, `training/` used `from .core.constants import …` where `core` is a *sibling*, not a parent. Correct is `from ..core.constants import …`.
**Why it matters:** Every module in those packages raises `ImportError` / `ModuleNotFoundError` when imported outside the package — i.e. the engine **does not import at all** in many entry paths. This is a hard crash, not a subtle bug.
**How the fix works:** Confirm all cross-package imports are double-dot. Current files (`core/search/minimax.py:14`, `core/multiverse/posterior.py:19`) already use `..core`. Grep the tree for `from .core` and `from .training` to be sure none remain.
**See:** §10.2.

### C4 — Inverted tactical penalty sign (historical, verify current)
**What:** A prior `_score_core` penalized White when *Black's* king was under force — the sign was backwards, so the engine played *against* its own threats; estimated Elo loss of hundreds of points (Code Review v2 BUG 1).
**Why it matters:** If any code path still uses the old sign, the engine self-sabotages. The current code (`core/evaluate.py:83-84`) uses `bonus_b` (good for White) and `pen_w` (bad for White) — correct — but this must be asserted by a regression test (e.g. "a position with heavy White mass near Black's king scores higher than the symmetric one").
**How the fix works:** Add a unit test asserting the sign convention; keep `bonus_b`/`pen_w` as the canonical form. See §7.5.

---

## 14.3 P1 — Should fix (consequences if skipped)

### C5 — Python `max` in the $c$-prior broke gradient flow
**What/Why:** `max(0, 2-c)` is not differentiable under `jax.grad`; it either crashed or froze $c$ as concrete, defeating the entire point of learning $c$. **Fix:** `jnp.maximum` (present in `core/constants.py:45` and `training/loss.py:98`). Verify both. See §5.3.

### C6 — `c` not a traced type
**What/Why:** A `@dataclass` field `c: float` is not JAX-traced; `jax.grad` can't move it. **Fix:** route `c` as a positional scalar argument through `_score_core` (done: `core/evaluate.py:104`). Confirm `train.py`/`loss.py` pass it positionally. See §5.5.

### C7 — Boundary Hessian wrong for ~25% of King positions
**What/Why:** Naive central differences collapse at ranks 1/8, files a/h (edge stencils become first derivatives). The win condition is mathematically wrong for any King on the border. **Fix (present):** `jnp.pad(Ug,1,mode="edge")` before differencing (`core/tidal.py:38`). **Verify it is present and that `tidal_disruption` also uses it.** See §3.4.

### C8 — King mass match fragile under accretion — ✅ resolved
**What/Why:** `int(masses[sq]) == 1000.0` fails once accretion makes the King `1002.4`, so the King becomes unfindable (compounding C2). **Resolution:** `_king_idx` now uses `jnp.isclose(masses, 1000, atol=5.0)` (and `_king_found` guards the terms). See §7.4.

### C9 — Small-data training collapse
**What/Why:** A 378-example run drove `roche→0.05` (floor) and `gamma→0.0`, *dropping* puzzle accuracy. **Fix:** scale to 5k+ puzzles / 10k+ positions, Adam + cosine LR, 80/20 val split, two-phase (outcome then policy) training. Full plan in `verification_and_training_suggestions.md`. See §12.5.

### C10 — "$G$ learned" overclaim
**What/Why:** In the η formula $G$ cancels, so training collapses it to 0; the project freezes $G$ (`fix_G=True`) by default. The README's flagship line is literally true only when `fix_G=False`. **Fix:** phrase precisely — "constants including $\varepsilon, c, \rho_{\text{roche}}$ are learned; $G$ is typically frozen for identifiability." See §9.4.

### C11 — Verlet rollout not in the live eval; $d\eta/dt$ absent — ✅ resolved
**What/Why:** The README implied the rollout projects collapse, but the deployed score used the static η snapshot + force-sigmoid. **Resolution:** the rollout is now wired into the eval as the **Verlet tidal-drift term** (`_eta_drift`, gated by the `lambda_drift` leaf), reading the change in tidal stress over a short Leapfrog projection via an exact analytical Hessian. See §6.5, §4.6.

---

## 14.4 P2 — Nice to have / Q&A risk

### C12 **(new)** — Visualizer η denominator mismatch
The visualizer's `_eta_from_U64` uses $\eta=R_g^3\lambda_1/(G\,M_{\text{king}}^2)$ while the evaluator uses $/m_{\text{ref}}^2$. The red "danger" coloring can contradict the engine's own score. **Fix:** call the evaluator's `_eta` (with `mref`) in the visualizer, or document the display-scale approximation. See §11.4.

### C13 — Accretion doesn't update $R_g$ — ✅ resolved
Mass is absorbed but radius of gyration is not, so the "overextended piece torn by a pawn" fragility was not implemented. **Resolution:** the eval now uses an effective radius of gyration $R_{g,\text{eff}} = R_g(|m_{\text{king}}|/1000)^{1/3}$ (§4.5), so an accreted King grows more extended and tears more easily. See §13.4.

### C14 **(new)** — Lorentz mass unwired — ✅ resolved
Lorentz mass was defined but inactive. **Resolution:** `FastBoard` tracks a per-piece velocity accumulator and `mass_vector()` applies the boost; `child_mass_vector` threads it through the search, and the compounding double-Lorentz bug (parent factor × child factor across plies) was fixed by unboosting the parent first. See §13.5.

### C15 — Image FFT of compressed bytes — ✅ resolved
Reshaping JPEG bytes and FFTing them was not "the FFT of the image." **Resolution:** `_fft_magnitudes` now decodes with Pillow (8×8 grayscale) and takes the real FFT of the *pixels*, falling back to byte-entropy seeding only when Pillow is unavailable. See §13.6.

### C16 — Image normalization collapses constants — ✅ resolved
`norm = spectrum/sum` over natural-image spectra drove constants to clamps. **Resolution:** the spectrum is log-compressed before normalization, so low-frequency-dominated images don't collapse every constant onto a clamp. See §13.6.

### C17 — $d\eta/dt$ documented, not implemented — ✅ resolved
`dη/dt` was documented but absent. **Resolution:** implemented as the Verlet tidal-drift term (§4.6, §6.5), gated by the `lambda_drift` leaf. See §4.6.

### C18 — Opening book stub
`openings.py` returns `None`. **Fix:** implement the quasi-equilibrium book or mark as future work. See §10.6.

### C19 — Multiverse weighting honesty — ✅ resolved
Older code used `softmax(-scores)` (adversarial), mislabeled BMA. The current `mean` is correct, built trace-safely with `jax.vmap`, and the posterior now covers all 14 leaves; keep the narrative honest ("Bayesian model average," never "worst-case"). See §13.2.

### C20 — Observer doesn't update accretion efficiency
Only $G,c$ shift; the Layer-2 feedback loop omits $\eta_{\text{acc}}$. **Fix:** extend or note the limitation. See §13.3.

---

## 14.5 P3 — Awareness (for Q&A)

- **C21** `batch_score` builds JAX arrays in a Python loop; `_score_body` recomputes potential. Minor speed wins (§8.4).
- **C22** `tests/` should add `__init__.py`; strengthen `test_gravity.py` (Newton's 3rd law, direction, softening finiteness) and fix the "negative discriminant" test which actually has a *positive* discriminant (`[[1,2],[2,1]]` → eigenvalues 3, −1). See §3.3 note.
- **C23** Project is "Kapler-64" but package is `kepler64` — pick one naming convention in docs.
- **C24** `verlet.py` `int(king_sq)` breaks under `vmap`; use JAX indexing (already fixed at `core/verlet.py:36`).

---

## 14.6 Missed innovations & misassumptions (new analysis)

1. **Schwarzschild-radius mechanic (audit §"Additional Absurdity") is unimplemented.** Defining $r_s = 2GM/c^2$ per piece and softly penalizing clustering would add a real "don't pile heavy pieces" principle. Low cost, high thematic value. *Tradeoff:* another constant to learn; risk of over-constraining.

2. **Gravitational-wave energy loss (Peters–Mathews) is unimplemented.** A pairwise $dE/dt \propto G^4 m_1^2 m_2^2/r^5$ term naturally penalizes piece shuffling (anti-repetition) and falls out of real physics. *Tradeoff:* must be verified not to dominate at sane $G$ (audit addition 3).

3. **"Real astrophysics" misassumption.** The board is 2D discrete; a King is a point with no volume; masses are chess values, not solar masses. The honest framing (metaphor, not literal universe) must be maintained in every public claim (P II §1, Audit Honest/Dishonest). The strongest version of the thesis is *"a differentiable N-body potential acting as an evaluation heuristic"* — say that, not "real astrophysics."

4. **Material dominance undermines the thesis (addressed).** With `mat_gain=0.3` and material ranging ±20 while η≈0.1–0.5, the engine was effectively "material + tiny physics." After the 2026-08-17 sacrifice-bug diagnosis, the defaults were re-bounded so **material is decisively first-class** and the physics levers act as refinement: `mat_gain` 0.3→2.0 (a captured rook moves the score ~10, a queen ~18), `bonus` 220→300, `lambda_delta` 1→2, `entropy_gain` 2.5→4, `com_gain`/`inertia_gain` reduced to 1, and `gamma` 0.25→0 (the absolute cohesion gap punished development). The tidal/disruption terms are competitive *without* letting a positional knob override a hanging piece. The ablation table (C1) is still the empirical proof required to convert "irony" into "credibility."

5. **The retardation story needs the prior to survive.** If the $c$-prior is ever dropped or `max` creeps back, $c\to\infty$ and Layer 2's "delayed ripple" collapses. Guard it with a test asserting $c$ stays in $[1,10]$ after training.

---

## 14.7 Bottom line — what to fix first

If a reviewer walks in tomorrow, the things most likely to hurt:

1. **The ablation gate now EXISTS and runs** (C1 closed at pilot scale): `scripts/credibility_gate.py` + a README table with caveats. The residual risk is *scale* — an 8-game match and 84 examples invite the (fair) rebuttal "not statistically meaningful." Quote the pilot as method demonstration only.
2. **Small-data overclaim (C9, C10)** remains the honest frontier until a ≥5k-example run lands — the pilot's ranking regression is itself evidence of it.
3. **Former open items C3, C12, C18, C20, C21 are all closed** (verified in code 2026-08-23): imports verified clean, visualizer η matches the evaluator, opening book deleted by vision decision, Observer wired opt-in with its limitation documented, batch_score single-vmap.

What remains open is engineering scale, not correctness or honesty of claims.
