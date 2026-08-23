# Chapter 13 — Layer 2: The Multiverse, the Observer, and Accretion

> Layer 1 replaced the heuristic with gravity. Layer 2 asks a stranger question: *what if the physics itself is uncertain, and the engine observes itself?* This chapter covers the four Layer-2 mechanisms as actually implemented in `kepler64/multiverse/` and `core/lorentz.py`, `core/image_seed.py`.

---

## 13.1 Intuition: one universe is boring

Layer 1 evaluates every move under a *single* set of constants ($G, \varepsilon, c$). Layer 2 says: *we don't even know which universe we're in.* The constants are drawn from a **posterior distribution**; a move is good only if it survives across many possible physics. A Queen sacrifice that wins in most universes but catastrophi in one gets downweighted. The engine develops a kind of **epistemic humility** — hedging across realities.

> **Intuition box:** Instead of asking one weather forecast, you ask ten forecasters and average their advice, weighting the confident ones. A plan that's "good in all weather" beats one that's brilliant only in sunshine.

## 13.2 The Multiverse (2A): Bayesian model average over constants

`_perturb_constants` (`core/evaluate.py`) samples realizations of the physics around the learned base, each perturbed multiplicatively by Gaussian noise $\sigma=0.1$, with $c$ clipped back to $[1,10]$. The posterior now covers **all 14 trainable leaves** — $G, \varepsilon, c, \rho_{\text{roche}}$, bonus, kgain, $\gamma$, $R_g$, mat_gain, the four delta-term gains, and the Verlet drift gain `lambda_drift` — not just $(G,\varepsilon,c)$. The evaluation is their **equal-weight mean** (`multiverse_score_white`, built trace-safely with `jax.vmap` over the split keys):

$$\text{Eval}(m) = \frac{1}{K}\sum_{i=1}^{K} \text{Eval}(m;\, \theta_i),\qquad \theta_i \sim p(G,\varepsilon,c)$$

This is **true Bayesian model averaging** (uniform posterior mass), not worst-case hedging. The audit explicitly warns against the earlier "Boltzmann reweighting" that double-counted the score (`Kepler-64 Audit` §A point 3, counter-audit correction 3) — the current `mean(scores)` is the corrected, honest version.

> **⚠ [ISSUE: MULTIVERSE-WEIGHTING] (P2, Code Review v2 ISSUE 19):** An earlier `softmax(-scores)` weighted *lower* scores more (adversarial hedging, mislabeled "BMA"). The current code uses a plain mean — correct. But the docstring/narrative should still say "Bayesian model average," never "worst-case," to stay honest.

## 13.3 The Observer (2B): the true layer-2

> **Read me first (no background needed).** Every chess program so far plays under *fixed* rules of thumb. Kepler asks a stranger question: what if even the laws themselves are slightly uncertain? After each move, the engine glances at the resulting position and asks, *"which universe would make this position make sense?"* — then nudges its own constants (the strength of gravity, the speed of light) one tiny step toward that universe, while an elastic band ("KL anchor") keeps it from wandering into fantasy. Play a whole game this way and you watch physics itself slowly adapt: slightly stronger gravity as attacks mount, slightly faster light as the board opens.

`multiverse/observer.py` implements exactly that (`observer.py:30-56`). One observation shifts **every trainable leaf** by `alpha·tanh(score)` (multiplicatively, so scale-free), then pulls each back toward base with weight `beta_kl`; `c` alone is clamped to its physical prior [1, 10]. This is **online (meta-)variational inference**: the evaluation function co-evolves with the game — the opening is played under slightly different laws than the endgame, and the drift is measurable but tiny (alpha=0.01), never runaway.

> **STATUS — WIRED (opt-in), 2026-08-23.** `RocheEngine.play(..., observe=True)` calls `observe_update` after every searched move and persists the shifted constants on the engine instance, so successive moves genuinely play under evolving laws. Default is `observe=False`: matches and benchmarks stay reproducible under frozen physics. Two defects were fixed at wiring time: (1) the observer reconstructed only `(G, eps, c, roche)` and silently reset the other ten leaves to dataclass defaults — it now rebuilds through `dataclasses.replace` over all of `TRAINABLE_LEAVES`; (2) nothing ever called it.

> **⚠ [ISSUE: OBSERVER-ACCRETION-FEEDBACK] (P2 → open limitation):** The observation reads a score that may include accreted masses, but it adjusts no accretion efficiency — the feedback loop is incomplete relative to the full Layer-2 story. Documented as a known scope limit; extend deliberately if the narrative needs it.

$$\theta_{t+1} = \theta_t\big[1 + (1-\beta_{\text{KL}})\,\alpha\,\tanh(\text{score})\big],\qquad c_{t+1} = \text{clip}(\theta_{t+1}[c],\,1,\,10)$$

The KL anchor is essential: without it, in-game drift overfits one game and the engine hallucinates (counter-audit correction 4).

This is **online (meta-)variational inference**: the evaluation function co-evolves with the game. The "opening is played under one set of laws; the endgame under slightly different ones" — that is the narrative the math delivers (`Kepler-64 Audit` §B, converged decisions §1). The KL anchor is essential: without it, in-game drift overfits one game and the engine hallucinates (counter-audit correction 4).


## 13.4 Accretion (2C): captured mass is absorbed, not deleted

`multiverse/accretion.py` implements the cleanest Layer-2 mechanic (`accretion.py:14-17`):

$$m_{\text{new}} = m_{\text{captor}} + \eta_{\text{acc}}\,m_{\text{captured}},\qquad \eta_{\text{acc}} = 0.8$$

Captured mass is **not** removed — it's absorbed. This (a) conserves mass (fixing the broken energy-conservation blunder metric, P II §5) and (b) creates a real chess principle: **hoarding captures makes you powerful but fragile** (Eddington-style). The search wires this via `_accreted_mass` (`search/minimax.py:33-41`, correct parent-mass read — the Code Review v2 BUG 2 fix).

> **(Partially resolved — scope clarified 2026-08-23.)** The former [ISSUE: ACCRETION-RG-NOT-UPDATED] is closed *for kings*: the evaluation grows an effective radius of gyration $R_{g,\text{eff}} = R_g\,(|m_{\text{king}}|/1000)^{1/3}$ (§4.5), so a King that accretes mass grows more extended and tears more easily. For non-king pieces the fragility story is **designed but not yet active**: `apply_capture` computes the captor's inflated `Rg_new`, but no caller threads per-piece extent into the evaluator yet (roadmap innovation #4). Until then, read "hoarding makes you powerful but fragile" as a statement about *kings* — verified in code — not about every capturing piece.

## 13.5 Lorentz mass escalation: anti-repetition via relativity

`core/lorentz.py` inflates a piece's mass with repeated movement, using a **singularity-free** remap (`lorentz.py:14-20`):

$$u = \frac{v}{v + c}\in(0,1),\qquad \gamma_{\text{Lorentz}} = \frac{1}{\sqrt{1-u^2}}\in(1,\infty),\qquad m_{\text{Lorentz}} = m\cdot\gamma_{\text{Lorentz}}$$

where $v$ = squares moved. The naive $\gamma = 1/\sqrt{1-v^2/c^2}$ breaks when $v>c$ (imaginary mass); the $u = v/(v+c)$ remap never hits the singularity (`Kepler-64 Audit` Lorentz fix). Lorentz mass (own inflation) pairs with accretion mass (stolen mass): fast + greedy pieces become supermassive and fragile.

Lorentz mass is **wired into play**: `FastBoard` keeps a per-piece velocity accumulator (`core/fastboard.py`), and `mass_vector()` applies the boost directly, so a piece oscillating back and forth becomes supermassive — the physics-native pressure against pointless repetition. The boost is applied **exactly once** per ply (the double-application bug where the parent's factor compounded with the child's was fixed in `child_mass_vector` by unboosting the parent first).

> **(Resolved)** The former [ISSUE: LORENTZ-STATE-UNWIRED] — "Lorentz mass is implemented but not active in play" — is closed: the velocity accumulator feeds `mass_vector` and the search's `child_mass_vector` threads it through, and the compounding double-Lorentz bug is fixed.

## 13.6 Image-seeded universe: constants from a photograph

`core/image_seed.py` derives the universe's origin from a file (`seed_from_image`, `image_seed.py:24-41`): SHA-256 of the bytes gives a deterministic PRNG seed (decorative), and a **genuine 2D FFT of the decoded 8×8 grayscale image** → log-compressed → normalized → maps to $[G,\varepsilon,\rho_{\text{roche}},c]_{\text{init}}$. An 8×8 crop maps pixel intensity → initial piece masses (`initial_masses_from_image`, `image_seed.py:44-53`).

> **(Resolved)** The former [ISSUE: IMAGE-FFT-OF-BYTES] and [ISSUE: IMAGE-NORM-COLLAPSE] are closed: `_fft_magnitudes` now decodes the image with Pillow (8×8 grayscale) and takes the real FFT of the *pixels* (falling back to byte-entropy seeding only when Pillow is unavailable), and the spectrum is **log-compressed before normalization** so low-frequency-dominated natural images do not collapse every constant onto a clamp.

## 13.7 The (conditional) fluid layer

`multiverse/fluid.py` is kept **only if grounded** in the path-integral variance: $U_{\text{uncertain}}(x,y) = \mathrm{Var}_k[\text{Eval}_k(x,y)]$, then Stokes flow $\vec v = -\nabla U_{\text{uncertain}}/\mu$ (`fluid.py:13-23`). Laminar = stable plans; turbulent = sharp tactics. The audit warns this is "cargo cult" unless $U_{\text{uncertain}}$ is *defined as the Multiverse sample variance* (`Kepler-64 Audit` converged §2). As written it requires `eval_samples` as input — so it is dormant unless the Multiverse feeds it variance. Keep it that way or cut it.

## 13.8 Forward link

All concepts and mechanisms are now covered. Part D (Chapters 14–15 of the book = the Review and Mapping) delivers the independent critique and the requirements mapping. Note: book chapters 14–15 below are the critique/mapping; the project's source chapters end at 13.

**Cross-references:** $c$-prior / posterior → §5. Tidal/η → §3, §4. Bayesian average math → §13.2. Training that feeds Layer 2 → §12.
