# Appendix — Glossary, Formula Sheet & References

## A.1 Glossary

| Term | Meaning |
|---|---|
| **N-body simulation** | Computing the motion/field of many bodies interacting via gravity. Here: 64 squares as bodies. |
| **Plummer softening** | Replacing $r^2$ with $r^2+\varepsilon^2$ so forces stay finite at $r\to 0$. |
| **Softening $\varepsilon$** | The fuzzy-radius length; a learned leaf (init 0.5). |
| **Mass vector $\mathbf m$** | $(64,)$ signed array: piece masses, white $+$, black $-$. |
| **Potential $U$** | Scalar "depth of well" at each square; negative, deeper = stronger pull. |
| **Force field $\vec F$** | $(64,2)$ vector field; direction of pull at each square. |
| **Tidal tensor $\mathbf A$** | Hessian $\nabla\nabla U$ at the King; a $2\times 2$ symmetric matrix. |
| **Eigenvalue $\lambda_1,\lambda_2$** | Principal stretch rates; $\lambda_1\ge\lambda_2$; $\lambda_1$ = line of failure. |
| **Roche / Hill limit** | Distance/threshold where tidal tearing overcomes self-binding. |
| **Disruption parameter $\eta$** | $\eta = R_g^3\lambda_1/m_{\text{ref}}^2$; dimensionless; > $\rho_{\text{roche}}$ ⇒ collapse. |
| **Radius of gyration $R_g$** | King's effective spatial extent (base 1.0; scaled as $(m/1000)^{1/3}$ under accretion). |
| **Verlet tidal drift** | Leapfrog projection of each King under the opponent's field; the change in η over the horizon is the "impending collapse" signal, gated by `lambda_drift`. |
| **Reference scale $m_{\text{ref}}$** | Fixed unit (3.5) keeping η in a usable range; not trained. |
| **Speed of light $c$** | Squares/ply; gates reach via $\sigma(c-d)$; learned, prior-bounded $[1,10]$. |
| **Retarded potential** | Force depends on where masses *were* $r/c$ plies ago (finite propagation). |
| **Reach gate $\sigma(c-d)$** | Smooth sigmoid making distant masses contribute less when $c$ is small. |
| **Source-attributed disruption** | A King is torn only by the *opponent's* masses, never its own. |
| **218-pad** | Pad every candidate-move batch to 218 (max legal moves) with zero-mass dummies. |
| **vmap** | JAX vectorizing map — scores all 218 rows in one compiled kernel. |
| **Quiescence** | Extending search along captures so eval is read at a quiet position. |
| **Alpha-beta / negamax** | Standard tree search; score from side-to-move perspective; prune when $\alpha\ge\beta$. |
| **Autodiff** | Automatic differentiation — JAX computes $\nabla_\theta\mathcal L$ through the pipeline. |
| **Bayesian model average** | $\text{Eval}=\frac1K\sum_i\text{Eval}(\theta_i)$ over sampled constants. |
| **Observer (2B)** | Opt-in per-move KL-anchored shift of *all* trainable leaves (`observe=True`); persisted across a game. |
| **Accretion (2C)** | Captured mass absorbed: $m_{\text{new}}=m_{\text{cap}}+0.8\,m_{\text{captured}}$. |
| **Lorentz mass** | Relativistic mass inflation from movement: $\gamma$ from $u=v/(v+c)$; governed by the same learned light speed as the field. |
| **Velocity accumulator** | Per-piece momentum in `FastBoard`; gains true lattice distance per move, decays ×0.985/ply. |
| **Gravitational-wave edge** | Peters–Mathews-style pairwise radiation $\propto G^4m_1^2m_2^2/r^5$, gated by learnable `lambda_gw` (init 0). |
| **Schwarzschild edge** | Horizon-overlap penalty: $r_s = 2Gm/c^2$ per piece; pairs whose horizons overlap pay a mass-weighted cost, gated by learnable `lambda_sch` (init 0). |
| **Drift horizon $dt_{\text{drift}}$** | Learnable Leapfrog time step (17th leaf, init 0.1) — how far ahead "impending collapse" projects. |
| **Posterior distillation** | One-at-a-time leaf sensitivity map (`scripts/distill_posterior.py`) — which constants the physics constrains. |
| **Trainable leaves** | The 17 learnable constants packed in `TRAINABLE_LEAVES` order — the universe's entire tunable surface. |
| **EvalTerms** | Named tuple of all twelve weighted score terms plus `total`; what the Glass Box decomposes. |
| **Credibility gate** | Automated learned-vs-frozen ablation (self-play → train → validate → match); `scripts/credibility_gate.py`. |
| **Glass Box** | The two-panel visualizer (board + potential heat-map + tidal ellipses). |
| **Ablation** | Training with vs without a feature (here: learned leaves vs frozen physics) to prove it works. |

## A.2 Formula sheet

**Force field (Plummer + $c$-gate):**
$$\vec F_i = G\sum_j |m_j|\,\sigma(c-d_{ij})\,\frac{\vec r_i-\vec r_j}{(d_{ij}^2+\varepsilon^2)^{3/2}}$$

**Potential:**
$$U_i = -G\sum_j |m_j|\,\frac{\sigma(c-d_{ij})}{\sqrt{d_{ij}^2+\varepsilon^2}},\qquad \sigma(x)=\frac{1}{1+e^{-x}}$$

**Tidal tensor (central differences, edge-padded):**
$$\mathbf A = \begin{pmatrix} U_{xx} & U_{xy} \\ U_{xy} & U_{yy}\end{pmatrix}$$

**Closed-form eigenvalues:**
$$\lambda_{1,2} = \frac{\operatorname{tr}\mathbf A}{2} \pm \sqrt{\left(\frac{\operatorname{tr}\mathbf A}{2}\right)^2 - \det\mathbf A}$$

**Disruption parameter (as implemented):**
$$\eta = \frac{R_g^3\,\lambda_1}{m_{\text{ref}}^2},\qquad m_{\text{ref}}=3.5$$
with $R_g$ the **effective** radius of gyration $R_g(|m_{\text{king}}|/1000)^{1/3}$ (mass-scaled under accretion, §4.5).

**Learned Roche threshold:** $\rho_{\text{roche}}$ (init 1.0, clamp $[0.05,20]$).

**Monotonicity prior on $c$:**
$$\text{prior}(c) = -\lambda_f\max(0,2-c) - \lambda_s\max(0,c-10)$$

**Score (White perspective) — all thirteen weighted terms (`EvalTerms`):**
$$\text{score} = \underbrace{\eta_b - \eta_w}_{\text{tidal}} + \underbrace{\text{bonus}_b + \text{pen}_w}_{\text{disruption force}} + \underbrace{\gamma(E_w{-}E_b)}_{\text{binding}} + \underbrace{m_{\text{gain}}(\Sigma m_w - \Sigma m_b)}_{\text{material}} + \underbrace{\lambda_\Delta\,\Delta\eta + c_g\,\Delta\text{CoM} + i_g\,\Delta I + e_g\,\Delta H}_{\text{move-sensitivity deltas}} + \underbrace{\lambda_{\text{drift}}\,\big(\Delta\eta_{b} - \Delta\eta_{w}\big)}_{\text{impending collapse}} + \underbrace{\lambda_{\text{gw}}\,(\text{gw}_b - \text{gw}_w)}_{\text{wave losses}} + \underbrace{\lambda_{\text{sch}}\,(\text{sch}_b - \text{sch}_w)}_{\text{horizon overlaps}}$$

**Loss:**
$$\mathcal L = \mathcal L_{\text{outcome}} + 0.5\,\mathcal L_{\text{policy}} + \mathcal L_{\text{prior}}$$

**Accretion:** $m_{\text{new}} = m_{\text{cap}} + 0.8\,m_{\text{captured}}$.

**Lorentz (single light speed):** $u=\frac{v}{v+c},\ \gamma_L=\frac{1}{\sqrt{1-u^2}},\ m_L=m\,\gamma_L$ — applied exactly once per ply via the `FastBoard` velocity accumulator; $c$ is the same learned constant that gates the field.

**Leapfrog (gated dynamics):** $\vec v\leftarrow\vec v+\vec F(\vec p)\,dt;\ \vec p\leftarrow\vec p+\vec v\,dt$ — used by `_eta_drift` with the same $\sigma(c-d)$ reach gate as the static field.

**Multiverse BMA:** $\text{Eval}(m)=\frac1K\sum_{i=1}^K\text{Eval}(m;\theta_i),\ \theta_i\sim p(\text{all 15 leaves})$, built trace-safely with `jax.vmap`.

**Observer update (all leaves, KL-anchored):** $\theta_{t+1} = \theta_t\big[1+(1-\beta_{KL})\alpha\tanh(s)\big]$ per leaf, with $c$ clipped to $[1,10]$; opt-in via `observe=True`.

## A.3 Notation consistency check

- Masses: white positive, black negative; gravity uses $|m|$. King = 1000.
- All chapter formulas use $\sigma$ for the sigmoid gate and $\lambda_1$ for the principal stretch.
- The evaluator's η denominator is $m_{\text{ref}}^2$ (Ch 4, 7); the visualizer uses the same (C12 verified fixed).
- $c$ is always in **squares/ply**, bounds $[1,10]$, and is the *only* light speed: it gates both the field reach and the Lorentz motion boost.
- The trainable surface is exactly 17 leaves (`TRAINABLE_LEAVES` in `core/constants.py`); everything else that looks tunable is either `mref` (fixed unit scale) or a documented fixed hyperparameter.

## A.4 References (primary sources in this repo)

1. `README.md` — project overview, pipeline diagram, usage.
2. `Planning/Kepler-64.md` — master architecture spec (static JIT padding, Roche, Verlet).
3. `Planning/Kapler-64 Code Review.md` — v1 line-by-line review (BUG 1–27).
4. `Planning/Kapler-64 Code Review v2.md` — v2 review (sign, accretion, Hessian, priors).
5. `Planning/Kepler-64 Audit & Layer-2 Development.md` — audit, honest/dishonest table, Multiverse/Observer/Accretion, build order, converged decisions.
6. `Planning/Kepler-64 P II.md` — physics correction (point-mass illusion), Glass Box, hybrid safety, training, blunder metric, disclaimer.
7. `Planning/Kepler-64 Extended Absurdity.md` — retarded potentials, Lorentz, byte-array seed.
8. `Planning/Kapler-64 Scaffold.md` — module map, build sequence, locked decisions.
9. `Planning/rg-dynamic-future.md` — dynamic $R_g$ plan.
10. `Planning/verification_and_training_suggestions.md` — verification report + training fixes.
11. `Planning/Kapler-64 Blog Writeup.md` — narrative writeup.
12. Code: `kepler64/core/{gravity,tidal,evaluate,constants,board,fastboard,lorentz,image_seed,transitions}.py`
13. Code: `kepler64/{search, training, multiverse, viz, bench, tests}/`
14. `kepler64/training/trained_constants_gate.json` — learned constants from the credibility-gate pilot.
15. `scripts/credibility_gate.py` + `docs/credibility_gate_results.md` — the learned-vs-frozen ablation harness and its pilot report.
16. `docs/causal_gravity_design.md` — design note for the next major innovation (field-age propagation).

**External (cited in planning docs):** Feynman (1981) chess/gods analogy; Roche & Hill sphere scaling; Plummer (1911) model; Verlet/Leapfrog integrators; JAX `vmap`/`jit`/`lax.scan`; BayesElo / Bradley-Terry; Peters–Mathews gravitational-wave quadrupole; Schwarzschild radius $r_s=2GM/c^2$.
