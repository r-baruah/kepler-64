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
| **Observer (2B)** | In-game update of $G,c$ from the post-move score, KL-anchored. |
| **Accretion (2C)** | Captured mass absorbed: $m_{\text{new}}=m_{\text{cap}}+0.8\,m_{\text{captured}}$. |
| **Lorentz mass** | Relativistic mass inflation from repeated movement (singularity-free remap). |
| **Glass Box** | The two-panel visualizer (board + potential heat-map + tidal ellipses). |
| **Ablation** | Training with vs without a feature (here: learned $G$ vs $G{=}1$) to prove it works. |

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

**Score (White perspective):**
$$\text{score} = \eta_b - \eta_w + \text{bonus}_b + \text{pen}_w + \gamma(E_w-E_b) + m_{\text{gain}}(\Sigma m_w - \Sigma m_b) + \lambda_{\text{drift}}\big(\Delta\eta_{\text{on black K}} - \Delta\eta_{\text{on white K}}\big)$$

**Loss:**
$$\mathcal L = \mathcal L_{\text{outcome}} + 0.5\,\mathcal L_{\text{policy}} + \mathcal L_{\text{prior}}$$

**Accretion:** $m_{\text{new}} = m_{\text{cap}} + 0.8\,m_{\text{captured}}$.

**Lorentz (singularity-free, wired):** $u=\frac{v}{v+c},\ \gamma_L=\frac{1}{\sqrt{1-u^2}},\ m_L=m\,\gamma_L$ — applied exactly once per ply via the `FastBoard` velocity accumulator.

**Leapfrog (wired as the drift term):** $\vec v\leftarrow\vec v+\vec F(\vec p)\,dt;\ \vec p\leftarrow\vec p+\vec v\,dt$ — used by `_eta_drift` to read the "impending collapse" change in η.

**Multiverse BMA:** $\text{Eval}(m)=\frac1K\sum_{i=1}^K\text{Eval}(m;\theta_i),\ \theta_i\sim p(\text{all 14 leaves})$, built trace-safely with `jax.vmap`.

**Observer update:** $G_{t+1}=G_t+\beta_{KL}(G_{\text{base}}-G_t)+\alpha\tanh(s)\,G_t$ (and analogously $c$, clipped $[1,10]$).

## A.3 Notation consistency check

- Masses: white positive, black negative; gravity uses $|m|$. King = 1000.
- All chapter formulas use $\sigma$ for the sigmoid gate and $\lambda_1$ for the principal stretch.
- The evaluator's η denominator is $m_{\text{ref}}^2$ (Ch 4, 7); the *visualizer* still uses $G\,M_{\text{king}}^2$ (flagged C12).
- $c$ is always in **squares/ply**, bounds $[1,10]$ (Ch 5).

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
12. Code: `kepler64/core/{gravity,tidal,verlet,evaluate,constants,board,fastboard,lorentz,image_seed}.py`
13. Code: `kepler64/{search, training, multiverse, viz, bench, tests}/`
14. `kepler64/training/trained_constants.json` — post-training constants snapshot.

**External (cited in planning docs):** Feynman (1981) chess/gods analogy; Roche & Hill sphere scaling; Plummer (1911) model; Verlet/Leapfrog integrators; JAX `vmap`/`jit`/`lax.scan`; BayesElo / Bradley-Terry; Peters–Mathews gravitational-wave quadrupole; Schwarzschild radius $r_s=2GM/c^2$.
