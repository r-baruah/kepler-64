# Kepler-64 (The Roche Engine): Project Portfolio & Profile Dossier

> **Author & Creator:** Ripuranjan Baruah  
> **Repository:** [github.com/r-baruah/kepler-64](https://github.com/r-baruah/kepler-64)  
> **Live Web Simulator & Observatory:** [r-baruah.github.io/kepler-64](https://r-baruah.github.io/kepler-64/)  
> **Scientific Compendium:** [The Kepler-64 Book (`/Kepler-64_Book`)](./Kepler-64_Book/00_INDEX.md)  
> **Keywords:** Differentiable Physics, JAX, Computational Astrophysics, N-Body Simulation, Combinatorial Games, Custom Canvas Engine, Creative Engineering  

---

## 1. Executive Summary & The Elevator Pitch

**What if a chess position were evaluated by genuine gravitational physics instead of chess heuristics?**

**Kepler-64 (The Roche Engine)** is an experimental, differentiable chess engine built from first principles. It strips away 70 years of conventional chess programming—no piece-square tables, no mobility grids, no king safety penalty heuristics, and no hand-crafted pawn structure scores.

Instead, the entire board is modeled as a 2D discrete spacetime lattice where every piece is a point mass. Positions are evaluated by computing the **Plummer gravitational potential field ($\Phi$)**, the **tidal stress tensor ($\nabla\nabla\Phi$)** at the King, and the **astronomical Roche disruption limit ($\eta$)**. 

The King is never checkmated—it suffers **gravitational tidal disruption** when external tidal forces tear its defensive cohesion apart.

---

## 2. The Narrative Hook: The Feynman Reversal

> *"Richard Feynman once observed that discovering the laws of physics is like watching the gods play chess without being told the rules. If you watch long enough, you might catch on to a few tricks, and eventually deduce the underlying laws of the universe.*
> 
> *Kepler-64 asks the opposite question: What if the universe already knows the laws of physics, but has to play chess with them?"*

When you feed an AI engine nothing but Newton’s laws, Plummer softening, and tidal tensors, something remarkable emerges: **it naturally rediscovers romantic, 19th-century attacking chess**.

Without any programmed chess heuristics, the engine independently discovers that:
1. **Centralizing mass** creates deep gravitational wells that restrict enemy mobility.
2. **Coordinating attacking pieces** creates constructive tidal interference on the enemy King.
3. **Sacrificing material** can be mathematically optimal if the resulting shift in the center of mass triggers a tidal collapse ($\eta > \rho_{\text{roche}}$).

---

## 3. The 7 Layers of Absurdity (From Newtonian Gravity to the Multiverse)

Kepler-64 is designed with nested layers of physical and mathematical absurdity, progressing from classical mechanics to general relativity, astrophysics, and quantum-style Bayesian multiverse ensembles:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE 7 LAYERS OF ABSURDITY                                │
│                                                                                        │
│  [ Layer 7 ]  The Multiverse, Online Observer & Byte-Array Seed (p(G,ε,c) Ensembles)   │
│                                    ▲                                                   │
│  [ Layer 6 ]  Relativistic Mass Escalation (Lorentz Factor γ = 1 / √(1 - u²))          │
│                                    ▲                                                   │
│  [ Layer 5 ]  Accretion Queens & Black Holes (m_new = m_captor + 0.80 · m_victim)      │
│                                    ▲                                                   │
│  [ Layer 4 ]  Retarded Potentials & Finite Speed of Light c (Rolling Tensor Wavefronts)│
│                                    ▲                                                   │
│  [ Layer 3 ]  King Tidal Tensor (A = ∇∇Φ) & The Roche Disruption Limit (η > ρ)         │
│                                    ▲                                                   │
│  [ Layer 2 ]  Plummer Softening Length ε & C^∞ Differentiability (Learnable Leaves)    │
│                                    ▲                                                   │
│  [ Layer 1 ]  Newtonian Gravity on a Discrete 64-Square Lattice (Point Masses)         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Layer 1: Newtonian Gravity on a Discrete 64-Square Lattice
* **The Concept:** Every chess piece carries physical mass based on its classical material value ($m_{\text{pawn}} = 1\text{m}$, $m_{\text{minor}} = 3\text{m}$, $m_{\text{rook}} = 5\text{m}$, $m_{\text{queen}} = 9\text{m}$), while the King anchors the coordinate system as an immense stellar core ($m_{\text{king}} = 1000\text{m}$).
* **The Physics:** Every square on the board feels a net vector force $\vec F = G \sum \frac{m_j}{r^2}\hat r$. The evaluation function reads the physical gravity well rather than counting static points.

---

### Layer 2: Plummer Softening Length ($\varepsilon$) & $C^\infty$ Smooth Differentiability
* **The Concept:** In pure Newtonian physics, distance $r \to 0$ causes the force to diverge to infinity ($\frac{1}{0}$). On a discrete chessboard, adjacent squares would create infinite tears.
* **The Physics:** Kepler-64 adopts **Plummer softening (1911)** from globular cluster astrophysics:
  $$\Phi(p) = -G \sum_{j=1}^{64} \frac{|m_j| \cdot \sigma(c - \lVert p - r_j \rVert)}{\sqrt{\lVert p - r_j \rVert^2 + \varepsilon^2}}$$
* **The Auto-Diff Magic:** Plummer softening ensures that position evaluations are $C^\infty$ infinitely differentiable. Every physical constant ($G, \varepsilon, c, \rho_{\text{roche}}$) is a **differentiable leaf in JAX**, trained on Grandmaster matches using gradient descent (`jax.grad`).

---

### Layer 3: The King Tidal Tensor ($\mathbf{A} = \nabla\nabla\Phi$) & The Roche Disruption Limit ($\eta$)
* **The Concept:** Total force is the wrong metric: a King pinned between two equal attackers feels zero net force ($\sum F = 0$), yet is in fatal danger.
* **The Physics:** The engine computes the **tidal Hessian tensor** $\mathbf{A} = \nabla\nabla\Phi$ at the King’s exact coordinate. Using a closed-form analytical $2\times2$ eigensolver ($\sim 10$ FLOPs, zero LAPACK overhead), it extracts principal stretching eigenvalues $\lambda_1 \ge \lambda_2$ along the geometric line of failure:
  $$\mathbf{A} = \begin{pmatrix} \dfrac{\partial^2 \Phi}{\partial x^2} & \dfrac{\partial^2 \Phi}{\partial x \partial y} \\ \dfrac{\partial^2 \Phi}{\partial y \partial x} & \dfrac{\partial^2 \Phi}{\partial y^2} \end{pmatrix}$$
* **The Roche Collapse:** Derived from astronomical satellite stability (Roche 1849, Hill 1878), the dimensionless disruption parameter $\eta$ measures tidal stress against self-gravitational binding energy:
  $$\eta = \frac{R_g^3 \cdot \lambda_1}{m_{\text{king}}^2}, \qquad \text{Disruption Trigger: } \eta > \rho_{\text{roche}}$$

---

### Layer 4: Retarded Potentials & Finite Speed of Light ($c$)
* **The Concept:** In standard chess, moving a piece creates an instantaneous threat across the board. In relativistic reality, gravitational influence cannot travel faster than the speed of light ($c$).
* **The Physics:** $c$ acts as a differentiable reach gate. Moves outside the light cone cannot instantly exert full gravitational pull; the engine maintains a rolling historical buffer of previous board states. A queen sacrifice on `h8` does not instantly register as a threat to a king on `a1` until several plies later, when the gravitational wave propagates across the lattice.

---

### Layer 5: Accretion Mechanics & Black Hole Queens
* **The Concept:** In standard chess, captured pieces vanish into the ether. In Kepler-64, mass is conserved.
* **The Physics:** Capturing pieces absorb **80% of the victim's mass** ($\eta_{\text{acc}} = 0.80$):
  $$m_{\text{new}} = m_{\text{captor}} + 0.80 \cdot m_{\text{victim}}$$
* **The Result:** A Queen that hoards captures balloons into a supermassive localized black hole ($m > 25\text{m}$). It creates an inescapable gravitational basin that bends opponent trajectories, but also increases its own spatial exposure to tidal shear.

---

### Layer 6: Relativistic Mass Escalation (The Lorentz Factor $\gamma$)
* **The Concept:** Moving the same piece repeatedly across long distances in the opening is a positional blunder. Kepler-64 penalizes this via Special Relativity.
* **The Physics:** Each piece tracks a velocity scalar $v$ (squares per ply). To prevent singularities when $v > c$, Kepler-64 uses a singularity-free velocity remapping $u = \frac{v}{v + c} \in (0, 1)$ and calculates the relativistic Lorentz inflation factor:
  $$\gamma_{\text{Lorentz}} = \frac{1}{\sqrt{1 - u^2}}, \qquad m_{\text{rel}} = m \cdot \gamma_{\text{Lorentz}}$$
* High-velocity maneuvers create temporary mass spikes and kinetic shockwaves across the board.

---

### Layer 7: The Multiverse, Online Observer & The Byte-Array Universe Seed
* **The Multiverse (2A):** Instead of assuming one true set of physical laws, the engine performs **Bayesian Model Averaging** across $K=8$ posterior realizations of universes ($G_k, \varepsilon_k, c_k$). A move is only selected if it is robust across multiple parallel physical realities.
* **The Online Observer (2B):** As the game progresses, the engine performs online variational inference: the physical constants ($G, c$) dynamically evolve and tune themselves to the emotional tension and tactical volatility of the match.
* **The Image-Seeded Universe:** To ensure deterministic yet poetic origins, the initial constants of the physical universe are generated by taking the SHA-256 hash and 2D FFT spectral energy distribution of the raw binary bytes of a JPEG photograph.

---

## 4. Technical Architecture & Engineering Innovations

| Subsystem | Stack | Key Engineering Innovation |
|:---|:---|:---|
| **Differentiable Physics Core** | Python 3.11+, JAX, NumPy, XLA | Full $C^\infty$ smooth N-body potential pipeline; analytical $2\times2$ closed-form eigensolvers; gradient descent optimization of physical constants. |
| **The 218-Pad vmap Trick** | JAX Vectorized Batching | Pads variable legal moves ($0 \dots 218$) with zero-mass dummy tokens to compile a single fixed-shape `(218, 64)` XLA tensor kernel, eliminating compilation stalls. |
| **High-Performance Search** | Python, Numba / Bitboards | Alpha-Beta minimax with Principal Variation Search (PVS), quiescence search with capture/check horizons, and transposition tables. |
| **Interactive Canvas Observatory** | TypeScript, HTML5 2D Canvas, Web Workers, Vite | **100% custom-built 60 FPS canvas engine** (zero third-party UI board dependencies); real-time equipotential marching contours, vector streamlines, and high-DPI scaling. |
| **Multiverse Worker Pipeline** | Web Workers, Bayesian Ensembles | Non-blocking background worker evaluating positions across multiple physical parameter sets concurrently. |

---

## 5. Portfolio Showcase Snippets & Blurbs

### 5.1 One-Liner (GitHub Bio / Resume Header)
> **Kepler-64:** A differentiable N-body gravitational chess engine built with JAX and TypeScript where positions are evaluated through Plummer potential fields, tidal tensors, and astronomical Roche limits.

### 5.2 Short Project Card (Portfolio Website / Project Grid)
> **Kepler-64 (The Roche Engine) — Differentiable Astrophysical Chess**  
> *What if gravity evaluated chess?* Kepler-64 replaces conventional chess heuristics with computational astrophysics across 7 nested layers of physical modeling. Built with JAX, Python, and a custom TypeScript 2D canvas observatory, the engine models chess pieces as point masses that warp discrete spacetime. Kings lose not by checkmate, but by tidal disruption when exceeding the dimensionless Roche limit. Features include learnable physical constants ($G, \varepsilon, c$), capture accretion mechanics (Black Hole Queens), relativistic Lorentz mass inflation, and Bayesian multiverse ensembles.

### 5.3 Technical Resume Bullet Points
* **Architected a differentiable astrophysical chess engine** using JAX, replacing heuristic piece-square evaluation tables with $C^\infty$ smooth Plummer potential fields and $2\times2$ tidal Hessian tensors ($\nabla\nabla\Phi$).
* **Implemented analytical closed-form eigensolvers** in JAX/XLA to evaluate directional tidal strain on the King in $\approx 10$ FLOPs per node without LAPACK dependencies.
* **Engineered a 0-dependency interactive 60 FPS Canvas Observatory** in TypeScript/Vite featuring real-time equipotential marching contours, vector streamlines, and dual-mode mouse/touch interaction.
* **Trained learnable physical universe constants** ($G, \varepsilon, c, \rho$) on Grandmaster match datasets using gradient descent and Bayesian ensemble averaging across web workers.

---

## 6. Interview Talking Points & Deep-Dive Questions

1. **First-Principles Thinking:** Demonstrates how two radically different domains (classical combinatorial game theory and astrophysical tensor calculus) can be synthesized into a mathematically coherent, working system.
2. **Mathematical Rigor vs. Performance Optimization:** Deriving closed-form algebraic solutions for $2\times2$ symmetric matrix eigenvalues to keep search hot-loops blazing fast while maintaining mathematical integrity.
3. **Full-Stack Engineering Craftsmanship:** Seamlessly bridging high-performance scientific computing (JAX, autograd, numerical tensors) with bespoke frontend graphics (custom HTML5 Canvas rendering, Web Workers, responsive mobile UX).
4. **Authentic Creative Voice:** A project with strong intellectual identity, witty self-awareness, and rigorous execution that stands out memorably in any technical review.

---

*Authored by Ripuranjan Baruah — Creator & Lead Architect of Kepler-64 (The Roche Engine).*