# Kepler-64

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-0.4+-orange?logo=google&logoColor=white)

An astrophysical chess engine. Move-generation is handled by [python-chess](https://github.com/python-chess/python-chess); the evaluation function is a differentiable [JAX](https://github.com/google/jax) N-body gravitational simulation. Pieces are masses. The King loses by tidal disruption past a learned Roche limit instead of by checkmate.

---

## The idea

Every chess engine since the 1970s evaluates positions with hand-tuned heuristics — piece-square tables, mobility scores, king safety patterns. Even neural engines like Leela replace heuristics with a black-box network.

What if the evaluation was physics instead?

Not a metaphor for physics. Plummer-softened Newtonian gravity on a 64-square lattice, tidal tensors computed from the Hessian of the potential field, a Verlet rollout to project whether the King's formation will hold. The gravitational constant $G$ isn't chosen by hand — it's learned via gradient descent on game outcomes. The whole pipeline is differentiable.

The result is an engine that doesn't know any chess rules. It discovers positional principles through orbital mechanics.

---

## How the evaluation works

### The board as an N-body system

Each square is a point in 2D space. Occupied squares carry mass — pawn 1, knight 3, bishop 3, rook 5, queen 9, king 1000. The gravitational acceleration at every square is Plummer-softened Newtonian gravity:

$$F_i = G \sum_j \frac{m_j \, (r_i - r_j)}{\left( |r_i - r_j|^2 + \varepsilon^2 \right)^{3/2}}$$

The $(64, 64)$ pairwise distance matrix is static — the board geometry never changes, only the masses do. The entire force field is one `einsum` over a $(64,)$ mass vector. No Python loops in the hot path.

### Tidal disruption as the win condition

The tidal tensor is the Hessian of the gravitational potential at the King's square. For a 2×2 matrix this has a closed-form eigenvalue decomposition — no LAPACK, about 10 FLOPs:

$$\lambda = \frac{\text{tr}(A)}{2} \pm \sqrt{\left(\frac{\text{tr}(A)}{2}\right)^2 - \det(A)}$$

The largest eigenvalue $\lambda_1$ is the principal stretching rate — the axis along which the King's defensive structure is being pulled apart. The dimensionless tidal-disruption parameter is:

$$\eta = \frac{\lambda_1 \cdot R_g^3}{G \cdot M_{\text{king}}^2}$$

This recovers the Hill/Roche scaling from first principles on the board. When $\eta$ exceeds a learned critical threshold, the King's position structurally collapses. The engine tracks $d\eta/dt$ across a short Verlet rollout — an increasing tidal stress is the precursor to failure.

### Differentiable tactical safety

A boolean `is_checkmate()` check has zero gradient. That kills the learned-$G$ hook. The tactical penalty instead uses a sigmoid on the gravitational force magnitude at the King:

$$\text{penalty} = -M \cdot \sigma\!\left( k \cdot \left( \|F_{\text{king}}\| - 1 \right) \right)$$

Smooth, end-to-end differentiable, expressed in the same force language as the rest of the physics.

### Learned constants

The physical constants of this chess universe — $G$, the Plummer softening $\varepsilon$, the speed of light $c$, and the Roche threshold — are all leaves that `jax.grad` moves. A logistic loss on game outcomes backpropagates through the entire pipeline (Plummer → tidal tensor → eigenvalues → $\eta$) into the constants themselves.

The speed of light $c$ carries a monotonicity prior so gradient descent can't collapse the retardation story by pushing $c \to \infty$:

$$\text{prior}(c) = -\lambda_{\text{fast}} \cdot \max(0,\; 2 - c) \;-\; \lambda_{\text{slow}} \cdot \max(0,\; c - 10)$$

Sweet spot is around $c \approx 3\text{–}6$ squares per ply, giving 1–3 ply delays for cross-board threats.

### Verlet rollout

For each candidate move, a short Leapfrog (Verlet) integration projects the King's continuous coordinate forward under the local tidal force. Symplectic integrators conserve energy, so the projection of whether the King crosses its Roche threshold a few plies out is honest — without needing deep minimax search.

---

## The evaluation pipeline

```mermaid
graph LR
    A[Board] --> B[Mass vector<br/>(64,)]
    B --> C[Plummer gravity<br/>einsum]
    B --> D[Potential field<br/>U]
    C --> E[Force at King<br/>‖F_king‖]
    D --> F[Tidal tensor<br/>Hessian of U]
    F --> G[Closed-form eig<br/>λ₁, λ₂]
    G --> H[η = λ₁·Rg³ / G·M²]
    E --> I[Sigmoid<br/>tactical penalty]
    H --> J[Eval = −η_enemy<br/>+ η_self + penalty]
    I --> J
```

---

## The 218-pad trick

Chess has a theoretical maximum of 218 legal moves in any position. Every batch of candidate moves is statically padded to exactly 218, with zero-mass dummy pieces filling the unused slots. XLA traces the $(218, 64, 2)$ shape exactly once, giving a single `vmap` evaluation sweep.

The sub-millisecond claim refers to this XLA-compiled kernel. At v1, the surrounding python-chess move generation is the documented overhead — the pure-JAX board (v1.5) removes it.

---

## Layer 2: when the physics gets strange

Layer 1 replaced the heuristic with gravity. Layer 2 does something different: the evaluation function itself becomes physical.

### The Multiverse

The engine evaluates each candidate move under a *distribution* of physical constants sampled from a learned posterior, not under a single fixed set. The combination is a Bayesian model average:

$$\text{Eval}(m) = \sum_i p(\theta_i) \cdot \text{Eval}(m;\; \theta_i), \qquad \theta_i \sim \text{posterior}(G, \varepsilon, c)$$

A move that's robust across many possible universes gets boosted. One that's catastrophic in some gets downweighted. The retarded potential — gravitational influence propagating at finite $c$ — means a Queen sacrifice doesn't register as a threat until the gravity wave arrives several plies later. The engine doesn't know which universe's $c$ governs that delay.

### The Observer

After choosing a move, the engine updates its posterior over $(G, c)$ — the act of playing informs the engine about which physics it inhabits. The update is KL-anchored so it can't drift into nonsensical constants. The opening is played under one set of physical laws; the endgame under slightly different ones.

### Accretion

Captured pieces don't disappear. Their mass is absorbed by the capturing piece:

$$m_{\text{new}} = m_{\text{captor}} + \eta \cdot m_{\text{captured}}$$

The capturing piece grows heavier but also more extended — its radius of gyration increases, which raises its tidal eigenvalue. A Queen that captures three minor pieces becomes a supermassive but structurally fragile object. A single enemy pawn nearby can trigger tidal disruption of the overextended piece.

### Lorentz mass

Repeated movement of the same piece inflates its relativistic mass. To avoid the $v > c$ singularity, velocity is remapped:

$$u = \frac{v}{v + c}, \qquad \gamma = \frac{1}{\sqrt{1 - u^2}}$$

The result: piece shuffling has a real cost. Moving the same piece every opening turn inflates its mass, warping the local tidal tensor and destabilizing nearby friendly pieces. Lorentz mass is the piece's *own* mass; accretion mass is *stolen* mass. They combine.

### Image-seeded universe

The physical constants can be derived from a photograph. The raw bytes are reshaped, FFT'd, and the magnitude spectrum maps to initial values of $[G, \varepsilon, \text{roche}, c]$. An 8×8 crop maps pixel intensity to initial piece masses. The SHA-256 hash of the file provides deterministic seeding. The fundamental constants of the chess universe are, literally, the spectral content of one image.

---

## Usage

```bash
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, JAX (CPU or GPU), python-chess, matplotlib.

```python
import chess
from kepler64 import RocheEngine, Board

engine = RocheEngine()
board  = Board.from_chess(chess.Board())

score = engine.evaluate(board)       # positive = good for side to move
move  = engine.play(board, depth=2)  # alpha-beta + 218-pad vmap sweep
```

Image-seeded universe:

```python
engine = RocheEngine(seed_image="1000113151.jpg")
```

Generate a visualisation GIF:

```python
from kepler64.viz.glassbox import game_gif
game_gif(engine, out_path="kepler64.gif", max_ply=30)
```

---

## Project structure

```
kepler64/
├── __init__.py            # RocheEngine
├── core/
│   ├── board.py           # board state + signed mass vector
│   ├── constants.py       # G, ε, c, roche — learnable leaves + c-prior
│   ├── gravity.py         # static (64,64) distance matrix + Plummer einsum
│   ├── tidal.py           # Hessian, closed-form 2×2 eig, η
│   ├── verlet.py          # symplectic Leapfrog rollout
│   ├── lorentz.py         # relativistic mass escalation
│   ├── evaluate.py        # Eval = physics + soft sigmoid tactical penalty
│   └── image_seed.py      # image FFT → constants + initial masses
├── multiverse/
│   ├── posterior.py       # Bayesian average over (G, ε, c)
│   ├── observer.py        # in-game KL-anchored posterior update
│   ├── accretion.py       # captured mass absorbed, not deleted
│   └── fluid.py           # variance-field Stokes flow
├── search/
│   ├── minimax.py         # alpha-beta + 218-pad vmap sweep
│   └── openings.py        # quasi-equilibrium opening book
├── training/
│   ├── data.py            # game → (mass_vector, outcome) samples
│   ├── loss.py            # logistic outcome + c-prior
│   ├── train.py           # jax.grad through the physics engine
│   └── ablation.py        # G learned vs G=1 comparison
├── viz/
│   └── glassbox.py        # potential heatmap + quiver + line of failure
├── bench/
│   ├── sweep_time.py      # 218-move evaluation sweep benchmark
│   └── elo_ladder.py      # BayesElo internal rating pool
└── tests/
    ├── test_gravity.py    # force field correctness
    ├── test_tidal.py      # closed-form eig vs scipy
    └── test_board.py      # board roundtrip vs python-chess
```

---

## Status

| Component | Status |
|---|---|
| Plummer gravity (static einsum) | Done |
| Closed-form 2×2 tidal eigenvalues | Done |
| $\eta$ tidal disruption + $d\eta/dt$ tracking | Done |
| Soft sigmoid tactical penalty | Done |
| Signed mass vector | Done |
| Verlet rollout | Done |
| Alpha-beta search + 218-pad sweep | Done |
| Accretion on capture | Done |
| Lorentz mass escalation | Done |
| Image-seeded universe | Done |
| Glass Box visualizer | Done |
| Training loop | Done |
| Ablation study | Done |
| Multiverse (posterior avg) | Done |
| Observer (in-game update) | Done |
| Variance-field fluid | Done |
| Pure-JAX board (removes move-gen tax) | v1.5 |

---

## Testing

```bash
pytest kepler64/tests/ -v
```

---

## What this is not

Kepler-64 is not a competitive chess engine. It is a differentiable physics simulation that happens to play chess. It replaces static evaluation heuristics with a fully vectorized N-body gravitational potential, optimized in JAX. It does not understand chess theory. It understands orbital mechanics and structural collapse.

---

## License

MIT
