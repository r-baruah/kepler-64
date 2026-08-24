<div align="center">

# 🪐 Kepler-64
### *We didn't teach a computer chess. We built a tiny universe — and let gravity figure out the game.*

Every chess piece has **mass**. Every piece **pulls** on every other piece. Attacks are gravitational fields. Captures are **absorbed into your body**. And you don't checkmate the enemy King — you get close enough that the **tidal forces tear it apart**, exactly like a moon shredded by Jupiter.

Here's the twist: **nobody wrote any chess rules into this engine.** No opening books, no "knights belong on outposts," no grandmaster data. Just the laws of physics — and then gradient descent tuned the constants of that universe (how strong gravity is, how fast its influence travels, when a King collapses) until the physics started playing real chess.

[![Live Observatory](https://img.shields.io/badge/🔭_Live_Observatory-Deploy-2448b8?style=for-the-badge)](https://r-baruah.github.io/kepler-64/)
[![Tests](https://img.shields.io/badge/tests-62%20passing-16a34a?style=for-the-badge)](#testing--verification)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-differentiable-f06426?style=for-the-badge&logo=google)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

[**🔭 Explore the Live Observatory**](https://r-baruah.github.io/kepler-64/) • [**📖 Read the Book**](Kepler-64_Book/PRELUDE.md) • [**🚀 Try It Yourself**](#-try-it-in-3-lines) • [**🧠 How It Works**](#-how-it-works-the-simple-version)

*If a universe where orbital mechanics outplays pawn structures sounds fun, a ⭐ helps other curious people find it.*

</div>

---

## 🌌 The Idea in 30 Seconds

**1. Pieces become planets.** ♙ Pawn = 1 unit of mass, ♗ Bishop = 3, ♖ Rook = 5, ♕ Queen = 9, ♔ King = a whopping 1000. Place them on the 64 squares and they create a real gravitational field — computed with actual Newtonian equations.

**2. Attacks become tides.** One enemy piece near your King is an annoyance. But pieces pulling from *opposite sides at once* stretch the King the way the Moon stretches Earth's oceans. The engine measures that stretching with a **tidal tensor** — and when it crosses the critical threshold (the same Roche limit that shreds moons around Jupiter), the King is judged *structurally collapsing*. That's how this universe wins.

**3. Captures feed your mass.** Take a piece and you don't just remove it — you **absorb 80% of its mass**. Your piece gets heavier, its pull stronger… and physically more fragile. Greed has a price written into the laws.

**4. The laws themselves are learned.** This universe has **17 knobs** — the strength of gravity, how soft pieces are, how fast gravitational influence travels, the collapse threshold, and more. Training runs gradient descent *straight through the physics simulation* and tunes every knob until the universe's judgment matches what actually happens in real games.

> **The honest footnote:** this is real physics math running on a board — not a claim that chess *is* astrophysics. It's a deliberately engineered universe whose equations happen to be exactly computable, exactly explainable, and exactly learnable.

---

## 🔭 See It Live — No Install Needed

The **[Web Observatory](https://r-baruah.github.io/kepler-64/)** renders the invisible parts of every game right in your browser:

- 🗺️ **Gravitational field contours** warping in real time as pieces move
- 🔴 **Tidal stress ellipses** on both Kings, stretching along their exact "line of failure"
- 📈 **Game trajectory timeline** — watch who is physically winning, move by move
- 📚 **Interactive compendium** explaining every formula on screen
- 🎞️ **Export animated GIFs** of any game with live physics overlays

---

## ✨ The Fun Parts

| Mechanic | In plain words |
|---|---|
| 🪐 **Accretion** | Capture a piece → absorb most of its mass. Hoarding captures makes you powerful *and* fragile. |
| 🌊 **Light-speed gravity** | Influence travels at `c` squares per move (a *learnable* number). Distant threats literally take plies to arrive. |
| 🏃 **Relativistic pieces** | Move a piece constantly and it gains relativistic mass — kinetic energy is mass even here. |
| 🌌 **The Multiverse** | Every candidate move is judged across several randomly-varied universes; only plans that survive in *most* realities win. |
| 🪞 **The Observer** | Optionally, the engine nudges its own constants after each move — the laws co-evolve with the game. |
| 🕳️ **Event horizons & waves** | Piled-up heavy pieces pay horizon-overlap penalties and radiate away energy — if training decides those laws matter. |

---

## 🧠 How It Works (the simple version)

1. **Board → masses.** The position becomes 64 numbers: each square's mass (White positive, Black negative).
2. **Masses → field.** Real Newtonian gravity (softened so nothing explodes) builds a force-and-potential landscape across the whole board.
3. **Field → danger.** At each King, the engine measures how violently the field *stretches* it — and whether that stretch is crossing the learned collapse threshold.
4. **Search picks the future.** A classic alpha-beta search tries moves and asks the physics: *which line ends with the enemy King torn apart and mine intact?*

<details>
<summary><b>🔬 The full mathematical pipeline (click to expand)</b></summary>

Traditional chess engines evaluate positions using hand-crafted piece-square tables, mobility counts, or neural networks. **Kepler-64 replaces these heuristics with gravitational field calculations:**

```
                    ┌─────────────────────────┐
                    │     64-Square Lattice    │
                    │   Signed Mass Vector m  │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Plummer Potential Field          King Tidal Tensor (Hessian)
       Φ(p) = -G ∑ m_j / √(r² + ε²)     A = ∇∇Φ |_{King}  ⟹  λ₁, λ₂
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                    Dimensionless Roche Limit
                    η = (R_g³ · λ₁) / m_ref²
                                 │
                                 ▼
                     Evaluator Score Vector
              (Enemy Tide - Own Tide + ΔE_binding + Material)
```

### 1. The Plummer Potential Field $\Phi(p)$
Every piece acts as a point mass on the 2D board ($P=1, N=3, B=3, R=5, Q=9, K=1000$). To avoid infinite gravitational forces when pieces are close together, potential is smoothed using the **Plummer kernel**:

$$\Phi(p) = -G \sum_{j=1}^{64} \frac{|m_j| \cdot \sigma(c - \lVert p - r_j \rVert)}{\sqrt{\lVert p - r_j \rVert^2 + \varepsilon^2}}$$

where $\varepsilon$ is the softening length and $\sigma(c - d)$ regulates the interaction reach across the board.

### 2. The Tidal Tensor & Line of Failure
A King is not destroyed by simple attraction, but by **differential gravitational stretching (tide)** across its immediate neighborhood. Kepler-64 calculates the Hessian matrix of the potential at each King's location:

$$\mathbf{A} = \nabla\nabla\Phi\big|_{\text{King}} = \begin{pmatrix} \Phi_{xx} & \Phi_{xy} \\ \Phi_{yx} & \Phi_{yy} \end{pmatrix}$$

The eigenvalues $(\lambda_1, \lambda_2)$ describe the principal stretching and compression axes, while the eigenvector $\vec{v}_1$ points along the **Line of Failure** through the King's defense.

### 3. The Roche Disruption Win Condition ($\eta$)
In celestial mechanics, when an orbiting body passes within the **Roche limit**, tidal forces overpower its self-gravity and pull it apart. Kepler-64 expresses this as a dimensionless index $\eta$:

$$\eta = \frac{R_g^3 \cdot \lambda_1}{m_{\text{ref}}^2}$$

When $\eta > \rho_{\text{roche}}$, the King undergoes **Roche Disruption**—a physical checkmate.

### 4. Capture Accretion
When a piece captures an opponent, mass is preserved: the capturing piece **absorbs 80% of the captured piece's mass**:

$$M_{\text{capturer}} \leftarrow M_{\text{capturer}} + 0.8 \cdot M_{\text{captured}}$$

A Queen that has captured multiple rooks and minor pieces becomes a dense gravitational well, exerting stronger pull across the board.

</details>

---

## ⚡ The Laws Were Learned, Not Written

The physical constants of this universe are **learnable parameters** optimized via automatic differentiation in JAX:

$$\theta = \{ G, \; \varepsilon, \; c, \; \rho_{\text{roche}}, \; R_g, \; k_{\text{gain}}, \; \gamma, \; \dots \}$$

Using gradient descent on master games, Kepler-64 tunes these constants by balancing game outcomes and expert move rankings:

$$\mathcal{L} = \mathcal{L}_{\text{outcome}}(\sigma(S(p)), y) + \alpha \mathcal{L}_{\text{policy}}(S(p_{\text{expert}}), S(p_{\text{legal}}))$$

Gradients flow directly through the analytical potential equations and eigenvalue calculations.

---

## 🚀 Try It in 3 Lines

### Installation

```bash
git clone https://github.com/r-baruah/kepler-64.git
cd kepler-64
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Watch the universe make its first move

```python
import chess
from kepler64 import RocheEngine
from kepler64.core.fastboard import FastBoard

engine = RocheEngine()                       # creates the universe (auto-loads trained laws)
board = chess.Board()
physics = FastBoard.from_chess(board)        # the board as the physics sees it

print(f"Gravitational verdict: {engine.evaluate(physics):+.3f}")

move = engine.play(board, depth=2)           # the universe picks a move
print("Kepler-64 plays:", board.san(move))
```

Want the full visual experience?

```bash
cd viz-web && npm install && npm run dev   # → http://localhost:5173
```

---

## 🗺️ Project Map

```text
kepler-64/
├── kepler64/                    # The engine (pure Python + JAX)
│   ├── core/                    #   THE LAWS: gravity, tidal tensor, evaluation,
│   │                            #   constants (17 learnable leaves), accretion,
│   │                            #   Lorentz boost, Schwarzschild & wave terms
│   ├── search/                  #   Alpha-beta search over the physics
│   ├── training/                #   Learning the laws from games (JAX autodiff)
│   ├── multiverse/              #   Layer 2: posterior sampling & Observer
│   ├── match/                   #   UCI harness for measuring vs real engines
│   └── tests/                   #   62 tests — every law has a regression test
│
├── Kepler-64_Book/              # 📖 The complete textbook (start at PRELUDE.md)
│
├── viz-web/                     # 🔭 Web Observatory (TypeScript + Canvas)
│   ├── src/core/                #   TypeScript ports of the gravitational math
│   ├── src/render/              #   Contours, tidal ellipses, timelines
│   └── src/ui/                  #   Board, timeline, compendium UI
│
├── scripts/
│   ├── credibility_gate.py      # Learned-vs-frozen ablation harness
│   ├── distill_posterior.py     # Per-leaf sensitivity map
│   └── generate_replay.py       # Replay data for the Observatory
│
├── docs/                        # Audit reports, gate results, design notes
├── REFERENCES.md                # Scientific bibliography
├── CITATION.cff                 # Citation metadata
└── README.md
```

---

## 🧪 Testing & Verification

Kepler-64 includes a test suite covering force symmetry, eigenvalue calculations against SciPy, and mass accretion rules:

```bash
pytest kepler64/tests/ -v
```

The same suite runs automatically on every push and pull request via [`​.github/workflows/tests.yml`](.github/workflows/tests.yml), followed by a one-move engine smoke test — so "tests passing" is a live claim, not a memory.

### Learned vs Frozen Physics — the Credibility Gate (pilot)

The central claim — *"the constants learned by gradient descent do real work"* — is checked by an automated ablation (`scripts/credibility_gate.py`): self-play harvest → train all 17 leaves → held-out ranking metrics → color-balanced head-to-head match. **Pilot run (84 self-play examples, 300 Adam steps, 8-game match):**

| Metric | Frozen | Learned |
|---|---|---|
| Head-to-head (W/D/L) | — | **8 / 0 / 0** |
| Held-out MRR | 0.683 | 0.619 |

**Honest read:** the harness works end-to-end and the learned universe won its pilot match, but 84 examples cannot constrain 17 constants (held-out ranking regressed — small-data overfitting) and n=8 games is far too few to claim an Elo edge. Scaled runs are pending; until then this table is a demonstration of method, not a result. Full numbers: [`docs/credibility_gate_results.md`](docs/credibility_gate_results.md).

---

## ❓ Frequently Asked Questions

**Is this real physics?**
Real *equations*, honestly computed — Newtonian gravity with Plummer softening, real tidal tensors, real eigenvalues, a real symplectic integrator. But the *setting* is an engineered analogy: flat 2D board, point-mass pieces. The project never claims chess is literally astrophysics; the Book has an entire honesty chapter about exactly this.

**So… is it actually good at chess?**
It plays legal, coherent, occasionally creative chess. Against Stockfish? It loses, and proudly so — that was never the point. The point is that every single evaluation can be *explained*: "this move is better because it increases the tidal stretching of the enemy King by X." Try getting that out of a neural network.

**How does it learn without being told anything about chess?**
Two signals from real games: *who won* (outcome) and *which move a deeper search would pick* (policy). Gradient descent then adjusts the 17 physical constants until the universe's judgment matches reality. No chess rules are ever written in — if the engine develops a liking for central control, gravity discovered it.

**Can I watch what it's thinking?**
Yes — that's the whole philosophy. The [Web Observatory](https://r-baruah.github.io/kepler-64/) shows the live gravitational field and tidal stress every move, and the [Book](Kepler-64_Book/PRELUDE.md) teaches you to read them from zero.

---

## 🔬 The Bigger Questions

1. **Emergent strategy:** Can gradient descent through gravitational fields discover positional ideas — center control, pawn structure — with zero human chess knowledge?
2. **Tidal warnings:** Does rising tidal stress predict tactical disasters *before* material is lost?
3. **Accretion dynamics:** How do openings change when captures make you heavier instead of just richer?

---

## 👤 Author

**Kepler-64** was created by:

- **Ripuranjan Baruah** — *Original Creator & Lead Architect*
  - GitHub: [@r-baruah](https://github.com/r-baruah)

Contributions and discussions are welcome via GitHub Issues and Pull Requests.

---

## 📖 Citation

If you use Kepler-64 in your research or projects, please cite:

```bibtex
@software{baruah2026kepler64,
  author       = {Baruah, Ripuranjan},
  title        = {Kepler-64: Differentiable N-Body Gravitational Chess Engine},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/r-baruah/kepler-64}}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
