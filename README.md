<div align="center">

# 🪐 Kepler-64
### *What if gravity could play chess?*

A differentiable chess engine built with JAX where pieces exert mass, spacetime curves across the 64 squares, and the enemy King is defeated not by conventional checkmate heuristics, but by **tidal disruption past the Roche limit**.

[![Live Observatory](https://img.shields.io/badge/🔭_Live_Observatory-Deploy-2448b8?style=for-the-badge)](https://r-baruah.github.io/kepler-64/)
[![Tests](https://img.shields.io/badge/tests-38%20passing-16a34a?style=for-the-badge)](#testing--verification)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-differentiable-f06426?style=for-the-badge&logo=google)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

[**Explore Live Web Observatory**](https://r-baruah.github.io/kepler-64/) • [**Quickstart**](#quickstart) • [**How It Works**](#how-gravity-plays-chess) • [**Scientific References**](REFERENCES.md) • [**Cite Project**](#citation)

</div>

---

## 🔭 The Live Web Observatory

Explore the gravitational field and watch games unfold directly in your browser:

👉 **[https://r-baruah.github.io/kepler-64/](https://r-baruah.github.io/kepler-64/)**

- **Gravitational Potential Wells:** Watch the 2D field contours warp in real-time as pieces move across the board.
- **King Tidal Tension:** Visualizes directional stretching forces acting on both Kings on every turn.
- **Game Trajectory Timeline:** Follows the gravitational advantage and energy shifts move-by-move.
- **Interactive Compendium:** Illustrated companion guide explaining the underlying physics and formulas.
- **Export Animated Clips:** Download animated GIFs of positions and games with live evaluations.

---

## 🌌 How Gravity Plays Chess

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

---

## ⚡ Differentiable Learning via JAX

The physical constants of this universe are **learnable parameters** optimized via automatic differentiation in JAX:

$$\theta = \{ G, \; \varepsilon, \; c, \; \rho_{\text{roche}}, \; R_g, \; k_{\text{gain}}, \; \gamma, \; \dots \}$$

Using gradient descent on master games, Kepler-64 tunes these constants by balancing game outcomes and expert move rankings:

$$\mathcal{L} = \mathcal{L}_{\text{outcome}}(\sigma(S(p)), y) + \alpha \mathcal{L}_{\text{policy}}(S(p_{\text{expert}}), S(p_{\text{legal}}))$$

Gradients flow directly through the analytical potential equations and eigenvalue calculations.

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/r-baruah/kepler-64.git
cd kepler-64
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 3-Line Python Demo

```python
import chess
from kepler64 import RocheEngine, Board

# Initialize the engine
engine = RocheEngine()
board = Board.from_chess(chess.Board())

# Evaluate the position's gravitational curvature
score = engine.evaluate(board)
print(f"Position Score: {score}")

# Search for the best move
best_move = engine.play(board, depth=2)
print(f"Best Move: {best_move}")
```

### Running the Web Observatory Locally

```bash
cd viz-web
npm install
npm run dev
```
Open **`http://localhost:5173/`** in your browser.

---

## 📁 Repository Structure

```text
kepler-64/
├── kepler64/                 # Core Python Engine & JAX Math
│   ├── core/
│   │   ├── board.py          # Board state and mass vector pipeline
│   │   ├── constants.py      # Learnable physical constants (G, eps, c, roche)
│   │   ├── gravity.py        # Plummer potential & pairwise distance solvers
│   │   ├── tidal.py          # 2x2 Hessian eigenvalues and tidal ratio eta
│   │   ├── evaluate.py       # Position score breakdown
│   │   └── transitions.py    # Capture mass accretion mechanics
│   ├── search/
│   │   └── minimax.py        # Alpha-beta search and move ranking
│   ├── training/
│   │   ├── train.py          # JAX autodiff optimization
│   │   └── loss.py           # Outcome and policy loss functions
│   └── viz/
│       └── glassbox.py       # Matplotlib potential visualization
│
├── viz-web/                  # Web Observatory & Interactive Companion
│   ├── src/
│   │   ├── core/             # TypeScript ports of gravitational algorithms
│   │   ├── render/           # Canvas renderers for contours and tidal ellipses
│   │   ├── ui/               # UI components, timeline, and KaTeX guides
│   │   └── style.css         # Typography, layout, and visual styles
│   └── public/               # Piece SVGs and icons
│
├── .github/workflows/        # Automated deployment to GitHub Pages
├── CITATION.cff              # Citation metadata
├── REFERENCES.md             # Scientific bibliography & references
└── README.md
```

---

## 🧪 Testing & Verification

Kepler-64 includes a test suite covering force symmetry, eigenvalue calculations against SciPy, and mass accretion rules:

```bash
pytest kepler64/tests/ -v
```

---

## 🔬 Research Inquiries

Kepler-64 is an exploratory project investigating physics-inspired machine learning:

1. **Emergent Strategy:** Can gradient descent on gravitational fields learn positional ideas like center control and pawn structure without human-crafted rules?
2. **Tidal Indicators:** Does the tidal stress parameter $\eta$ predict tactical blunders before material is lost?
3. **Accretion Dynamics:** How do opening lines change when pieces gain mass and exert stronger pull after captures?

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
