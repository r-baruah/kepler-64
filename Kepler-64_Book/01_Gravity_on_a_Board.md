# Chapter 1 — Gravity on a Chessboard: Newtonian Attraction as an Evaluation Function

> **Part A — Foundations.** Before we describe the engine's architecture, we must establish the physics. This chapter and the next five teach gravity, softening, the tidal tensor, Roche scaling, retarded potentials, and automatic differentiation — all from zero. The architecture chapter (Part B) only *assembles* these pieces; it introduces no new concept.

---

## 1.1 Intuition: every piece is a little planet

Imagine the chessboard is a flat sheet of space. Each piece placed on a square is a small planet with mass. A pawn is light; a queen is heavy; the king is *enormously* heavy — its mass is set so large that, locally, it behaves like a tiny star. Just as real planets pull on one another with gravity, every piece on the board pulls on every other piece and on every empty square.

Now here is the key idea: instead of asking "is White's position good?" the way a normal engine does (with material counts and king-safety tables), Kepler-64 asks a *physical* question — **"what is the gravitational field across this board, and is the enemy King being torn apart by it?"**

If you pile heavy pieces (rooks, queens) near the enemy King, you create a deep gravitational well and a strong stretching force right where the King sits. In Kepler-64, that stretching force *is* the threat. A human would say "the King is under attack." Kepler-64 says "the tidal eigenvalue at the King's square exceeds the Roche limit." Same thing, different language.

> **Intuition box:** Pieces are masses; gravity is the evaluation. Concentrating mass near a King literally stretches its position apart — that is how the engine "attacks."

## 1.2 The math: Newton's law of universal gravitation

Newton's law says the gravitational force between two point masses is

$$F = G\,\frac{m_1\,m_2}{r^2}$$

where:

- $G$ is the **gravitational constant** — a single number that sets how strong gravity is in this universe. In Kepler-64 it is *learned*, not hand-picked.
- $m_1, m_2$ are the two masses.
- $r$ is the distance between them.

Force is a **vector** — it has both magnitude and direction. The full vector form points along the line joining the two masses:

$$\vec F_{ij} = G\,\frac{m_i\,m_j}{r_{ij}^2}\,\hat r_{ij}$$

where $\hat r_{ij}$ is the unit vector from mass $i$ to mass $j$. The force on $i$ points *toward* $j$ (attraction). In component form, with positions $\vec r_i, \vec r_j \in \mathbb{R}^2$:

$$\vec F_i = G \sum_{j \neq i} m_j\,\frac{\vec r_i - \vec r_j}{\lVert \vec r_i - \vec r_j\rVert^3}$$

(We moved the cube into the denominator because dividing a vector by $r^2$ and multiplying by the unit direction $\frac{\vec r_i-\vec r_j}{r}$ is the same as dividing by $r^3$.) For a chessboard, each square $i$ is a point $(x_i, y_i)$ in a $2$-dimensional grid with coordinates $0\ldots7$.

**Worked example (2 pieces).** Put a white queen (mass 9) at square a1 = $(0,0)$ and a black pawn (mass 1) at square e1 = $(4,0)$. The pawn feels a force toward the queen:

$$\vec r_i-\vec r_j = (4,0)-(0,0)=(4,0),\quad \lVert\cdot\rVert = 4$$

$$\vec F_{\text{pawn}} = G\cdot 1 \cdot \frac{(4,0)}{4^3} = G\,\frac{(4,0)}{64} = G\,(0.0625,\,0)$$

So the pawn is pulled leftward toward the queen with magnitude $0.0625\,G$. Direction matters: in the engine, the *vector field* at every square tells you which way "downhill" the gravitational well points.

> **Touch it — real numbers from the real kernel.** Don't take the algebra on faith; here is actual output of `kepler64.core.gravity` for three masses stacked on the e-file — White pawn e4 (mass 1), White rook e5 (5), White queen e6 (9) — with the engine's default constants ($G{=}1$, $\varepsilon{=}0.5$):
>
> | Square | Force felt there ‖F‖ | Potential U |
> |---|---|---|
> | d4 (beside the pawn) | **4.19** | −8.15 |
> | d5 (beside the rook) | **6.96** | −11.14 |
> | d6 (beside the queen) | **8.17** | −11.82 |
> | g4 (far side of the board) | **2.16** | −5.80 |
>
> Read the pattern like a physicist: force roughly *doubles* as you climb the file past heavier masses, and the potential well deepens toward them (more negative = deeper). Distance dilutes everything — far-away g4 feels only a faint tug even from a queen. This inverse-square falloff is exactly why "concentrating heavy pieces near the enemy King" is a *physical* attack in Kepler-64, not just a chess proverb.

## 1.3 The math: G is a learned parameter (not a constant of nature)

In our universe $G \approx 6.674\times 10^{-11}\,\text{N·m}^2\text{/kg}^2$. In the Kepler-64 chess universe, $G$ is just a tunable number that gradient descent will move to make positions evaluate like good chess. In code (`core/constants.py:15`):

```python
G: float = 1.0          # initialized at 1.0
```

The engine never "knows" the real $G$. It discovers its own via training (see Chapter 13). The README's tagline — *"the gravitational constant of this chess universe was learned via gradient descent"* — is literally true: `G` is a leaf that `jax.grad` differentiates.

> **Project link:** Every force, potential, and tidal computation below multiplies by this single `G`. The whole pipeline is written so that changing `G` is a differentiable operation.

## 1.4 Limits of "real" gravity on a discrete board

Real gravity lives in continuous 3D space. A chessboard is a discrete 2D $8\times 8$ lattice. Three honest caveats (developed further in the critique, Chapter 18):

1. **Dimensionality.** We are in 2D, not 3D. The $1/r^2$ law is the 3D inverse-square law; on a 2D lattice the "natural" law is $1/r$. The engine uses the 3D-style formula anyway because it is the recognizable, trainable one (see `core/gravity.py:29`).
2. **Point masses on a grid.** A chess piece is a single point. There is no extended body to physically "tear." The tidal tensor is therefore a *mathematical proxy* for "directional stress," not literal spaghettification (see `Kepler-64 P II.md` §1).
3. **The King has mass 1000.** This is far larger than any real mass ratio and exists purely so the King's self-gravity dominates local pawns. It is a modeling choice, not physics (`core/constants.py:56`).

> **⚠ [ISSUE: METAPHOR-SCOPE] (awareness):** The project must never claim "real astrophysics." The README is correctly honest: *"It replaces heuristics with physics… not an approximation of physics. Physics."* This is rhetorically strong but technically a metaphor — the audit table (`Kepler-64 Audit & Layer-2 Development.md`, Honest/Dishonest Claims) explicitly rates "Real astrophysics" as **No**. Keep the disclaimer prominent.

## 1.5 Intuition recap and forward link

We now know: pieces are masses, gravity is computable as a vector field over 64 squares, and $G$ is learned. The next chapter fixes the one thing that breaks this naive formula — the **force singularity** when two masses sit on the same square — using Plummer softening, and shows how the board's fixed geometry lets us compute the whole field with a single `einsum`.

**Cross-references:** Plummer softening → §2. The force field as code → `core/gravity.py` (Chapter 7). The tidal tensor built from this field → §3.
