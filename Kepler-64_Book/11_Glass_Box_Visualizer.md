# Chapter 11 — The Glass Box Visualizer

> **Read me first (no background needed).** Most chess engines are black boxes: they say "the best move is Nf3" and that's it — trust us. Kepler-64 is built to be the opposite: a **glass box** where you can literally watch the reasoning. The Glass Box visualizer draws every game as two panels side by side: on the left, the ordinary chessboard; on the right, the same position rendered as what it *physically is* — a gravitational landscape with deep wells under heavy pieces, contour lines like a weather map, and at each King a colored ellipse that stretches along the exact axis where the tidal pull threatens to tear it. When the engine says "this move attacks your King," you can see the ellipse turn red.

## 11.1 Intuition: show, don't tell

The single most persuasive artifact of Kepler-64 is not a number — it's a **GIF**: a real game where, on the right panel, you watch the gravitational potential heat-map deepen as pieces cluster, and a red ellipse at each King stretch along the exact axis of failure. A skeptical reviewer who won't read the JAX can *see* the physics working.

The project calls this the **Glass Box** — the engine's reasoning made visible (`Kepler-64 Audit` "what would ship": the visualizer is the highest-ROI feature).

> **Intuition box:** A weather map doesn't explain meteorology, but you immediately grasp "storm here, calm there." The Glass Box is the weather map of the chess universe.

## 11.2 What the two panels show

`render_position` (`viz/glassbox.py:357-414`) draws a figure with:

- **Left — the real board:** standard squares, Unicode piece glyphs, last-move arrow, check ring.
- **Right — the gravitational field portrait** (`render_field`, `viz/glassbox.py:229-350`):
  1. **Potential heat-map** $\Phi$ on a $64\times 64$ continuous grid, percentile-clipped (3rd–97th) so minor-piece wells stay visible next to the King's 1000× well (`viz/glassbox.py:241-250`).
  2. **Equipotential contours** (white lines).
  3. **Piece dots** scaled by $\sqrt{|m|/9}$ (queen largest non-king dot).
  4. **Tidal stress ellipses** at each King: semi-axes ∝ $|\lambda_1|,|\lambda_2|$, colored by $\eta/\rho_{\text{roche}}$ (blue safe → amber → crimson danger), with a red arrow along the $\lambda_1$ "line of failure" (`viz/glassbox.py:283-325`).

The header prints the live eval and the physics constants ($G, \varepsilon, c, \rho_{\text{roche}}$).

## 11.3 The math behind the render

The heat-map potential on a fine grid (`_potential_on_grid`, `viz/glassbox.py:79-98`) re-evaluates the Plummer potential at $n\times n$ continuous points (same $c$-gate formula as §2.4/§5.2). The ellipse uses the **eigensystem** of the tidal tensor exactly as Chapter 3: `eig2x2` / `np.linalg.eigh` gives $\lambda_1,\lambda_2$ and the stretching eigenvector; that eigenvector's angle sets the red arrow direction.

The disruption color maps the **ratio** $\eta/\rho_{\text{roche}}$ to RGB (`_disruption_color`, `viz/glassbox.py:122-138`) — a three-stop gradient matching the "safe → danger" semantics of the score.

## 11.4 Visualizer η vs evaluator η — resolved

Both now use the same formula: $\eta = R_g^3\lambda_1/m_{\text{ref}}^2$. The evaluator computes it in `_eta` (`core/evaluate.py`); the visualizer's `_eta_from_U64` (`viz/glassbox.py`) applies the identical expression with `constants.mref`, so the ellipse danger-coloring agrees with the score's disruption term.

> **(Resolved, verified 2026-08-23)** The former [ISSUE: VIZ-ETA-MISMATCH] — the visualizer dividing by the old $G\,M_{\text{king}}^2$ — is closed (C12 in Ch.14). A reviewer checking `viz/glassbox.py` against `core/evaluate.py` will find matching denominators and the explicit comment "MUST match `core.evaluate._eta`".

## 11.5 GIF assembly and robustness

`game_gif` (`viz/glassbox.py:421-472`) plays a full game with the engine, renders each ply to PNG, then assembles a GIF with `imageio`. It warns (doesn't crash) if `imageio`/`Pillow` are missing, and catches assembly errors. The matplotlib import is at module level (`viz/glassbox.py:29`) — acceptable since viz is opt-in, but Code Review v1 ISSUE 25 suggests lazy-importing it to keep non-viz imports light.

## 11.6 Project link

`RocheEngine.play` → `game_gif(engine, out_path="kepler64.gif")` produces the showcase (`README.md` Usage). This is the artifact to lead with in any writeup.

## 11.7 Forward link

Chapter 12 (relabeled) covers training through physics. Then Chapter 13 covers Layer 2. Then the critique.

**Cross-references:** Potential/force → §2. Tidal tensor/η → §3, §4. Score → §7.
