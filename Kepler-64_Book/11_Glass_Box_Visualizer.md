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

## 11.4 Inconsistency: visualizer η vs evaluator η

The evaluator's η uses $\eta = R_g^3\lambda_1/m_{\text{ref}}^2$ (§4.3). The visualizer's `_eta_from_U64` (`viz/glassbox.py:115-119`) still uses the **old** formula $\eta = R_g^3\lambda_1/(G\,M_{\text{king}}^2)$ — note the denominator $G\cdot 1000^2$. These two **disagree** numerically: the ellipse color (driven by the old formula) will not match the score's disruption term (driven by the new $m_{\text{ref}}$ formula).

> **⚠ [ISSUE: VIZ-ETA-MISMATCH] (P2, new finding):** The visualizer and evaluator compute η with different denominators, so the red "danger" coloring can contradict the engine's own score. Fix: have `_eta_from_U64` call the same `_eta` used by `evaluate.py` (with `mref`), or at minimum document that the visualizer uses a display-scale approximation. This is the kind of detail a physics reviewer will spot immediately.

## 11.5 GIF assembly and robustness

`game_gif` (`viz/glassbox.py:421-472`) plays a full game with the engine, renders each ply to PNG, then assembles a GIF with `imageio`. It warns (doesn't crash) if `imageio`/`Pillow` are missing, and catches assembly errors. The matplotlib import is at module level (`viz/glassbox.py:29`) — acceptable since viz is opt-in, but Code Review v1 ISSUE 25 suggests lazy-importing it to keep non-viz imports light.

## 11.6 Project link

`RocheEngine.play` → `game_gif(engine, out_path="kepler64.gif")` produces the showcase (`README.md` Usage). This is the artifact to lead with in any writeup.

## 11.7 Forward link

Chapter 12 (relabeled) covers training through physics. Then Chapter 13 covers Layer 2. Then the critique.

**Cross-references:** Potential/force → §2. Tidal tensor/η → §3, §4. Score → §7.
