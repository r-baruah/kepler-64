"""Glass Box visualizer — two-panel render.

Layout
------
LEFT  — Real chess board.  Standard Lichess square colours, Unicode piece
        glyphs rendered via matplotlib text (correct orientation, no raster
        flip issues), last-move highlight, check ring.

RIGHT — Gravitational field portrait (physics-paper aesthetic):
          • Plummer potential Φ as a continuous 2-D heatmap (viridis_r)
            clipped to the 3rd–97th percentile so minor-piece structure
            is legible while the king wells dominate at the extremes
          • Equipotential contour lines
          • Pieces as mass-scaled scatter dots
          • Tidal stress ellipse at each King: axes are the eigenvectors of
            the tidal tensor ∇∇Φ|_K, lengths ∝ |eigenvalue|, coloured by
            disruption level η/ρ_roche (blue → amber → red)

Public API (unchanged):
    render_position(board, constants, out_path, move_no, turn) → str
    game_gif(engine, out_path, max_ply, duration) → str
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import jax.numpy as jnp

from ..core.gravity import potential_field
from ..core.tidal import tidal_tensor_at
from ..core.evaluate import evaluate

try:
    import chess as _chess_mod
except ImportError:
    _chess_mod = None

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_SQ_LIGHT  = "#f0d9b5"           # classic Lichess light square
_SQ_DARK   = "#b58863"           # classic Lichess dark square
_FIG_BG    = "#f7f7f7"           # off-white figure background
_BOARD_BG  = "#f7f7f7"           # axes margin colour
_FRAME_COL = "#6b4c2a"           # board wood-frame colour

_W_PIECE   = "#ffffff"
_B_PIECE   = "#1c1c1c"

_GLYPHS = {
    (True,  1): "♙", (True,  2): "♘", (True,  3): "♗",
    (True,  4): "♖", (True,  5): "♕", (True,  6): "♔",
    (False, 1): "♟", (False, 2): "♞", (False, 3): "♝",
    (False, 4): "♜", (False, 5): "♛", (False, 6): "♚",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _king_sq(masses: np.ndarray, sign: float) -> Optional[int]:
    """Square index of the King of the given colour, or None.

    Consistency with the evaluator: use an atol=2.0 closeness test so
    accretion-shifted King masses (e.g. 1002.4) still match.
    """
    mask = np.isclose(np.abs(masses), 1000.0, atol=2.0) & (np.sign(masses) == sign)
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def _potential_on_grid(
    masses: np.ndarray, eps: float, G: float, c: float, n: int = 72
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plummer potential on an n×n continuous grid.

    Φ(p) = −G Σ_j |m_j| · σ(c − |p−r_j|) / √(|p−r_j|² + ε²)

    Returns (GX, GY, U) of shape (n, n).
    """
    abs_m  = np.abs(np.asarray(masses, dtype=np.float32))
    coords = np.array([(k % 8, k // 8) for k in range(64)], dtype=np.float32)
    g      = np.linspace(0.0, 7.0, n)
    GX, GY = np.meshgrid(g, g)
    dx = GX[:, :, None] - coords[None, None, :, 0]
    dy = GY[:, :, None] - coords[None, None, :, 1]
    d  = np.sqrt(dx**2 + dy**2)
    r  = np.sqrt(d**2 + eps * eps)
    gate = 1.0 / (1.0 + np.exp(-(c - d)))
    U  = -G * np.einsum("xyk,k->xy", gate / r, abs_m)
    return GX, GY, U


def _tidal_params(U64: np.ndarray, king_sq: int):
    """Eigensystem of the tidal tensor ∇∇Φ at king_sq.

    Returns (λ1, λ2, angle_deg, v1):
        λ1 ≥ λ2, v1 = unit stretching eigenvector, angle_deg = CCW from x.
    """
    A = np.asarray(tidal_tensor_at(jnp.asarray(U64), king_sq))
    vals, vecs = np.linalg.eigh(A)
    lam1, lam2 = float(vals[1]), float(vals[0])
    v1 = vecs[:, 1]
    angle_deg = float(np.degrees(np.arctan2(float(v1[1]), float(v1[0]))))
    return lam1, lam2, angle_deg, v1


def _eta_from_U64(U64: np.ndarray, king_sq: int, G: float, Rg: float = 1.0,
                  mref: float = 3.5) -> float:
    """Dimensionless tidal disruption index η at a king.

    MUST match `core.evaluate._eta`: η = Rg³ λ1 / mref² (NOT / Mking², and G is
    absent because λ1 already carries G). Keeping this identical to the
    evaluator prevents the red "danger" coloring from contradicting the score.
    """
    A = np.asarray(tidal_tensor_at(jnp.asarray(U64), king_sq))
    vals, _ = np.linalg.eigh(A)
    return (Rg**3) * float(vals[1]) / (mref**2 + 1e-9)


def _disruption_color(eta: float, roche: float) -> tuple[float, float, float]:
    """Map η/ρ_roche to RGB: steel-blue (safe) → amber → crimson (danger)."""
    t = float(np.clip(eta / (roche + 1e-9), 0.0, 1.0))
    # Three-stop gradient: blue → amber → red
    if t < 0.5:
        s = t * 2.0
        # blue (0.20, 0.60, 0.90) → amber (0.95, 0.65, 0.10)
        r = 0.20 + s * 0.75
        g = 0.60 + s * 0.05
        b = 0.90 - s * 0.80
    else:
        s = (t - 0.5) * 2.0
        # amber (0.95, 0.65, 0.10) → crimson (0.80, 0.10, 0.10)
        r = 0.95 - s * 0.15
        g = 0.65 - s * 0.55
        b = 0.10
    return (r, g, b)


# ---------------------------------------------------------------------------
# Panel A — Chess board
# ---------------------------------------------------------------------------

def render_board(ax, board, constants, last_move=None, check_sq=None) -> None:
    """Draw the chess position on *ax* using plain matplotlib primitives.

    Squares: filled Rectangle patches.
    Pieces:  ax.text() with a contrasting stroke — no raster compositing,
             no orientation issues, identical quality at any DPI.
    """
    masses = np.asarray(board.mass_vector())
    cb = getattr(board, "_chess", None)

    # --- squares ---
    for sq in range(64):
        x, y = sq % 8, sq // 8
        col = _SQ_LIGHT if (x + y) % 2 == 0 else _SQ_DARK
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=col, zorder=0))

    # --- last-move highlight (yellow tint + gold arrow) ---
    if last_move is not None:
        for s in (last_move.from_square, last_move.to_square):
            x, y = s % 8, s // 8
            ax.add_patch(plt.Rectangle(
                (x - 0.5, y - 0.5), 1, 1, color="#f6e27a", alpha=0.50, zorder=1))
        fx, fy = last_move.from_square % 8, last_move.from_square // 8
        tx, ty = last_move.to_square   % 8, last_move.to_square   // 8
        ax.annotate("", xy=(tx, ty), xytext=(fx, fy), zorder=2,
                    arrowprops=dict(arrowstyle="->", color="#c8a400",
                                   lw=2.5, mutation_scale=18))

    # --- check ring ---
    if check_sq is not None:
        cx, cy = check_sq % 8, check_sq // 8
        ax.add_patch(plt.Circle(
            (cx, cy), 0.44, fill=False, edgecolor="#cc2200", lw=3.0, zorder=3))

    # --- piece glyphs via ax.text ---
    for sq in range(64):
        x, y = sq % 8, sq // 8
        if cb is not None:
            if _chess_mod is None:
                continue
            pc = cb.piece_at(sq)
            if pc is None:
                continue
            is_white = (pc.color == _chess_mod.WHITE)
            glyph    = _GLYPHS[(is_white, pc.piece_type)]
        else:
            pt = int(np.asarray(board.pieces)[sq])
            if pt == 0:
                continue
            is_white = bool(masses[sq] > 0)
            glyph    = _GLYPHS[(is_white, pt)]
        fg = _W_PIECE if is_white else _B_PIECE
        bg = "#1c1c1c" if is_white else "#e8e8e8"
        ax.text(x, y, glyph,
                ha="center", va="center_baseline",
                fontsize=28, zorder=4, color=fg,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=bg)])

    # --- wood border ---
    ax.add_patch(plt.Rectangle(
        (-0.5, -0.5), 8, 8, fill=False,
        edgecolor=_FRAME_COL, linewidth=3.5, zorder=5))

    # --- axes ---
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.set_facecolor(_BOARD_BG)
    ax.set_xticks(range(8))
    ax.set_xticklabels(list("abcdefgh"),
                       fontsize=8.5, color="#5a5a5a", fontfamily="monospace")
    ax.set_yticks(range(8))
    ax.set_yticklabels([str(i) for i in range(1, 9)],
                       fontsize=8.5, color="#5a5a5a", fontfamily="monospace")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(_FRAME_COL)
        spine.set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Panel B — Gravitational field portrait
# ---------------------------------------------------------------------------

def render_field(ax, board, constants):
    """Gravitational field portrait on *ax*.

    Renders:
      1. Plummer potential Φ as a 2-D heatmap (viridis_r, percentile-clipped)
      2. Equipotential contour lines
      3. Pieces as mass-scaled scatter dots
      4. Tidal stress ellipse + stretching-axis arrow at each King
    """
    masses = np.asarray(board.mass_vector())

    # -- 1. Potential heatmap -------------------------------------------------
    GX, GY, U = _potential_on_grid(
        masses, constants.eps, constants.G, constants.c, n=64)

    # Percentile clip so minor-piece wells stay visible alongside the
    # king's ~1000× stronger well.
    p3, p97 = np.percentile(U, [3.0, 97.0])
    im = ax.imshow(
        U, origin="lower", extent=[-0.5, 7.5, -0.5, 7.5],
        cmap="plasma", aspect="equal", interpolation="bilinear",
        vmin=p3, vmax=p97, zorder=0)

    # -- 2. Equipotential contours --------------------------------------------
    Umin, Umax = float(U.min()), float(U.max())
    if Umax > Umin:
        levels = np.linspace(Umin, Umax, 16)[1:-1]
        ax.contour(GX, GY, U, levels=levels,
                   colors="white", alpha=0.18, linewidths=0.45, zorder=1)

    # -- 3. Piece scatter dots ------------------------------------------------
    # Radius ∝ √(|m|/9): queen (9) is the largest non-king dot.
    for sq in range(64):
        m = float(masses[sq])
        if m == 0.0:
            continue
        x, y   = sq % 8, sq // 8
        m_abs  = abs(m)
        is_king = m_abs > 500

        if is_king:
            size = 220
            fc   = "#ffffff" if m > 0 else "#1c1c1c"
            ec   = "#aad4f5" if m > 0 else "#f5aaaa"
            lw   = 1.8
        else:
            size = max(18.0, 38.0 * np.sqrt(m_abs / 9.0))
            fc   = "#e8e8e8" if m > 0 else "#2c2c2c"
            ec   = "#888888" if m > 0 else "#888888"
            lw   = 0.8

        ax.scatter(x, y, s=size, c=fc, edgecolors=ec,
                   linewidths=lw, zorder=5, alpha=0.90)

    # -- 4. Tidal stress ellipses at kings ------------------------------------
    # Engine's own 8×8 lattice potential — same values the search uses.
    U64_lattice = np.asarray(
        potential_field(
            jnp.asarray(np.abs(masses)),
            constants.eps, constants.G, constants.c))   # (64,)

    king_tidal: dict[float, tuple] = {}
    abs_lams: list[float] = []
    for sign in (1.0, -1.0):
        sq = _king_sq(masses, sign)
        if sq is None or abs(float(masses[sq])) < 500:
            continue
        try:
            l1, l2, ang, v1 = _tidal_params(U64_lattice, sq)
            eta = _eta_from_U64(U64_lattice, sq, constants.G,
                                getattr(constants, "Rg", 1.0),
                                getattr(constants, "mref", 3.5))
        except Exception:
            continue
        king_tidal[sign] = (sq, l1, l2, ang, v1, eta)
        abs_lams.extend([abs(l1), abs(l2)])

    if abs_lams:
        max_lam = max(abs_lams)
        SCALE   = 1.6   # max semi-axis in board units

        for sign, (sq, l1, l2, ang, v1, eta) in king_tidal.items():
            kx, ky = sq % 8, sq // 8
            a = max(SCALE * abs(l1) / (max_lam + 1e-9), 0.12)
            b = max(SCALE * abs(l2) / (max_lam + 1e-9), 0.05)
            dc = _disruption_color(eta, constants.roche)

            ell = mpatches.Ellipse(
                xy=(kx, ky), width=2 * a, height=2 * b, angle=ang,
                fill=False, edgecolor=dc, linewidth=2.2, alpha=0.90, zorder=6)
            ax.add_patch(ell)

            # Arrow along the λ₁ (stretching) axis
            tip = np.array([kx + a * 0.78 * v1[0],
                            ky + a * 0.78 * v1[1]])
            ax.annotate("", xy=tip, xytext=(kx, ky), zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color=dc,
                                        lw=1.0, mutation_scale=9))

    # -- axes -----------------------------------------------------------------
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.set_facecolor(_FIG_BG)
    ax.set_xticks(range(8))
    ax.set_xticklabels(list("abcdefgh"),
                       fontsize=8.5, color="#5a5a5a", fontfamily="monospace")
    ax.set_yticks(range(8))
    ax.set_yticklabels([str(i) for i in range(1, 9)],
                       fontsize=8.5, color="#5a5a5a", fontfamily="monospace")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.6)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label(r"$\Phi\;[G \cdot m \cdot \ell^{-1}]$",
                   color="#555555", fontsize=7.5, labelpad=5)
    cbar.ax.tick_params(labelsize=6.5, colors="#555555", length=2)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#555555")

    return im


# ---------------------------------------------------------------------------
# Public: render_position
# ---------------------------------------------------------------------------

def render_position(
    board,
    constants,
    out_path: str = "frame.png",
    move_no: int | None = None,
    turn: str | None = None,
    eta_history: list | None = None,   # kept for API compat, not rendered
) -> str:
    """Render one position to *out_path* (PNG).

    Two-panel layout: chess board on the left, gravitational field on the right.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    cb        = getattr(board, "_chess", None)
    last_move = cb.move_stack[-1] if (cb is not None and cb.move_stack) else None
    check_sq  = cb.king(cb.turn) if (cb is not None and cb.is_check()) else None

    fig = plt.figure(figsize=(13.0, 6.5))
    fig.patch.set_facecolor(_FIG_BG)
    gs = fig.add_gridspec(
        1, 2, wspace=0.10, left=0.04, right=0.96, top=0.88, bottom=0.09)
    ax_board = fig.add_subplot(gs[0, 0])
    ax_field = fig.add_subplot(gs[0, 1])

    render_board(ax_board, board, constants, last_move, check_sq)
    render_field(ax_field, board, constants)

    # Panel titles
    board_title = "Position"
    if move_no is not None:
        board_title = f"Ply {move_no}"
    if turn is not None:
        side = "White" if turn == "white" else "Black"
        board_title += f"  ·  {side} to move"
    ax_board.set_title(board_title, color="#3a3a3a", fontsize=9, pad=5)
    ax_field.set_title(r"Gravitational field  $\Phi$",
                       color="#3a3a3a", fontsize=9, pad=5)

    # Eval + physics constants in the header
    eval_str = ""
    try:
        score = float(evaluate(board, constants))
        eval_str = f"    eval {score:+.3f}"
    except Exception:
        pass

    c = constants
    phys = (f"G = {c.G:.3f}    ε = {c.eps:.3f}    "
            f"c = {c.c:.2f} sq/ply    ρ_roche = {c.roche:.3f}")
    fig.text(0.50, 0.965, f"Kepler-64{eval_str}",
             ha="center", va="top", color="#1a1a2e",
             fontsize=11, fontweight="bold", fontfamily="monospace")
    fig.text(0.50, 0.947, phys,
             ha="center", va="top", color="#888899",
             fontsize=7.5, fontfamily="monospace")

    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public: game_gif
# ---------------------------------------------------------------------------

def game_gif(
    engine,
    out_path: str = "kepler64.gif",
    max_ply: int = 40,
    duration: float = 0.55,
) -> str:
    """Play a game with *engine* and render each ply to an animated GIF."""
    import chess as _c
    import tempfile
    from ..core.board import Board

    b   = _c.Board()
    frames: list[str] = []
    tmp = tempfile.mkdtemp(prefix="kepler64_gif_")

    for ply in range(max_ply):
        if b.is_game_over():
            break
        mv = engine.play(b)
        if mv is None:
            break
        b.push(mv)

        brd  = Board.from_chess(b)
        turn = "white" if b.turn else "black"
        fp   = f"{tmp}/frame_{ply:03d}.png"
        render_position(brd, engine.constants, fp,
                        move_no=ply + 1, turn=turn)
        frames.append(fp)

    try:
        import imageio
        from PIL import Image
    except ImportError:
        warnings.warn(
            "imageio/pillow not installed — PNG frames were written but no GIF "
            "was assembled.  Run `pip install imageio pillow` to assemble.",
            stacklevel=2)
        return frames  # type: ignore[return-value]

    try:
        W = H = 0
        for f in frames:
            im = Image.open(f)
            W, H = max(W, im.width), max(H, im.height)
        seq = [np.array(Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS))
               for f in frames]
        imageio.mimsave(out_path, seq, duration=duration, loop=0)
    except Exception as exc:
        warnings.warn(f"GIF assembly failed ({exc}); PNG frames retained.",
                      stacklevel=2)
    return out_path


# ---------------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------------

def render_sheet(ax, board, constants):
    """Deprecated — delegates to render_field."""
    warnings.warn("render_sheet() is deprecated; use render_field() instead.",
                  DeprecationWarning, stacklevel=2)
    return render_field(ax, board, constants)


def render_eta_timeline(ax, eta_history, roche: float) -> None:
    """Deprecated — timeline strip removed; this is now a no-op."""
    ax.axis("off")
