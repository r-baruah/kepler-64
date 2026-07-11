"""Glass Box visualizer — the GitHub star magnet.

Renders: potential-well heatmap, force-field quiver plot, and the red
"line of failure" eigenvector through the enemy King. Outputs a PNG per position
and can animate a game into a GIF.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from ..core.gravity import force_field, potential_field
from ..core.tidal import tidal_tensor_at


def render_position(board, constants, out_path: str = "frame.png"):
    masses = np.asarray(board.mass_vector())
    abs_m = jnp.asarray(np.abs(masses))
    F = np.asarray(force_field(abs_m, constants.eps, constants.G))
    U = np.asarray(potential_field(abs_m, constants.eps, constants.G)).reshape(8, 8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(U, cmap="inferno", origin="lower")
    Y, X = np.mgrid[0:8, 0:8]
    Fg = F.reshape(8, 8, 2)
    ax.quiver(X, Y, Fg[..., 0], Fg[..., 1], color="cyan", alpha=0.5, pivot="mid")

    bk = int(np.argmax((np.abs(masses) - 1000 < 0.5) & (np.sign(masses) == -1)))
    if 0 <= bk < 64 and np.abs(masses[bk]) > 500:
        A = tidal_tensor_at(jnp.asarray(U), bk)
        w, _ = np.linalg.eigh(np.asarray(A))
        vec = w[:, 0]
        kx, ky = bk % 8, bk // 8
        ax.plot([kx - 2 * vec[0], kx + 2 * vec[0]], [ky - 2 * vec[1], ky + 2 * vec[1]], "r-", lw=3)
        ax.plot(kx, ky, "r*", ms=14)

    ax.set_title("Kepler-64: potential well + line of failure")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    fig.colorbar(plt.cm.ScalarMappable(cmap="inferno"), ax=ax, label="U")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def game_gif(engine, out_path: str = "kepler64.gif", max_ply: int = 40):
    import chess

    from ..core.board import Board

    b = chess.Board()
    frames = []
    for ply in range(max_ply):
        if b.is_game_over():
            break
        mv = engine.play(Board.from_chess(b))
        if mv is None:
            break
        b.push(mv)
        fp = f"frame_{ply:03d}.png"
        render_position(Board.from_chess(b), engine.constants, fp)
        frames.append(fp)
    try:
        import imageio

        imageio.mimsave(out_path, [imageio.imread(f) for f in frames], duration=0.5)
    except Exception:
        pass  # frames remain as PNGs if imageio is unavailable
    return out_path
