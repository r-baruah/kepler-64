"""Professional replay assets for recorded Kepler-64 games."""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import chess
import imageio.v2 as imageio
import numpy as np

from ..core.board import Board
from .glassbox import render_board, render_field


_TERM_LABELS = {
    "tidal_enemy": "Enemy king tide",
    "tidal_self": "Own king tide",
    "force_enemy_king": "Enemy king force",
    "force_own_king": "Own king force",
    "binding": "Binding",
    "material": "Material",
    "delta_tidal": "Delta tide",
    "delta_com": "Delta advance",
    "delta_inertia": "Delta pressure",
    "delta_entropy": "Delta coordination",
}

_REPOSITORY_URL = "https://github.com/r-baruah/kepler-64"
_FRAME_COLORS = {
    "paper": "#edf1f0",
    "white": "#fbfcfb",
    "ink": "#172235",
    "muted": "#526170",
    "rule": "#aab8bf",
    "plate": "#2448b8",
    "orange": "#f06426",
    "positive": "#2b9665",
    "negative": "#bd4032",
}


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _display_candidates(row, limit=8):
    candidates = row["analysis"]["candidates"]
    if len(candidates) <= limit:
        return candidates
    selected = next((c for c in candidates if c["move_uci"] == row["move_uci"]), None)
    shown = candidates[:limit]
    if selected is not None and selected not in shown:
        shown = [*candidates[:limit - 1], selected]
    return shown


def _mover_terms(row):
    sign = 1.0 if row["mover"] == "white" else -1.0
    return {key: sign * value for key, value in row["played_move_evaluation"]["terms"].items()}


def _material_balance(fen: str) -> float:
    """Signed total gravitational mass of a position (White positive)."""
    board = chess.Board(fen)
    mv = np.asarray(Board.from_chess(board).mass_vector(), dtype=np.float64)
    return float(mv.sum())


def _render_eval_bar(ax, score_white: float, palette: dict, clamp: float = 3.0):
    """Chess.com-style vertical evaluation bar on *ax*."""
    import matplotlib.patches as mpatches

    s = float(np.clip(score_white, -clamp, clamp))
    white_frac = 0.5 + 0.5 * (s / clamp)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#dbe1e6"))
    ax.add_patch(mpatches.Rectangle((0, 1 - white_frac), 1, white_frac,
                                    color="#f7f7f4"))
    ax.axhline(1 - white_frac, color=palette["ink"], lw=1.1)
    ax.axhline(0.5, color=palette["rule"], lw=0.6, ls=":")
    ax.text(0.5, 0.985, f"{score_white:+.2f}", color=palette["ink"],
            fontsize=8.5, weight="bold", ha="center", va="top",
            family="monospace")
    ax.text(0.5, 0.012, f"±{clamp:.0f}", color=palette["muted"], fontsize=6.5,
            ha="center", va="bottom", family="monospace")
    ax.axis("off")
    ax.set_title("Evaluation", color=palette["ink"], fontsize=9,
                 loc="left", weight="bold", pad=6)


def render_replay_frame(replay, index: int, constants, out_path) -> str:
    """Render one observation: a dominant board, a chess.com-style eval bar,
    a compact field plate, and a tiled decision record."""
    plt = _matplotlib()
    row = replay["plies"][index]
    board = chess.Board(row["fen_after"])
    wrapped = Board.from_chess(board)
    fig = plt.figure(figsize=(15, 10), facecolor=_FRAME_COLORS["paper"])
    gs = fig.add_gridspec(1, 3, width_ratios=[8.0, 0.45, 3.55], wspace=0.30,
                          left=0.05, right=0.96, top=0.80, bottom=0.05)
    ax_board = fig.add_subplot(gs[0, 0])
    ax_eval = fig.add_subplot(gs[0, 1])
    right = gs[0, 2].subgridspec(12, 1, hspace=0.45)
    ax_field = fig.add_subplot(right[0:4, 0])
    ax_stats = fig.add_subplot(right[4:12, 0])

    played_move = chess.Move.from_uci(row["move_uci"])
    render_board(ax_board, wrapped, constants,
                 played_move, board.king(board.turn) if board.is_check() else None,
                 palette={"square_light": _FRAME_COLORS["white"],
                          "square_dark": "#cbd7d9", "move_fill": "#f6a06c",
                          "move_line": _FRAME_COLORS["orange"],
                          "frame": _FRAME_COLORS["ink"],
                          "axis": _FRAME_COLORS["muted"],
                          "background": _FRAME_COLORS["paper"]})

    _render_eval_bar(ax_eval, row["eval_after"]["score_white"], _FRAME_COLORS)

    render_field(ax_field, wrapped, constants,
                 palette={"field_low": "#dfe8e8", "field_mid": _FRAME_COLORS["plate"],
                          "field_high": _FRAME_COLORS["ink"], "contour": "#ffffff",
                          "axis": "#ffffff", "background": _FRAME_COLORS["plate"]},
                 colorbar=False)
    ax_field.set_title("Gravitational field", color=_FRAME_COLORS["white"],
                       fontsize=9, loc="left", weight="bold")

    # ── Compact decision record ───────────────────────────────────────────
    mv = np.asarray(wrapped.mass_vector(), dtype=np.float64)
    mass_bal = float(mv[mv > 0].sum() + mv[mv < 0].sum())
    terms = row["played_move_evaluation"]["terms"]
    sign = 1.0 if row["mover"] == "white" else -1.0
    stats_rows = [
        ("MOVE", row["move_san"]),
        ("UCI", row["move_uci"]),
        ("MOVER", row["mover"]),
        ("MOVE NO.", str(row["move_number"])),
        ("RANK", f"{row['selected_candidate_rank'] or '—'} / {row['analysis']['legal_move_count']}"),
        ("DEPTH", str(row["analysis"]["depth"])),
        ("NODES", f"{row['analysis']['nodes']:,}"),
        ("ELAPSED", f"{row['analysis']['elapsed_ms']:.0f} ms"),
        ("EVAL SHIFT", f"{row['eval_shift_mover']:+.3f}"),
        ("MASS BAL", f"{mass_bal:+.1f}"),
        ("ENEMY TIDE", f"{sign * terms['tidal_enemy']:+.3f}"),
        ("OWN TIDE", f"{sign * terms['tidal_self']:+.3f}"),
    ]
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 2)
    ax_stats.set_ylim(0, 7)
    for r in range(1, 7):
        ax_stats.axhline(7 - r, color=_FRAME_COLORS["rule"], lw=0.6)
    ax_stats.axvline(1, color=_FRAME_COLORS["rule"], lw=0.6)
    for i, (label, value) in enumerate(stats_rows):
        col = i % 2
        r = i // 2
        x = col + 0.06
        y = 6.75 - r
        ax_stats.text(x, y, label, color=_FRAME_COLORS["muted"], fontsize=6.5,
                      family="monospace", va="top")
        ax_stats.text(col + 1.0 - 0.06, y, value,
                      color=_FRAME_COLORS["ink"], fontsize=9, weight="bold",
                      family="monospace", va="top", ha="right")
    fen = row["fen_after"]
    fen = fen[:48] + ("…" if len(fen) > 48 else "")
    ax_stats.text(0.06, 0.45, "FEN", color=_FRAME_COLORS["muted"], fontsize=6.5,
                  family="monospace", va="top")
    ax_stats.text(1.94, 0.45, fen, color=_FRAME_COLORS["muted"], fontsize=6.5,
                  family="monospace", va="top", ha="right")
    ax_stats.set_title("Decision record", color=_FRAME_COLORS["ink"],
                       fontsize=9, loc="left", weight="bold", pad=6)

    title = row["move_san"]
    meta = (f"rank {row['selected_candidate_rank'] or '—'} · depth {row['analysis']['depth']} · "
            f"{row['analysis']['nodes']:,} nodes · {row['analysis']['elapsed_ms']:.0f} ms")
    fig.text(0.05, 0.965, "KEPLER-64 / OBSERVATION RECORD",
             color=_FRAME_COLORS["plate"], fontsize=9, family="monospace", weight="bold")
    fig.text(0.05, 0.885, title, color=_FRAME_COLORS["ink"], fontsize=46, weight="bold")
    fig.text(0.21, 0.905, "A chess move evaluated as a gravitational event.",
             color=_FRAME_COLORS["muted"], fontsize=14)
    fig.text(0.96, 0.925, meta, color=_FRAME_COLORS["muted"], fontsize=9,
             ha="right", family="monospace")
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out_path)


def render_summary(replay, out_path) -> str:
    plt = _matplotlib()
    plies = replay["plies"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=_FRAME_COLORS["paper"])
    xs = np.arange(1, len(plies) + 1)
    evals = [p["eval_after"]["score_white"] for p in plies]
    nodes = [p["analysis"]["nodes"] for p in plies]
    axes[0].plot(xs, evals, color=_FRAME_COLORS["plate"], lw=2.4)
    axes[0].axhline(0, color=_FRAME_COLORS["rule"], lw=0.8)
    axes[0].set_title("Game evaluation / White perspective", color=_FRAME_COLORS["ink"], loc="left")
    axes[1].bar(xs, nodes, color=_FRAME_COLORS["plate"])
    axes[1].set_title("Search work / nodes by ply", color=_FRAME_COLORS["ink"], loc="left")
    axes[1].set_xlabel("Ply", color=_FRAME_COLORS["muted"])
    for ax in axes:
        ax.set_facecolor(_FRAME_COLORS["white"])
        ax.tick_params(colors=_FRAME_COLORS["muted"])
        for spine in ax.spines.values():
            spine.set_color(_FRAME_COLORS["rule"])
    fig.suptitle("KEPLER-64 / OBSERVATION SUMMARY", color=_FRAME_COLORS["plate"], weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out_path)


def _data_uri(path):
    suffix = Path(path).suffix.lower()
    mime = "image/gif" if suffix == ".gif" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


def _script_json(value):
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


_COMPONENT_CSS = """
*{box-sizing:border-box}html,body{overflow-x:clip}html{scroll-behavior:smooth}body{margin:0;background:var(--color-paper);color:var(--color-ink);font-family:var(--font-body)}
a{color:inherit}button,a,input{font:inherit}button,a.action{min-height:44px}.shell{width:min(112rem,100%);margin:auto;padding:0 var(--space-lg)}
.mast{display:flex;align-items:center;justify-content:space-between;gap:var(--space-md);padding:var(--space-md) 0;border-bottom:var(--rule-thin) solid var(--color-ink);font:600 var(--text-sm) var(--font-mono)}.mast nav{display:flex;align-items:center;gap:var(--space-lg)}.mast a{white-space:nowrap;text-decoration:none}.mast a:not(.action):hover{text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:4px}
.action{display:inline-flex;align-items:center;justify-content:center;background:var(--color-accent);color:var(--color-ink);border:var(--rule-thin) solid var(--color-ink);padding:.7rem 1.1rem;text-decoration:none;font:700 var(--text-sm) var(--font-mono);white-space:nowrap;box-shadow:3px 3px 0 var(--color-ink);transition:transform var(--dur-fast) var(--ease-out),box-shadow var(--dur-fast) var(--ease-out)}.action:hover{transform:translate(-1px,-1px);box-shadow:5px 5px 0 var(--color-ink)}.action:active{transform:translate(2px,2px);box-shadow:1px 1px 0 var(--color-ink)}
.opening{display:grid;grid-template-columns:minmax(0,.72fr) minmax(0,1.28fr);min-height:min(74vh,760px);border-bottom:var(--rule-thin) solid var(--color-ink)}.thesis{display:flex;flex-direction:column;justify-content:space-between;padding:clamp(2rem,5vw,4.5rem) var(--space-xl) var(--space-xl) 0;border-right:var(--rule-thin) solid var(--color-ink)}.thesis h1{font:700 var(--text-display)/.92 var(--font-display);letter-spacing:-.035em;margin:0;max-width:10ch;overflow-wrap:anywhere;min-width:0}.thesis p{max-width:38rem;color:var(--color-muted);font-size:var(--text-lg)}.premise{display:block;margin-bottom:var(--space-md);font:600 var(--text-sm) var(--font-mono);color:var(--color-plate)}.thesis-actions{display:flex;gap:var(--space-md);align-items:center;flex-wrap:wrap}.text-link{font:600 var(--text-sm) var(--font-mono);text-underline-offset:4px;white-space:nowrap}
.instrument{display:flex;flex-direction:column;justify-content:center;padding:var(--space-xl) 0 var(--space-xl) var(--space-xl);min-width:0}figure{margin:0}img{display:block;width:100%;height:auto}.instrument img{border:var(--rule-thin) solid var(--color-ink);background:var(--color-panel)}figcaption{display:flex;justify-content:space-between;gap:var(--space-md);padding-top:var(--space-sm);font:var(--text-sm) var(--font-mono);color:var(--color-muted)}
.observation{padding:var(--space-xl) 0 var(--space-section)}.stage{display:grid;grid-template-columns:minmax(0,1fr) 16rem;gap:var(--space-xl);align-items:start}.controls{position:sticky;top:var(--space-md);align-self:start;border-top:3px solid var(--color-ink);padding-top:var(--space-md)}.controls label,.data-label{display:block;font:600 var(--text-sm) var(--font-mono);margin-bottom:var(--space-sm)}input[type=range]{width:100%;accent-color:var(--color-accent)}.step-controls{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:var(--space-sm)}button{margin-top:var(--space-sm);padding:.7rem;border:var(--rule-thin) solid var(--color-ink);background:var(--color-panel);color:var(--color-ink);font:650 var(--text-sm) var(--font-mono);white-space:nowrap}button:hover:not(:disabled){background:var(--color-panel-raised)}button:active:not(:disabled){transform:translateY(1px)}button:disabled{color:var(--color-muted);border-color:var(--color-rule)}:focus-visible{outline:3px solid var(--color-focus);outline-offset:3px}
.readout{margin-top:var(--space-lg);border-top:var(--rule-thin) solid var(--color-rule)}.readout-head{display:flex;align-items:baseline;justify-content:space-between;gap:var(--space-md);padding:var(--space-md) 0;border-bottom:var(--rule-thin) solid var(--color-rule)}.move-mark{font:700 clamp(1.6rem,3.2vw,2.6rem)/1 var(--font-display)}.measure{font:var(--text-sm) var(--font-mono);color:var(--color-muted);text-align:right}
.facts{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:var(--space-md);padding:var(--space-md) 0;border-bottom:var(--rule-thin) solid var(--color-rule)}.facts div{min-width:0}.facts dt{font:600 var(--text-sm) var(--font-mono);color:var(--color-muted)}.facts dd{margin:var(--space-xs) 0 0;font:var(--text-sm) var(--font-mono);overflow-wrap:anywhere}.facts .wide{grid-column:1/-1}
.secondary-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:var(--space-xl);padding-top:var(--space-md)}.ranked,.terms{list-style:none;padding:0;margin:0}.ranked li{display:grid;grid-template-columns:2.5rem minmax(0,1fr) auto;gap:var(--space-sm);padding:.5rem 0;border-bottom:var(--rule-thin) solid var(--color-rule);font:var(--text-sm) var(--font-mono)}.ranked li[data-selected=true]{color:var(--color-accent);font-weight:700}.terms li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:var(--space-sm);padding:.4rem 0;border-bottom:var(--rule-thin) solid var(--color-rule);font:var(--text-sm) var(--font-mono)}.positive{color:var(--color-positive)}.negative{color:var(--color-negative)}details{margin-top:var(--space-md)}details summary{cursor:pointer;font:650 var(--text-sm) var(--font-mono);text-decoration:underline;text-underline-offset:4px}
.explain{display:grid;grid-template-columns:minmax(0,.55fr) minmax(0,1.45fr);gap:var(--space-xl);padding:var(--space-section) 0;background:var(--color-plate);color:var(--color-panel);box-shadow:0 0 0 100vmax var(--color-plate);clip-path:inset(0 -100vmax)}.explain h2,.evidence h2,.ledger h2,.close h2{font:700 clamp(2rem,4vw,3.5rem)/1 var(--font-display);margin:0;letter-spacing:-.025em}.explain-copy{max-width:70ch;font-size:var(--text-md)}.equation{font:600 clamp(1rem,2.2vw,1.55rem)/1.4 var(--font-mono);padding:var(--space-lg) 0;border-top:var(--rule-thin) solid currentColor;border-bottom:var(--rule-thin) solid currentColor}
.evidence,.ledger{padding:var(--space-section) 0;border-bottom:var(--rule-thin) solid var(--color-ink)}.section-head{display:grid;grid-template-columns:minmax(0,.65fr) minmax(0,1.35fr);gap:var(--space-xl);align-items:end;margin-bottom:var(--space-xl)}.section-head p{max-width:65ch;color:var(--color-muted)}.summary-image{border:var(--rule-thin) solid var(--color-ink);background:var(--color-panel)}
table{width:100%;border-collapse:collapse;font:var(--text-sm) var(--font-mono)}th,td{padding:.7rem .5rem;border-bottom:var(--rule-thin) solid var(--color-rule);text-align:right}th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}th{border-bottom-color:var(--color-ink)}
.portable{display:grid;grid-template-columns:minmax(0,.7fr) minmax(0,1.3fr);gap:var(--space-xl);align-items:start;margin-top:var(--space-xl);padding-top:var(--space-xl);border-top:var(--rule-thin) solid var(--color-rule)}details img{margin-top:var(--space-md);border:var(--rule-thin) solid var(--color-ink)}.method{color:var(--color-muted);max-width:70ch}.close{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:var(--space-xl);align-items:end;padding:var(--space-section) 0}.close h2{max-width:13ch}
@media(max-width:1100px){.facts{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media(max-width:900px){.opening,.stage,.explain,.section-head,.portable,.secondary-grid{grid-template-columns:minmax(0,1fr)}.opening{min-height:0}.thesis{border-right:0;border-bottom:var(--rule-thin) solid var(--color-ink);padding-right:0}.instrument{padding-left:0}.controls{position:static}.close{grid-template-columns:minmax(0,1fr)}}
@media(max-width:640px){.shell{padding:0 var(--space-md)}.mast nav a:not(.action){display:none}.thesis h1{font-size:var(--text-display)}.opening,.observation,.explain,.evidence,.ledger,.close{padding-top:var(--space-xl);padding-bottom:var(--space-xl)}.facts{grid-template-columns:repeat(2,minmax(0,1fr))}figcaption{display:block}table,thead,tbody,tr,th,td{display:block}thead{position:absolute;inline-size:1px;block-size:1px;overflow:hidden;clip-path:inset(50%)}tr{padding:var(--space-sm) 0;border-bottom:var(--rule-thin) solid var(--color-ink)}td{display:grid;grid-template-columns:7rem minmax(0,1fr);text-align:left!important;border:0;padding:.22rem 0}td::before{content:attr(data-label);font-weight:700;color:var(--color-muted)}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition-duration:.01ms!important}}
"""


_SCRIPT = """<script>
const frames=__FRAMES__;const data=__DATA__;let i=0;const $=id=>document.getElementById(id);const image=$('frame'),slider=$('ply'),caption=$('caption'),output=$('ply-output'),prev=$('prev'),next=$('next'),moveMark=$('move-mark'),measure=$('measure'),candidateList=$('candidate-list'),termList=$('term-list'),facts={move:$('fact-move'),number:$('fact-number'),mover:$('fact-mover'),rank:$('fact-rank'),mass:$('fact-mass'),depth:$('fact-depth'),nodes:$('fact-nodes'),elapsed:$('fact-elapsed'),fen:$('fact-fen')};function signed(v){return (v>=0?'+':'')+Number(v).toFixed(3);}function renderList(row){if(!row)return;moveMark.textContent=row.move;measure.textContent=row.mover+' / rank '+(row.rank??'—')+' / depth '+row.depth+' / '+Number(row.nodes).toLocaleString()+' nodes / '+Math.round(row.elapsed)+' ms';candidateList.replaceChildren(...row.candidates.map(c=>{const li=document.createElement('li');li.dataset.selected=String(c.selected);li.innerHTML='<span>'+String(c.rank).padStart(2,'0')+'</span><span>'+c.move+(c.selected?' / selected':'')+'</span><span>'+signed(c.score)+'</span>';return li;}));termList.replaceChildren(...row.terms.map(t=>{const li=document.createElement('li');li.className=t.value>=0?'positive':'negative';li.innerHTML='<span>'+t.label+'</span><span>'+signed(t.value)+'</span>';return li;}));facts.move.textContent=row.move;facts.number.textContent=row.move_number;facts.mover.textContent=row.mover;facts.rank.textContent=row.rank??'—';facts.mass.textContent=row.mass;facts.depth.textContent=row.depth;facts.nodes.textContent=Number(row.nodes).toLocaleString();facts.elapsed.textContent=Math.round(row.elapsed)+' ms';facts.fen.textContent=row.fen;}function show(n){if(!frames.length)return;i=Math.max(0,Math.min(frames.length-1,n));image.src=frames[i];image.alt='Ply '+(i+1)+': '+data[i].mover+' played '+data[i].move+'. Rank '+(data[i].rank??'unavailable')+', White evaluation '+signed(data[i].score)+'.';slider.value=i;slider.setAttribute('aria-valuetext','Ply '+(i+1)+', '+data[i].move);caption.textContent='Ply '+(i+1)+' of '+frames.length+' / '+data[i].move;output.textContent=(i+1)+' / '+frames.length;prev.disabled=i===0;next.disabled=i===frames.length-1;renderList(data[i]);}if(frames.length){show(0);slider.addEventListener('input',()=>show(Number(slider.value)));prev.addEventListener('click',()=>show(i-1));next.addEventListener('click',()=>show(i+1));document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1);});}
</script>"""


def render_html(replay, frames, gif_path, summary_path, out_path,
                repository_url: str = _REPOSITORY_URL) -> str:
    tokens = Path(__file__).with_name("replay_tokens.css").read_text(encoding="utf-8")
    css = tokens + _COMPONENT_CSS
    frame_uris = [_data_uri(path) for path in frames]
    summary_uri = _data_uri(summary_path) if Path(summary_path).exists() else ""
    gif_uri = _data_uri(gif_path) if Path(gif_path).exists() else ""
    rows = []
    for p in replay["plies"]:
        rows.append(
            f"<tr><td data-label='Ply'>{p['ply']}</td>"
            f"<td data-label='Move'>{html.escape(p['move_san'])}</td>"
            f"<td data-label='Rank'>{p['selected_candidate_rank'] or '—'}</td>"
            f"<td data-label='White eval'>{p['eval_after']['score_white']:+.3f}</td>"
            f"<td data-label='Mover shift'>{p['eval_shift_mover']:+.3f}</td>"
            f"<td data-label='Nodes'>{p['analysis']['nodes']:,}</td></tr>"
        )
    title = html.escape(replay["game"].get("Event", "Kepler-64 replay"))
    white = html.escape(replay["game"].get("White", "White"))
    black = html.escape(replay["game"].get("Black", "Black"))
    result = html.escape(replay["result"])
    first_frame = frame_uris[0] if frame_uris else ""
    repository_url = html.escape(repository_url, quote=True)
    replay_data = [{
        "ply": p["ply"],
        "move": p["move_san"],
        "move_number": p["move_number"],
        "mover": p["mover"],
        "rank": p["selected_candidate_rank"],
        "score": p["eval_after"]["score_white"],
        "shift": p["eval_shift_mover"],
        "mass": _material_balance(p["fen_after"]),
        "fen": p["fen_after"],
        "nodes": p["analysis"]["nodes"],
        "depth": p["analysis"]["depth"],
        "elapsed": p["analysis"]["elapsed_ms"],
        "candidates": [{"move": c["move_san"], "rank": c["rank"],
                        "score": c["search_score_mover"],
                        "selected": c["move_uci"] == p["move_uci"]}
                       for c in _display_candidates(p)],
        "terms": [{"label": _TERM_LABELS.get(key, key), "value": value}
                  for key, value in _mover_terms(p).items() if key != "total"],
    } for p in replay["plies"]]

    frame_html = (f'<img src="{first_frame}" alt="Fused Kepler-64 observation showing the '
                  f'chessboard, gravitational field, decision record, and evaluation trace">'
                  if first_frame
                  else '<div class="empty">No analyzed plies are available in this replay.</div>')
    img_html = (f'<img id="frame" src="{first_frame}" alt="Kepler-64 move observation">'
                if first_frame else '')
    caption_text = f'Ply 1 of {len(frame_uris)}' if frame_uris else 'No replay positions'
    slider_max = max(len(frame_uris) - 1, 0)
    slider_disabled = '' if frame_uris else 'disabled'
    slider_output = f'1 / {len(frame_uris)}' if frame_uris else '0 / 0'
    next_disabled = '' if len(frame_uris) >= 2 else 'disabled'
    summary_html = (f'<img class="summary-image" src="{summary_uri}" '
                    f'alt="Game evaluation by ply and search nodes by ply">'
                    if summary_uri else '')
    gif_html = (f'<details><summary>Play portable animation</summary>'
                f'<img src="{gif_uri}" alt="Animated replay of the analyzed chess game">'
                f'</details>'
                if gif_uri else '<p class="method">No animation was generated.</p>')

    body = f"""<div class="shell"><header class="mast"><a href="#top">KEPLER-64 / ROCHE ENGINE</a><nav aria-label="Primary"><a href="#mechanism">How it works</a><a href="#ledger">Replay</a><a class="action" href="{repository_url}">View source on GitHub</a></nav></header>
<main id="top"><section class="opening"><div class="thesis"><div><span class="premise">A differentiable astrophysical chess engine</span><h1>What if gravity could play chess?</h1><p>Kepler-64 replaces the usual evaluation function with an N-body gravitational system. Every move becomes a measurable change in force, binding, and tidal stress.</p></div><div class="thesis-actions"><a class="action" href="{repository_url}">View source on GitHub</a><a class="text-link" href="#observation">Inspect one move</a></div></div><figure class="instrument">{frame_html}<figcaption><span>{title}</span><span>{white} vs {black} / {result}</span></figcaption></figure></section>
<section class="observation" id="observation"><div class="stage"><div>{img_html}<figcaption id="caption" aria-live="polite">{caption_text}</figcaption></div><aside class="controls" aria-label="Replay controls"><label for="ply">Observation position</label><input id="ply" type="range" min="0" max="{slider_max}" value="0" {slider_disabled}><output id="ply-output" for="ply">{slider_output}</output><div class="step-controls"><button id="prev" type="button" disabled>Previous ply</button><button id="next" type="button" {next_disabled}>Next ply</button></div><p class="method">Candidate scores and contribution signs use the mover's perspective. The evaluation trace uses White's perspective. Scores are native Kepler values, not centipawns.</p></aside></div><div class="readout"><div class="readout-head"><strong class="move-mark" id="move-mark">—</strong><span class="measure" id="measure">—</span></div><dl class="facts"><div><dt>Move</dt><dd id="fact-move">—</dd></div><div><dt>Move number</dt><dd id="fact-number">—</dd></div><div><dt>Mover</dt><dd id="fact-mover">—</dd></div><div><dt>Rank</dt><dd id="fact-rank">—</dd></div><div><dt>Material</dt><dd id="fact-mass">—</dd></div><div><dt>Depth</dt><dd id="fact-depth">—</dd></div><div><dt>Nodes</dt><dd id="fact-nodes">—</dd></div><div><dt>Elapsed</dt><dd id="fact-elapsed">—</dd></div><div class="wide"><dt>FEN</dt><dd id="fact-fen">—</dd></div></dl></div><div class="secondary-grid"><div><span class="data-label">Candidate search</span><ol class="ranked" id="candidate-list"></ol></div><details class="terms"><summary>Move contribution breakdown</summary><ul class="terms" id="term-list"></ul></details></div></section>
<section class="explain" id="mechanism"><h2>The physics is the evaluator.</h2><div class="explain-copy"><p>Pieces carry mass. Their fields combine across the 64-square lattice. The kings are tested against tidal disruption while captures, binding, material, and parent-aware changes contribute to the score.</p><p class="equation">score = tidal stress + king force + binding + material + move deltas</p><p>This report separates immediate physical contributions from deeper minimax search. It shows what the engine computed without pretending the model is stronger or more certain than the recorded evidence.</p></div></section>
<section class="evidence"><div class="section-head"><h2>A complete game leaves a trace.</h2><p>The evaluation line and search-work profile expose where the position changed and how much computation each ply required.</p></div>{summary_html}<div class="portable"><div><h3>Portable replay</h3><p class="method">The GIF is kept behind an explicit disclosure so reduced-motion visitors are not forced to watch an autoplaying sequence.</p></div>{gif_html}</div></section>
<section class="ledger" id="ledger"><div class="section-head"><h2>Every move is inspectable.</h2><p>Rank, native evaluation, mover-relative shift, and search work remain available as plain data rather than being trapped inside the graphics.</p></div><table><thead><tr><th>Ply</th><th>Move</th><th>Rank</th><th>White eval</th><th>Mover shift</th><th>Nodes</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class="close"><h2>The source is the final proof.</h2><div><p class="method">Read the evaluator, training pipeline, search, tests, and replay generator in the repository.</p><a class="action" href="{repository_url}">Open Kepler-64 on GitHub</a></div></section></main></div>"""

    document = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                f"<title>{title}</title><style>{css}\n</style></head><body>"
                + body
                + _SCRIPT.replace("__FRAMES__", _script_json(frame_uris))
                          .replace("__DATA__", _script_json(replay_data))
                + "</body></html>")
    Path(out_path).write_text(document, encoding="utf-8")
    return str(out_path)


def generate_replay_assets(replay, constants, output_dir, duration: float = 0.8):
    """Render all replay assets. *duration* is seconds per frame; the Pillow
    GIF writer is fed milliseconds internally."""
    output = Path(output_dir)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(len(replay["plies"])):
        path = frames_dir / f"frame_{index:03d}.png"
        render_replay_frame(replay, index, constants, path)
        frames.append(str(path))
    summary = output / "summary.png"
    if replay["plies"]:
        render_summary(replay, summary)
    gif_path = output / "replay.gif"
    images = [imageio.imread(path) for path in frames]
    if images:
        imageio.mimsave(gif_path, images, duration=round(duration * 1000), loop=0)
    replay_json = output / "replay.json"
    replay_json.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    html_path = output / "index.html"
    render_html(replay, frames, gif_path, summary, html_path)
    return {"html": str(html_path), "gif": str(gif_path),
            "summary": str(summary), "json": str(replay_json), "frames": frames}
