"""The Credibility Gate (audit Phase 3): does LEARNED physics beat frozen
physics? This is the single most important honesty artifact in the project.

Pipeline (fully self-contained, no external data):
  1. SELF-PLAY  - play N games with the default universe; harvest outcome +
     teacher-labeled policy examples (training/selfplay.py).
  2. TRAIN      - fit the trainable leaves (TRAINABLE_LEAVES) on those examples
     (Adam through the whole gravity kernel). fix_G=True: G is non-identifiable.
  3. VALIDATE   - held-out ranking metrics (top-1 / MRR) learned vs base.
  4. MATCH      - head-to-head, colors balanced, fixed time budget per move;
     ply-capped games adjudicated by terminal army-mass edge (the engine's
     own material reading, not a heuristic).
  5. REPORT     - docs/credibility_gate_results.md + JSON artifact.

Usage: python scripts/credibility_gate.py [--games 6] [--steps 300]
           [--match-games 8] [--move-ms 150] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from kepler64.core.constants import Constants, save_constants, TRAINABLE_LEAVES
from kepler64.training.selfplay import play_training_games, _terminal_mass_edge


def head_to_head(constants_a: Constants, constants_b: Constants, *,
                 games: int, move_ms: float, max_plies: int = 60,
                 seed: int = 0):
    """A vs B, colors balanced. Returns dict with wins/draws/losses for A."""
    from kepler64 import RocheEngine
    import numpy as np

    rng = np.random.default_rng(seed)
    eng_a = RocheEngine(constants=constants_a, load_trained=False)
    eng_b = RocheEngine(constants=constants_b, load_trained=False)

    w = d = l = 0
    for g in range(games):
        a_is_white = (g % 2 == 0)
        board = chess.Board()
        plies = 0
        while not board.is_game_over() and plies < max_plies:
            white_to_move = board.turn == chess.WHITE
            eng = eng_a if white_to_move == a_is_white else eng_b
            mv = eng.play(board, search_time_ms=move_ms, use_multiverse=False)
            if mv is None:
                break
            board.push(mv)
            plies += 1

        oc = board.outcome()
        if oc is not None and oc.winner is not None:
            a_won = (oc.winner == chess.WHITE) == a_is_white
            res = "win" if a_won else "loss"
        elif plies >= max_plies:
            edge = _terminal_mass_edge(board)          # +1 white ahead, -1 black
            if edge == 0.0:
                res = "draw"
            else:
                a_won = (edge > 0) == a_is_white
                res = "win" if a_won else "loss"
        else:
            res = "draw"
        w += res == "win"
        d += res == "draw"
        l += res == "loss"
        print(f"[match] game {g + 1}/{games}: A({'W' if a_is_white else 'B'}) "
              f"{res} after {plies} plies", flush=True)
    return {"wins": w, "draws": d, "losses": l, "games": games}


def elo_proxy(w: int, d: int, l: int) -> float:
    """Score-based Elo difference of A vs B (draws count half), clamped."""
    n = max(1, w + d + l)
    s = min(max((w + 0.5 * d) / n, 0.01), 0.99)
    return 400.0 * math.log10(s / (1.0 - s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=48)
    ap.add_argument("--move-ms", type=float, default=200.0)
    ap.add_argument("--teacher-ms", type=float, default=600.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--match-games", type=int, default=8)
    ap.add_argument("--match-move-ms", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    base = Constants()

    print("=== 1/4 self-play harvest ===", flush=True)
    examples, summary = play_training_games(
        base, games=args.games, max_plies=args.max_plies,
        move_ms=args.move_ms, teacher_ms=args.teacher_ms,
        seed=args.seed, verbose=True)
    print(f"harvest: {summary}", flush=True)
    if len(examples) < 40:
        print("Too few examples to train meaningfully; aborting.", flush=True)
        return 1

    print("=== 2/4 train learned universe ===", flush=True)
    from kepler64.training.train import train_examples
    trained, metrics = train_examples(
        base, examples, steps=args.steps, lr=args.lr, fix_G=True,
        seed=args.seed, verbose=True, return_metrics=True)
    print(f"validation metrics: {json.dumps(metrics, indent=2)}", flush=True)

    print("=== 3/4 head-to-head match ===", flush=True)
    match = head_to_head(trained, base, games=args.match_games,
                         move_ms=args.match_move_ms, seed=args.seed + 1)
    elo = elo_proxy(match["wins"], match["draws"], match["losses"])
    print(f"match: {match}  elo_proxy={elo:+.0f}", flush=True)

    print("=== 4/4 report ===", flush=True)
    leaf_dump = {n: float(getattr(trained, n)) for n in TRAINABLE_LEAVES}
    result = {
        "config": vars(args),
        "selfplay": summary,
        "metrics": metrics,
        "match_learned_vs_frozen": match,
        "elo_proxy": elo,
        "trained_leaves": leaf_dump,
        "wall_seconds": round(time.time() - t0, 1),
    }
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "credibility_gate_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    md = _markdown_report(base, leaf_dump, metrics["baseline"],
                          metrics["trained"], match, elo, summary, args)
    (out_dir / "credibility_gate_results.md").write_text(md, encoding="utf-8")
    save_constants(trained, Path("kepler64/training/trained_constants_gate.json"),
                   meta={"source": "credibility_gate"})
    print(f"done in {result['wall_seconds']}s -> docs/credibility_gate_results.md",
          flush=True)
    return 0


def _markdown_report(base, leaf_dump, m_b, m_t, match, elo, summary, args):
    verdict = ("the learned universe plays measurably stronger - the thesis "
               "holds on this run." if elo > 0 else
               "this run did NOT separate learning from noise - treat as an "
               "honest null result and scale data/steps before claiming "
               "anything.")
    md = f"""# Credibility Gate - Learned vs Frozen Physics

*Generated by `scripts/credibility_gate.py` on {time.strftime('%Y-%m-%d %H:%M')}.
Self-play: {summary['games']} games -> {summary['examples']} examples
({summary['wins']}W/{summary['losses']}L/{summary['draws']}D). Training:
{args.steps} Adam steps through the full gravity kernel, G fixed
(non-identifiable). Match: {args.match_games} games @ {args.match_move_ms:.0f} ms/move.*

| Metric | Frozen (base) | **Learned** | Delta |
|---|---|---|---|
| Policy top-1 | {m_b['top1']:.3f} | **{m_t['top1']:.3f}** | {m_t['top1'] - m_b['top1']:+.3f} |
| MRR (all) | {m_b['mrr']:.3f} | **{m_t['mrr']:.3f}** | {m_t['mrr'] - m_b['mrr']:+.3f} |
| MRR (captures) | {m_b['mrr_capture']:.3f} | **{m_t['mrr_capture']:.3f}** | {m_t['mrr_capture'] - m_b['mrr_capture']:+.3f} |
| MRR (quiet) | {m_b['mrr_quiet']:.3f} | **{m_t['mrr_quiet']:.3f}** | {m_t['mrr_quiet'] - m_b['mrr_quiet']:+.3f} |
| Head-to-head (W/D/L) | - | **{match['wins']}/{match['draws']}/{match['losses']}** | Elo ~ {elo:+.0f} |

**Verdict:** {verdict}

## Trained leaves after the run

| Leaf | Base | Trained |
|---|---|---|
"""
    for name, v in leaf_dump.items():
        md += f"| {name} | {getattr(base, name):.3f} | {v:.3f} |\n"
    return md


if __name__ == "__main__":
    sys.exit(main())
