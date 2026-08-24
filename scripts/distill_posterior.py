"""Posterior distillation (roadmap innovation #7): which constants does the
evaluation actually CONSTRAIN?

Method — one-at-a-time sensitivity: for each trainable leaf, perturb it
multiplicatively (×0.8 and ×1.25) around the base universe and measure how
much the score moves across a small panel of positions. A leaf whose
perturbation barely moves any score is (currently) unconstrained by the
physics — training will not learn it from outcomes. A leaf that swings scores
heavily is doing real work.

This is a cheap, local diagnostic (a few hundred evals, no search, no
training). It complements the credibility gate: the gate measures whether
learning helps play; this measures WHERE the physics has leverage.

Usage: python scripts/distill_posterior.py [--out docs/posterior_distillation.md]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from kepler64.core.constants import Constants, TRAINABLE_LEAVES
from kepler64.core.fastboard import FastBoard
from kepler64.core.evaluate import score_white


PANEL_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    "r4rk1/ppp2ppp/8/8/8/8/PPP2PPP/R4RK1 w - - 0 1",
]


def panel_pairs():
    """(child_masses, parent_masses) pairs — the delta terms are difference-
    form (child − parent), so a meaningful panel needs BOTH sides of each
    transition, exactly like the search sees them."""
    import numpy as np
    from kepler64.core.transitions import child_mass_vector

    out = []
    for fen in PANEL_FENS:
        b = FastBoard.from_chess(chess.Board(fen))
        moves = b.legal_moves()
        if not moves:
            continue
        mv = moves[0]
        parent_mv = b.mass_vector(float(Constants().c))
        child = b.apply(mv)
        child_mv = child_mass_vector(b, mv, parent_mv, child_board=child,
                                     c_lorentz=float(Constants().c))
        out.append((child_mv, parent_mv))
    return out


def distill(base, pairs, lo_factor=0.8, hi_factor=1.25):
    rows = []
    base_scores = [float(score_white(m, base, parent=p)) for m, p in pairs]
    for leaf in TRAINABLE_LEAVES:
        deltas = []
        for factor in (lo_factor, hi_factor):
            pert = replace(base, **{leaf: float(getattr(base, leaf)) * factor})
            for (m, p), s0 in zip(pairs, base_scores):
                deltas.append(abs(float(score_white(m, pert, parent=p)) - s0))
        rows.append((leaf, sum(deltas) / len(deltas)))
    rows.sort(key=lambda r: -r[1])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/posterior_distillation.md")
    args = ap.parse_args()

    base = Constants()
    pairs = panel_pairs()
    rows = distill(base, pairs)
    total = sum(s for _, s in rows) or 1.0

    lines = [
        "# Posterior Distillation — Which Constants Does the Physics Constrain?",
        "",
        "*One-at-a-time multiplicative sensitivity (×0.8 / ×1.25) of the Layer-1 "
        "score across a 6-position panel, base universe. Mean |Deltascore| per leaf, "
        "sorted. High sensitivity = the leaf does real work and training can "
        "learn it; near-zero = currently unconstrained by the physics.*",
        "",
        "| Rank | Leaf | Mean |dScore| | Share of total sensitivity |",
        "|---|---|---|---|",
    ]
    for rank, (leaf, s) in enumerate(rows, 1):
        lines.append(f"| {rank} | `{leaf}` | {s:.4f} | {100 * s / total:.1f}% |")
    dead = [leaf for leaf, s in rows if s < 1e-6]
    if dead:
        lines += ["", f"**Unconstrained at current defaults:** {', '.join(f'`{x}`' for x in dead)}",
                  "(expected for the zero-init gains `lambda_gw` / `lambda_sch` — they are",
                  "behavior-neutral until training or a hand raise switches them on.)"]
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"... -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
