"""Kepler-64 training runner (CPU-only: laptop / Codespaces / Colab).

Two phases — run them in order:

  Phase 1  outcome-only   : teaches the physics eval to judge winning/losing
                            positions. CHEAP (no child-move scoring). Proves
                            the kernel learns at all, before the expensive part.
  Phase 2  policy-finetune : ranks the expert move above all legal children
                            through the gravity kernel. EXPENSIVE (64 children
                            per position) — this is what needs 16 GB RAM.

Usage
-----
  # Phase 1 only (fast, runs on the laptop):
  python scripts/train.py --phase 1 --data /path/to/data --steps 200

  # Phase 2 (needs more RAM — run on Codespaces):
  python scripts/train.py --phase 2 --data /path/to/data --steps 80

  # Both, sequentially:
  python scripts/train.py --phase both --data /path/to/data

`--data` must contain puzzles_50k.csv and games/Ripu01.pgn (see README).
Trained constants are written to kepler64/training/trained_constants.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make the repo root importable whether run as a script or from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kepler64.core.constants import Constants
from kepler64.training import data, train


def _load(puzzle_csv, games_pgn, puzzle_limit, game_limit, game_positions):
    puz = data.puzzle_examples(puzzle_csv, limit=puzzle_limit)
    games = data.game_examples(games_pgn, limit_games=game_limit,
                               limit_positions=game_positions)
    return puz, games


def run_phase1(base, data_dir, steps, lr, fix_G, verbose):
    """Outcome-only. No child vectors -> ~64x cheaper than phase 2."""
    puzzle_csv = os.path.join(data_dir, "puzzles_50k.csv")
    games_pgn = os.path.join(data_dir, "games", "Ripu01.pgn")
    print("[phase1] loading data (outcome only) ...")
    puz, games = _load(puzzle_csv, games_pgn, 5000, 30, 8000)
    examples = puz + games
    print(f"[phase1] {len(examples)} examples")
    # Strip child vectors so the loss runs outcome-only (cheap).
    for e in examples:
        e["child_m"] = e["child_m"][:1]
        e["mask"] = e["mask"][:1]
    return train.train_examples(base, examples, steps=steps, lr=lr, fix_G=fix_G,
                                verbose=verbose)


def run_phase2(base, data_dir, steps, lr, fix_G, verbose):
    """Policy fine-tune with full child vectors (needs RAM)."""
    puzzle_csv = os.path.join(data_dir, "puzzles_50k.csv")
    games_pgn = os.path.join(data_dir, "games", "Ripu01.pgn")
    print("[phase2] loading data (full child vectors) ...")
    puz, games = _load(puzzle_csv, games_pgn, 5000, 30, 8000)
    examples = puz + games
    print(f"[phase2] {len(examples)} examples")
    return train.train_examples(base, examples, steps=steps, lr=lr, fix_G=fix_G,
                                verbose=verbose)


def main():
    ap = argparse.ArgumentParser(description="Kepler-64 training (CPU)")
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument("--data", required=True,
                    help="dir with puzzles_50k.csv and games/Ripu01.pgn")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fix-G", dest="fix_G", action="store_true", default=True)
    ap.add_argument("--no-fix-G", dest="fix_G", action="store_false")
    ap.add_argument("--out", default="kepler64/training/trained_constants.json")
    args = ap.parse_args()

    base = Constants()
    t0 = time.time()

    if args.phase in ("1", "both"):
        base = run_phase1(base, args.data, args.steps, args.lr, args.fix_G, True)
        print(f"[phase1] done in {time.time()-t0:.1f}s")

    if args.phase in ("2", "both"):
        base = run_phase2(base, args.data, args.steps, args.lr, args.fix_G, True)
        print(f"[phase2] done in {time.time()-t0:.1f}s")

    with open(args.out, "w") as fh:
        json.dump({k: getattr(base, k) for k in
                   ["G","eps","c","roche","bonus","kgain","gamma","Rg",
                    "mref","mat_gain"]}, fh, indent=2)
    print(f"[train] saved constants -> {args.out}")


if __name__ == "__main__":
    main()
