"""Generate a self-contained Kepler-64 stakeholder replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kepler64 import Constants, RocheEngine
from kepler64.analysis import record_game, record_selfplay
from kepler64.viz.replay import generate_replay_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", nargs="?", help="PGN file; omit for self-play")
    parser.add_argument("--game", type=int, default=0, help="zero-based PGN game index")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=40)
    parser.add_argument("--duration", type=float, default=0.8)
    parser.add_argument("--output", default="replay-output")
    parser.add_argument("--constants", default="kepler64/training/trained_constants.json")
    args = parser.parse_args()

    constants = Constants()
    constants_path = Path(args.constants)
    if constants_path.exists():
        constants = Constants(**json.loads(constants_path.read_text(encoding="utf-8")))
    engine = RocheEngine(constants)
    if args.pgn:
        with Path(args.pgn).open(encoding="utf-8", errors="replace") as fh:
            game = None
            for _ in range(args.game + 1):
                game = chess.pgn.read_game(fh)
                if game is None:
                    raise SystemExit("PGN game index is out of range")
        replay = record_game(game, engine, args.depth, args.candidates, args.max_plies)
    else:
        replay = record_selfplay(engine, args.depth, args.max_plies, args.candidates)
    assets = generate_replay_assets(replay, constants, args.output, args.duration)
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
