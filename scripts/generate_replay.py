"""Generate a Kepler-64 replay: replay.json (+ HTML, GIF, PNG frames).

The game is replayed through the real engine and the trained constants, with
accretion and velocity carried across the game line. `--json-only` skips the
rendered assets and writes just the web-ready JSON (schema:
kepler64/analysis/REPLAY_SCHEMA.md).

    python scripts/generate_replay.py game.pgn --json-only --output out/
    python scripts/generate_replay.py --max-plies 60 --json-only   # self-play
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kepler64 import Constants, RocheEngine
from kepler64.core.constants import load_constants
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
    parser.add_argument("--json-only", action="store_true",
                        help="write replay.json only (no HTML/GIF/PNG rendering)")
    args = parser.parse_args()

    # load_constants ignores the "_meta" provenance block and fills leaves an
    # older artifact predates; Constants(**json) crashed on both.
    constants = load_constants(args.constants) or Constants()
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
    if args.json_only:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "replay.json"
        path.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
        print(json.dumps({"json": str(path), "plies": len(replay["plies"]),
                          "highlights": replay["highlights"]}, indent=2))
        return
    assets = generate_replay_assets(replay, constants, args.output, args.duration)
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
