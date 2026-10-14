"""Pre-launch honesty benchmarks (kepler64/bench/honest.py).

Each benchmark is A vs B with colours alternating over seeded random
openings, games played through GameState (accretion carried), and the report
separates games decided on the board from games adjudicated at the ply cap.

  physics-vs-material   trained physics  vs  the same search on material only
  learned-vs-frozen     trained physics  vs  the hand-set default universe
  custom                --a / --b constants files

Examples:

    python scripts/benchmarks.py physics-vs-material --dry-run
    python scripts/benchmarks.py physics-vs-material --games 200 --depth 2 --max-plies 300
    python scripts/benchmarks.py learned-vs-frozen --games 200 --no-cap
    python scripts/benchmarks.py custom --a trained.json --b other.json --games 40

`--dry-run` prints the configuration and plays 2 games capped at 6 plies, so
the harness can be checked in under a minute without spending compute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kepler64.bench.honest import material_only, run_match
from kepler64.core.constants import Constants, load_constants

TRAINED = "kepler64/training/trained_constants.json"


def _constants(path: str | None) -> Constants:
    if path is None:
        return Constants()
    loaded = load_constants(path)
    if loaded is None:
        raise SystemExit(f"constants file not found or unreadable: {path}")
    return loaded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["physics-vs-material", "learned-vs-frozen", "custom"])
    ap.add_argument("--a", default=TRAINED, help="constants for side A (default: trained)")
    ap.add_argument("--b", default=None, help="constants for side B (custom only)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--depth", type=int, default=2, help="fixed search depth (deterministic)")
    ap.add_argument("--move-ms", type=float, default=None,
                    help="time budget per move instead of --depth (machine-dependent)")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--no-cap", action="store_true", help="play every game to a rule-based end")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench-results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    a = _constants(args.a)
    if args.benchmark == "physics-vs-material":
        b, b_label = material_only(a), "material-only (same search, G=0, positional gains 0)"
    elif args.benchmark == "learned-vs-frozen":
        b, b_label = Constants(), "frozen default universe"
    else:
        b, b_label = _constants(args.b), args.b

    games, max_plies = args.games, (None if args.no_cap else args.max_plies)
    if args.dry_run:
        games, max_plies = 2, 6

    config = {
        "benchmark": args.benchmark, "a": args.a, "b": b_label, "games": games,
        "depth": None if args.move_ms else args.depth, "move_ms": args.move_ms,
        "max_plies": max_plies, "opening_plies": args.opening_plies,
        "seed": args.seed, "dry_run": args.dry_run,
    }
    print(json.dumps(config, indent=2), flush=True)

    t0 = time.time()

    def on_game(rec):
        adj = f" (adjudicated {rec['adjudicated']})" if rec["adjudicated"] else ""
        print(f"[game {rec['game'] + 1}/{games}] A={rec['a_color']:5s} "
              f"{rec['result']} by {rec['termination']}{adj} after {rec['plies']} plies "
              f"[{time.time() - t0:.0f}s]", flush=True)

    out = run_match(a, b, games=games, depth=None if args.move_ms else args.depth,
                    move_ms=args.move_ms, max_plies=max_plies,
                    opening_plies=args.opening_plies, seed=args.seed, on_game=on_game)
    summary = out["summary"]
    print(json.dumps(summary, indent=2))

    if not args.dry_run:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{args.benchmark}-{stamp}.json"
        path.write_text(json.dumps({"config": config, **out}, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
