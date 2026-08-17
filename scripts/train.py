"""Kepler-64 self-play training loop (CPU laptop class).

The universe learns from itself: the engine plays games with its own physics,
and a DEEPER time-budget integration of the SAME gravity kernel labels every
sampled position. Gradient descent through the physics adjusts the 14
trainable physical constants so the shallow eval reads like the deep eval.

No opening books, no hardcoded chess values, no invented heuristics — the only
supervisory signals are real self-play outcomes and the kernel's own deeper
integrations. (External Lichess/PGN data is still supported for anyone who has
it via --source data.)

Pipeline per round
------------------
  1.  SELF-PLAY HARVEST — the champion (last persisted constants, or the
      pristine universe on round 1) plays `--games` games on a move time
      budget. Every `--sample-every` plies a position is sampled:
        * all legal children as mass vectors (the policy input)
        * the teacher verdict — the same constants, BIGGER time budget —
          becomes the expert move to imitate
        * the final game result as the outcome label
  2.  TRAIN — Adam + gradient clipping + physical-bound projection through the
      full eval (outcome logistic + policy cross-entropy [+ margin]).
  3.  GATE — accept the new constants only if the validation split's mean
      reciprocal rank on the deeper teacher's labels does NOT regress
      (mrr_trained >= mrr_baseline). Accepted constants become the champion
      and are persisted to trained_constants.json; rejected candidates are
      discarded.
  4.  MATCH (optional, --match-every) — head-to-head games between the
      candidate and the champion. This is the Elo axis the loop climbs and is
      the final gate when enabled (candidate Elo may not drop below
      --min-elo vs the champion).

Usage
-----
  # one quick round (sanity):
  python scripts/train.py --rounds 1 --games 4 --steps 40

  # a real training session:
  python scripts/train.py --rounds 6 --games 8 --steps 120 \
      --match-every 2 --match-games 8 --seed 7

The champion persists at kepler64/training/trained_constants.json and the
engine loads it automatically (RocheEngine default).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# JAX CPU tuning for this laptop: all logical threads as XLA devices.
_n_logical = os.cpu_count() or 4
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_n_logical}"
os.environ.setdefault("OMP_NUM_THREADS", str(_n_logical))

# Make the repo root importable whether run as a script or from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kepler64.core.constants import (
    Constants, TRAINED_CONSTANTS_PATH, load_constants, save_constants,
)
from kepler64.training import train as trainmod
from kepler64.training import selfplay
from kepler64.bench import selfmatch


def round_train(champion: Constants, args, round_no: int, seed: int):
    """One self-play -> train -> gate cycle. Returns (accepted, champion_next, report)."""
    print(f"\n==== round {round_no} " + "=" * 50)
    t0 = time.time()
    print(f"[round] harvesting self-play data: {args.games} games, "
          f"move {args.move_ms}ms, teacher {args.teacher_ms}ms ...")
    examples, summary = selfplay.play_training_games(
        champion,
        games=args.games,
        max_plies=args.max_plies,
        move_ms=args.move_ms,
        teacher_ms=args.teacher_ms,
        sample_every=args.sample_every,
        max_children=args.max_children,
        seed=seed,
        verbose=args.verbose,
    )
    if len(examples) < 12:
        print(f"[round] only {len(examples)} examples harvested - skipping round")
        return False, champion, {"examples": len(examples), "skip": True}
    print(f"[round] {len(examples)} examples from {summary['games']} games "
          f"(+{summary['wins']} -{summary['losses']} ={summary['draws']}) "
          f"in {time.time() - t0:.0f}s")

    # ── train + held-out metrics ─────────────────────────────────────────
    trained, metrics = trainmod.train_examples(
        champion, examples, steps=args.steps, lr=args.lr, fix_G=args.fix_G,
        batch_size=args.batch_size, seed=seed, tau=args.tau,
        margin=args.margin, verbose=True, return_metrics=True,
    )
    b, t = metrics["baseline"], metrics["trained"]
    gate_mrr = t["mrr"] >= b["mrr"]

    report = {
        "round": round_no,
        "examples": len(examples),
        "games": summary,
        "mrr_base": b["mrr"], "mrr_trained": t["mrr"],
        "top1_base": b["top1"], "top1_trained": t["top1"],
        "mrr_gate": bool(gate_mrr),
        "elapsed": time.time() - t0,
    }
    print(f"[round] mrr gate: base={b['mrr']:.3f} -> trained={t['mrr']:.3f} "
          f"({'pass' if gate_mrr else 'fail'})")

    # ── optional head-to-head Elo gate ───────────────────────────────────
    if args.match_every > 0 and round_no % args.match_every == 0:
        print(f"[round] match: candidate vs champion, {args.match_games} games "
              f"@ {args.match_ms}ms ...")
        m = selfmatch.play_match(trained, champion, games=args.match_games,
                                 move_ms=args.match_ms, seed=seed)
        report["match"] = m
        gate_elo = m["elo"] >= args.min_elo
        report["elo_gate"] = bool(gate_elo)
        print(f"[round] match +{m['wins']} -{m['losses']} ={m['draws']}  "
              f"score={m['score']:.3f}  Elo {m['elo']:+.0f} "
              f"({'pass' if gate_elo else 'fail'})")
        if not gate_elo:
            return False, champion, report

    accepted = bool(gate_mrr)
    return accepted, (trained if accepted else champion), report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=1,
                    help="self-play training rounds")
    ap.add_argument("--games", type=int, default=8, help="games per round harvest")
    ap.add_argument("--max-plies", type=int, default=48)
    ap.add_argument("--move-ms", type=float, default=250.0,
                    help="explorer time budget (ms/move)")
    ap.add_argument("--teacher-ms", type=float, default=750.0,
                    help="teacher time budget (ms/move) — the label source")
    ap.add_argument("--sample-every", type=int, default=3,
                    help="sample every N plies")
    ap.add_argument("--max-children", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80, help="Adam steps per round")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--fix-G", dest="fix_G", action="store_true", default=True,
                    help="freeze G (non-identifiable in the tidal index)")
    ap.add_argument("--no-fix-G", dest="fix_G", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-every", type=int, default=0,
                    help="head-to-head Elo gate every N rounds (0 = off)")
    ap.add_argument("--match-games", type=int, default=8)
    ap.add_argument("--match-ms", type=float, default=120.0)
    ap.add_argument("--min-elo", type=float, default=-50.0,
                    help="candidate Elo vs champion below which the round is rejected")
    ap.add_argument("--source", choices=["selfplay", "data"], default="selfplay",
                    help="selfplay (default) or external Lichess/PGN data")
    ap.add_argument("--data", default=None,
                    help="dir with puzzles_50k.csv and games/Ripu01.pgn (--source data)")
    ap.add_argument("--out", default=str(TRAINED_CONSTANTS_PATH))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    champion = load_constants()
    provenance = "loaded persisted champion" if champion else "pristine universe"
    champion = champion or Constants()
    print(f"[train] start: {provenance}")
    print(f"[train] constants: {champion}")

    if args.source == "data":
        if not args.data:
            ap.error("--source data requires --data <dir>")
        from kepler64.training.data import puzzle_examples, game_examples
        puz = puzzle_examples(os.path.join(args.data, "puzzles_50k.csv"))
        games = game_examples(os.path.join(args.data, "games", "Ripu01.pgn"),
                              limit_games=30, limit_positions=8000)
        examples = puz + games
        print(f"[train] external data: {len(examples)} examples")
        print(f"[train] outcome-only phase ...")
        champion = trainmod.train_examples(champion, examples, steps=args.steps,
                                           lr=args.lr, fix_G=args.fix_G,
                                           verbose=True, policy=False)
        print(f"[train] policy phase ...")
        champion = trainmod.train_examples(champion, examples, steps=args.steps,
                                           lr=args.lr, fix_G=args.fix_G,
                                           verbose=True, policy=True)
        save_constants(champion, args.out, meta={"source": args.data,
                                                 "rounds": 1,
                                                 "mode": "external-data"})
        print(f"[train] saved constants -> {args.out}")
        return

    accepted_rounds = 0
    for rnd in range(1, args.rounds + 1):
        seed = args.seed + rnd
        accepted, champion, report = round_train(champion, args, rnd, seed)
        if accepted:
            accepted_rounds += 1
            save_constants(champion, args.out, meta={
                "round": rnd, "seed": seed,
                "mrr_base": report["mrr_base"],
                "mrr_trained": report["mrr_trained"],
                "match": report.get("match"),
                "source": "selfplay",
            })
            print(f"[round] ACCEPTED — champion persisted -> {args.out}")
        else:
            print(f"[round] rejected — champion unchanged")

    print(f"\n[train] done: {accepted_rounds}/{args.rounds} rounds accepted")
    print(f"[train] final constants: {champion}")


if __name__ == "__main__":
    main()
