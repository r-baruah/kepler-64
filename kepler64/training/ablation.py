"""Ablation — the credibility gate.

The whole thesis rests on gravity doing REAL work. This compares training with G
learned vs G fixed at 1 and reports the delta. If the delta is tiny, the gravity
is cosmetic and the project must be recalibrated. (Self-play win-rate + mean
|score| are cheap stand-ins for Elo here.)
"""

from ..core.constants import Constants
from ..core.evaluate import score_white
from .train import train


def run_ablation(samples, base: Constants = None, steps: int = 150):
    base = base or Constants()
    learned = train(base, samples, steps=steps, fix_G=False)
    fixed = train(base, samples, steps=steps, fix_G=True)

    def mean_abs_score(c):
        return sum(abs(float(score_white(m, c))) for m, _ in samples) / max(len(samples), 1)

    table = {
        "G_learned": learned.G,
        "G_fixed": fixed.G,
        "eps_learned": learned.eps,
        "c_learned": learned.c,
        "roche_learned": learned.roche,
        "mean_abs_score_learned": mean_abs_score(learned),
        "mean_abs_score_fixed": mean_abs_score(fixed),
    }
    return table


def print_table(table: dict):
    print("| metric | value |")
    print("|---|---|")
    for k, v in table.items():
        print(f"| {k} | {v:.4f} |")
