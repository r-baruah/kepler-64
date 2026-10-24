"""Roche-law diagnostics from probe.cjs output (findings F31).

Usage: python3 analyse.py <rows.json> <ladder-games-dir> <generation.json>
Prints markdown tables: separation of mated kings (AUC), separation beyond the rest
of the evaluation (score-matched strata), and a held-out outcome fit.
"""
import glob, json, os, sys
import numpy as np


def auc(pos, neg):
    """Mann-Whitney AUC: P(value for a positive > value for a negative), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2
        i = j
    n1, n0 = len(pos), len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    rows = json.load(open(sys.argv[1]))
    c = json.load(open(sys.argv[3]))["constants"]
    beta = c["beta"]
    T = json.load(open(sys.argv[3]))["training"]["temperature"]
    keys = ["eta", "stress", "diff", "tidal", "lostFrac", "onKing"]
    by = {}
    for r in rows:
        by.setdefault((r["file"], r["gi"], r["ply"]), {})[r["color"]] = r
    mated = {(r["file"], r["gi"]): r["color"] for r in rows if r["doomed"]}
    # own-view score without the spec-1 Roche term
    for kk in by.values():
        if len(kk) == 2:
            for a, o in (("white", "black"), ("black", "white")):
                kk[a]["noroche"] = kk[a]["score"] - beta * (kk[o]["gauge"] - kk[a]["gauge"])

    print("## 1. Separation: kings of the side that gets mated, by plies to mate (AUC)\n")
    print("| plies to mate | subset | n | " + " | ".join(keys + ["material", "score"]) + " |")
    print("|---" * (len(keys) + 5) + "|")
    neg_all = [r for r in rows if mated.get((r["file"], r["gi"])) != r["color"]]
    rng = np.random.default_rng(1)
    for lo, hi in [(1, 4), (5, 8), (9, 14), (15, 24), (25, 40)]:
        for lab, flt in [("all", lambda r: True), ("material within 1.5", lambda r: abs(r["mat"]) <= 1.5)]:
            pos = [r for r in rows if mated.get((r["file"], r["gi"])) == r["color"] and lo <= r["toEnd"] <= hi and flt(r)]
            neg = [r for r in neg_all if flt(r)]
            neg = [neg[i] for i in rng.choice(len(neg), min(20000, len(neg)), replace=False)]
            vals = [auc([r[k] for r in pos], [r[k] for r in neg]) for k in keys]
            vals += [auc([-r[k] for r in pos], [-r[k] for r in neg]) for k in ("mat", "score")]
            print(f"| {lo}-{hi} | {lab} | {len(pos)} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    print("\n## 2. Beyond the rest of the evaluation (20 strata matched on the score without Roche; 0.5 = adds nothing)\n")
    print("| plies to the end | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    d = [dict(r, y=1 if mated.get((r["file"], r["gi"])) == r["color"] else 0) for r in rows if "noroche" in r and r["toEnd"] >= 5]
    for lab, flt in [("5+", lambda r: True), ("5-14", lambda r: r["toEnd"] <= 14), ("15-40", lambda r: 15 <= r["toEnd"] <= 40)]:
        s0 = sorted([r for r in d if flt(r)], key=lambda r: r["noroche"])
        n, B, tw = len(s0), 20, 0
        acc = dict.fromkeys(keys, 0.0)
        for b in range(B):
            s = s0[b * n // B:(b + 1) * n // B]
            pos = [r for r in s if r["y"]]
            neg = [r for r in s if not r["y"]]
            if len(pos) < 30 or len(neg) < 30:
                continue
            w = min(len(pos), len(neg))
            tw += w
            for k in keys:
                acc[k] += w * auc([r[k] for r in pos], [r[k] for r in neg])
        print(f"| {lab} | " + " | ".join(f"{acc[k] / tw:.3f}" for k in keys) + " |")

    print("\n## 3. Held-out outcome fit (logistic on score/T; one third of games held out; decisive games only)\n")
    res = {}
    for f in glob.glob(os.path.join(sys.argv[2], "*.json")):
        for gi, g in enumerate(json.load(open(f))["games"]):
            res[(os.path.basename(f), gi)] = {"1-0": 1.0, "0-1": 0.0}.get(g["result"])
    X, y, G = [], [], []
    for k, kk in by.items():
        if len(kk) < 2 or res.get(k[:2]) is None:
            continue
        w, b = kk["white"], kk["black"]
        X.append([w["noroche"] / T, beta * (b["gauge"] - w["gauge"]) / T, (b["tidal"] - w["tidal"]) / T, (b["stress"] - w["stress"]) / T])
        y.append(res[k[:2]])
        G.append(k[:2])
    X, y = np.array(X), np.array(y)
    games = sorted(set(G))
    held = {games[i] for i in np.random.default_rng(1).permutation(len(games))[: len(games) // 3]}
    te = np.array([g in held for g in G])
    print("| features | held-out log loss | weights (per unit / T) |")
    print("|---|---|---|")
    for name, cols in [("score without Roche", [0]), ("+ spec-1 Roche term", [0, 1]), ("+ tidal spread", [0, 2]), ("+ stress", [0, 3]), ("+ tidal spread + stress", [0, 2, 3])]:
        A, wv, m = X[:, cols], np.zeros(len(cols)), ~te
        for _ in range(3000):
            p = 1 / (1 + np.exp(-A[m] @ wv))
            wv -= 0.5 * A[m].T @ (p - y[m]) / m.sum()
        p = np.clip(1 / (1 + np.exp(-A[te] @ wv)), 1e-6, 1 - 1e-6)
        ll = -np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
        print(f"| {name} | {ll:.4f} | {', '.join(f'{v:.3f}' for v in wv)} |")
    print(f"\n{len(y)} positions from {len(set(G))} decisive games.")


main()
