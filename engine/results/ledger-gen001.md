# Law Ledger at Gen 1 (2026-10-24)

Each row is an SPRT on GitHub Actions (`r-baruah/kepler-64-compute`, 20 shards x 100 opening pairs = 4,000 games, depth 4,
seed 1001; elo0 = 0, elo1 = 10, alpha = beta = 0.05, judged once on the total). Player A is gen-001 and player B is gen-001
with one law changed, so a positive Elo means **the law helps**. The Roche row is reversed: A is gen-001 with the law switched
on (beta = 2.0), B is gen-001, so a negative Elo means switching it on hurts. Raw results and run links are in `ledger-gen001/`.
The determinism gate (D8) passed before the campaign.

| law | change tested | W / D / L (A's view) | Elo [95%] | SPRT | verdict |
|---|---|---|---|---|---|
| all physics (bind, space, roche off) | removing it | 3313 / 334 / 353 | +330 [+316, +345] | H1 | physics kept |
| L2 spreading, sigma | sigma = 0 | 2500 / 422 / 1078 | +129 [+118, +140] | H1 | **kept** |
| L4b held space | lambda_space = 0 | 2365 / 510 / 1125 | +111 [+101, +122] | H1 | **kept** |
| L3 light speed c = 2 vs 3 | c = 3 | 1799 / 567 / 1634 | +14 [+4, +24] | H1 | c = 2 preferred over 3 |
| L3 light speed c = 2 vs 1 | c = 1 | 1706 / 619 / 1675 | +3 [-7, +13] | inconclusive | tie: c = 1 is as strong |
| L2 king emission, kappa | kappa = 0 | 1732 / 632 / 1636 | +8 [-1, +18] | inconclusive | inconclusive |
| L1 learned masses | masses 3/3/5/9 | 1789 / 488 / 1723 | +6 [-4, +16] | inconclusive | inconclusive (learned is about equal to human values) |
| L4a binding energy | lambda_bind = 0 | 1609 / 608 / 1783 | -15 [-25, -5] | H0 | **cut** (stronger without it) |
| L5 Roche limit (gen-001 has beta = 0) | beta = 2.0 switched ON | 1343 / 512 / 2145 | -71 [-81, -61] | H0 | **cut as written** (switching it on hurts) |
| L6 accretion | f_acc = 0 | 1139 / 568 / 2293 | -103 [-113, -93] | H0 | **cut** (much stronger without it) |

Reading it:
- **Kept, strongly:** the field as a whole (+330 over material only), spreading (+129) and held space (+111).
- **Light speed:** c = 2 beats c = 3 (+14); c = 1 ties c = 2. Pull that travels one or two moves is enough, and three
  moves of reach is too much.
- **Cut:** accretion (gen-001 is 103 Elo *weaker* than the same universe without it), binding energy (15 Elo weaker),
  and the Roche limit in its spec-1 form (switching it on costs 71 Elo).
- **Inconclusive:** king emission (kappa), and learned vs human piece masses.

Caveats:
- Engine vs engine at depth 4.
- One law changed at a time, with the other constants left as gen-001 learned them (no re-tuning).
- gen-001 was trained with f_acc = 0.8, which was never tuned. A universe trained without accretion could judge the
  other laws differently.
