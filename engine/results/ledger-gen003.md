# Law Ledger at Gen 3 (2026-10-24), two rows

Same setup as `ledger-gen001.md`: each row is an SPRT on GitHub Actions (`r-baruah/kepler-64-compute`, 20 shards x 100
opening pairs = 4,000 games, depth 4, seed 41; elo0 = 0, elo1 = 10, alpha = beta = 0.05, judged once on the total).
Player A is gen-003 and player B is gen-003 with one law switched off, so a positive Elo means **the law helps**.
These two laws were tested because training regrew them: both were at 0 in gen-002, and gen-003 has
`beta` = 1.49 and `lambda_bind` = 0.014. Raw results: `ledger-gen003-roche0.json`, `ledger-gen003-bind0.json`.

| law | change tested | W / D / L (A's view) | Elo [95%] | SPRT | verdict |
|---|---|---|---|---|---|
| L4a binding energy | lambda_bind = 0 | 1951 / 495 / 1554 | +35 [+25, +45] | H1 | **kept**: it now helps |
| L5 Roche limit (spec-1) | beta = 0 | 1662 / 576 / 1762 | -9 [-19, +1] | H0 | does not help; not significant either way |

Gen 1 comparison: at Gen 1 removing binding gained 15 Elo and switching Roche on (beta = 2.0) lost 71
(`ledger-gen001.md`). Binding reversed its verdict once the other constants moved; the Roche limit did not.
