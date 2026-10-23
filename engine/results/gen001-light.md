# Light Gen 1 attempt (2026-10-24): rejected

Run: `kepler-train tdleaf` from gen-000, depth 3, 500 games per round, 6 threads, lr 0.01, lambda 0.7, scale s = 4.0
(fixed), adjudication on, seed 20261024. It stopped at the 42-minute cap after 27 rounds (13,500 self-play games).
Per-round constants are in `gen001-light-trajectory.csv`, and the final candidate is in `gen001-light-candidate.json`.

| constant | Gen 0 | after 27 rounds |
|---|---|---|
| m_n | 3.0 | 1.380 |
| m_b | 3.0 | 1.747 |
| m_r | 5.0 | 2.461 |
| m_q | 9.0 | 5.531 |
| kappa | 1.0 | 1.546 |
| alpha | 0.35 | 0.281 |
| sigma | 0.5 | 0.558 |
| lambda_bind | 0.01 | 0.002 |
| lambda_space | 0.05 | 0.000 |
| rho | 1.5 | 2.430 |
| k_roche | 3.0 | 0.481 |
| beta | 2.0 | 0.000 |

Pinned at a bound: lambda_space, beta.

Acceptance (`sprt-gen001light-vs-gen000-d3.json`, depth 3, 6 threads): **H0 after 366 games, W122 D39 L205,
Elo -80 [-116, -46].** The candidate is weaker than Gen 0 and is rejected. Gen 0 remains the current generation.

Diagnosis: all four piece masses shrank together (each to about half), and the space and Roche weights went to zero.
TDLeaf fits predictions `tanh(eval / s)` to noisy self-play outcomes with `s` fixed. The cheapest way to make
predictions less confident is to shrink every term, and with the pawn fixed at 1 that also changes the piece ratios and
erases the physics terms. Fitting predictions is not the same as playing better.
