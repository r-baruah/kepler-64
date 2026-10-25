# Gen 0 calibration (2026-10-24)

Rule (`docs/v1/02_LAWS_SPEC.md` §8): over the calibration positions, the standard deviation across sibling
moves of `T_bind + T_space + T_roche` must be at most 0.5 ku, and its largest deviation at most 2.5 ku
(positions where a king's eta exceeds rho excepted).

Positions: 300 seeded random-play positions (xorshift seed 12345, 6 to 65 plies, accretion carried), every legal
child scored. Spec-1 constants otherwise at their Gen 0 values.

| lambda_bind | lambda_space | beta | sibling std (mean) | largest deviation (mean) | worst | passes |
|---|---|---|---|---|---|---|
| **0.010** | **0.05** | **2.0** | **0.270** | **0.684** | **2.401** | **yes (chosen)** |
| 0.012 | 0.06 | 3.0 | 0.336 | 0.838 | 2.934 | no (worst) |
| 0.015 | 0.07 | 3.0 | 0.391 | 0.986 | 3.446 | no (worst) |
| 0.020 | 0.08 | 4.0 | 0.485 | 1.211 | 4.179 | no (worst) |

At the chosen values: mean |T_bind| = 0.400 ku, mean |T_space| = 0.493 ku, mean |T_roche| = 0.089 ku.
King index eta quantiles: p10 0.014, p50 0.129, p90 0.610, p99 1.522, max 3.436 (rho = 1.5, so a king
reaches its breaking point in about 1% of these positions).

Evaluation speed during the run: about 387,000 evaluations per second, single thread, release build,
Ryzen 5 5500U (includes move generation for the random play).
