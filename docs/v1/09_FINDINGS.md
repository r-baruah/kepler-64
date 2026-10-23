# 09 · Findings log

A lab notebook: everything we learn from the engine, written down when we learn it. Newest at the bottom. One entry per finding.

Rules:
- Every entry names its **evidence** (a committed file in `engine/results/`, a script, or a commit).
- **Status** is one of:
  - `observed`: seen once, small sample;
  - `confirmed`: passed the relevant gate or reproduced;
  - `refuted`: later evidence says otherwise. Keep the entry; add a line saying what refuted it.
- **Caveats** are part of the finding, not an afterthought.
- Nothing here is rounded up. If a number changes, add a new entry instead of editing the old one.

---

## Open questions (what we're waiting to learn)

| id | question | answered by |
|---|---|---|
| Q1 | Does self-play rediscover roughly 3/3/5/9 piece values, starting from those values and from perturbed ones? | M3 training; a perturbed-start control run |
| Q2 | Does the universe value a bishop above a knight? | learned `m_b` vs `m_n` |
| Q3 | Does accretion survive (`f_acc` > 0), or is it rejected? | SPSA on `f_acc` + Law Ledger L6 row |
| Q4 | Which light speed wins: pull that travels 1, 2 or 3 moves? | Law Ledger L3 row |
| Q5 | Which laws survive the Ledger, and which get cut? | Law Ledger at Gen 1 |
| Q6 | How strong is it against calibrated opponents? | Stockfish ladder (M4) |
| Q7 | Where does it lose: pawn races, endgames, closed positions? | ladder game review, tactics suite |
| Q8 | Is the gauge (king torn apart) a good predictor of the game result several moves later? | replay analysis of ladder games |

---

## Entries

### F1 · v0's headline physics never changed a move (2026-10-23) · confirmed
- **What:** removing the tidal-tensor / Roche η term changed the chosen move in **0%** of 120 positions. The decisive terms were material and a straight-line "pull at the king" gauge.
- **Evidence:** `docs/v1/01_V0_AUDIT.md` §1, `scripts/audit/v0_terms.py`.
- **Caveat:** random-play positions; the gap is too large for real games to reverse it.

### F2 · Accretion let v0 think a queen for two pawns was a good trade (2026-10-23) · confirmed
- **What:** after `Qd3 c4 Qxc4 dxc4 Bxc4` (French Advance), v0's material read **+1.94 for White**, who had given up a queen for two pawns. The mass came back through the capture chain.
- **Evidence:** `01_V0_AUDIT.md` §3, `scripts/audit/v0_launder.py`.
- **Consequence:** v1 counts material at rest mass only (spec L1/L6). The regression test is in `engine/crates/kepler-core/tests/laws.rs`.

### F3 · v0 blundered in closed positions (2026-10-23) · confirmed
- **What:** 13 of 24 moves v0 played in French Advance / King's Indian structures lost ≥ 2 pawns by Stockfish 19 (depth 18), including one-move oversights.
- **Evidence:** `01_V0_AUDIT.md` §2, `engine/data/suites/v0_closed_blunders.epd`.

### F4 · v1 plays none of those blunders (2026-10-24) · confirmed
- **What:** on the 12 blunder positions, v1 Gen 0 played none of v0's moves at depth 3 or 6. At depth 3, all 12 of its moves lose less than one pawn by Stockfish's judgement.
- **Evidence:** `engine/results/suite-v0_closed_blunders-gen000-*.json`.
- **Caveat:** 12 positions chosen *because* v0 failed them. A tactics suite is the broader test.

### F5 · v1 Gen 0 beat v0 100–0 at equal depth (2026-10-24) · confirmed
- **What:** 100 games, both at depth 3: +100 =0 −0 (93 adjudicated by Stockfish at more than 10 pawns for 8 plies, 7 checkmates).
- **Evidence:** `engine/results/m1-v1gen000-vs-v0-d3.json`, `m1-gate.md`.
- **Caveat:** a laws-and-correctness comparison, not a rating. A 100% score has no meaningful Elo.

### F6 · A field shaped by the move rules beats plain material counting, untrained (2026-10-24) · confirmed
- **What:** the same Rust search with the v1 laws vs material only:
  - **70.9%** over 400 games at depth 3 (Elo +155, 95% interval +121 to +191);
  - **68.8%** over 200 games at equal time, 100 ms per move (Elo +137, +91 to +188).
- **Why it matters:** mobility, centralisation, attack/defence, hanging pieces and king safety are not written anywhere in the code. They are quantities of one field whose geometry is the movement rules.
- **Evidence:** `engine/results/m1-v1gen000-vs-material-d3.json`, `m1-v1gen000-vs-material-mt100.json`.
- **Caveat:** Gen 0 constants are hand-set (calibrated, not learned). Engine vs engine only.

### F7 · Speed: from seconds per move to half a million positions per second (2026-10-24) · confirmed
- **What:**
  - v0 (Python + JAX) took 2.6–3.9 s per move at depth 3.
  - v1 searches about **510,000 nodes/s** and evaluates about 387,000 positions/s on one thread.
  - Generating all legal moves to depth 5 from the start position (4,865,609 positions) takes 23 ms.
- **Evidence:** `01_V0_AUDIT.md` §5, `engine/results/calibration-gen000.md`, perft test output.
- **Caveat:** Ryzen 5 5500U laptop, single thread.

### F8 · A king reaches its breaking point in about 1% of random positions (2026-10-24) · observed
- **What:** with Gen 0 constants, the king index η exceeds the Roche limit ρ = 1.5 in about 1% of random-play positions (median η 0.13, 90th percentile 0.61).
- **Evidence:** `engine/results/calibration-gen000.md`.
- **Caveat:** random play is not real play. Measure again on ladder games.

### F9 · Early training pushes binding energy to zero (2026-10-24) · observed, then **confounded** (see F10–F11)
- **What:** in two short smoke runs (2 × 200 games and 2 × 100 games, depth 3), TDLeaf drove `lambda_bind` (binding energy, law L4a) to its lower bound, 0, with a consistently negative gradient.
- **Evidence:** smoke-run outputs (not committed; too small). The light Gen 1 run will confirm or refute this.
- **Caveat:** tiny samples. The Law Ledger is the real test of whether L4a earns its place.
- **Update (2026-10-24):** the same runs shrank every physics weight and every piece mass together (F10), so this is not evidence about binding energy. Only the Law Ledger can say a law should go.

### F10 · The first training attempt made the engine weaker, and the gate caught it (2026-10-24) · confirmed
- **What:**
  - 27 TDLeaf rounds (13,500 self-play games, depth 3) moved every piece mass down together: N/B/R/Q 3/3/5/9 went to 1.38/1.75/2.46/5.53. Space and Roche weights went to 0.
  - The candidate lost the acceptance test against Gen 0: **W122 D39 L205 over 366 games, Elo −80 [−116, −46]**. It was rejected; Gen 0 stays.
- **Evidence:** `engine/results/gen001-light.md`, `gen001-light-trajectory.csv`, `sprt-gen001light-vs-gen000-d3.json`.
- **Caveat:** **the learned masses are an artefact (F11), not piece values.** They must never be quoted as "what the universe thinks a knight is worth".

### F11 · Why that training failed: a scale mismatch plus a feedback loop (2026-10-24) · confirmed (diagnosis)
- **What:** measured on 200 Gen 0 self-play games (depth 3):
  - **Outcome scale:** the best-fitting `s` in tanh(eval/s) is about 5.5–6, but training fixed it at 4. That's constant pressure to shrink every term.
  - **Capped games:** games stopped at the move cap are scored as draws even when one side is clearly ahead (2 of 4 had a 2–3 pawn imbalance), which teaches that material doesn't predict results. As the evaluation softened, games got longer (mean 123 to 157 plies) and capped more often (2% to 8%), reinforcing the shrinking.
  - **Leaf fidelity:** the replayed PV leaf's static eval differs from the root search score by more than 0.5 ku in 11% of plies (mean 0.25 ku). Those are mid-exchange positions, a further source of noise.
- **Evidence:** `engine/crates/kepler-core/examples/tdleaf_diag.rs` on `selfplay --seed 99 --games 200 --depth 3`.
- **Consequence:** fixes before any rerun: a fitted or learned scale, capped games excluded from TD targets, and a quiet (quiescence) leaf.
