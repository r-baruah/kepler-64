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
| Q1 | Does self-play rediscover roughly 3/3/5/9 piece values, starting from those values and from perturbed ones? | **F20: yes in one equal-mass run (order and ratios); replication pending** |
| Q2 | Does the universe value a bishop above a knight? | learned `m_b` vs `m_n` |
| Q3 | Does accretion survive (`f_acc` > 0), or is it rejected? | **answered by F17: rejected at Gen 1** |
| Q4 | Which light speed wins: pull that travels 1, 2 or 3 moves? | **answered by F18: 1 or 2, not 3** |
| Q5 | Which laws survive the Ledger, and which get cut? | **answered by F16–F19 (Gen 1 ledger)** |
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

### F12 · Results are bit-identical across machines (2026-10-24) · confirmed
- **What:** the three determinism reference units (a TDLeaf shard, an SPRT shard and an evolve shard) produced the same `result_hash` on the maintainer's laptop (AMD Ryzen 5 5500U, Fedora) and on a GitHub Actions runner (AMD EPYC 7763, Ubuntu): `b1f56a3e30f1eb9c`, `d7c71a0b84c5142b`, `da3f1b02cbf711c9`. 1-thread and 3-thread runs also match.
- **Why it matters:** work units from any machine can be merged without changing a single bit. A mismatch always means a bug or a bad worker, never noise.
- **Evidence:** `engine/data/determinism/reference.json`; workflow run 40077939725 in `r-baruah/kepler-64-compute`; rules D1–D8 in `08_DISTRIBUTED_COMPUTE.md` §3a.

### F13 · Learning by playing itself made the engine stronger: Gen 1 (2026-10-24) · confirmed
- **What:** 11 TDLeaf rounds (5,500 self-play games, depth 3, 35 minutes on the laptop) with the universe-inspired fixes (learned temperature, quiet leaves, capped games dropped) produced Gen 1. Against Gen 0 over 400 fixed games: **+181 Elo [+146, +219] at depth 3 and +132 [+98, +168] at depth 4.**
- **Contrast with F10:** the same method with a fixed temperature collapsed and lost 80 Elo. The learned temperature started at 6 and cooled to 3.7: the universe grew more confident as it learned.
- **Evidence:** `engine/results/gen001.md`, `engine/data/generations/gen-001.json`.
- **Caveat:** engine vs engine at fixed depth. Not a rating. The ladder (M4) measures strength against calibrated opponents.

### F14 · Training switches the Roche limit off (2026-10-24) · observed, twice
- **What:** in both TDLeaf runs (fixed and learned temperature), `beta`, the weight of a breaking king (law L5), went steadily to 0. **Gen 1 plays without the Roche limit.** Its king safety comes only from binding energy and held space.
- **Why it matters:** "a king is torn apart when the enemy's pull outweighs its defenders" is central to the public story. If the law contributes nothing, the story must not claim it drives play.
- **Possible causes, untested:**
  - the gauge rarely fires (about 1% of positions at Gen 0), so it carries little signal;
  - its sharp threshold gives little gradient;
  - it duplicates what binding and space already capture;
  - or king danger is simply better handled by search.
- **Next:** the Law Ledger row L5 (SPRT with `beta = 0` vs `beta > 0`), and a look at whether a smoother or differently normalised Roche index survives.

### F15 · The bishop pulls ahead of the knight (2026-10-24) · observed, not a discovery
- **What:** in the Gen 1 run, the bishop's mass rose steadily, 3.00 → 3.63, faster than the knight's (3.00 → 3.15). The queen rose 9.00 → 10.58 and the rook 5.00 → 5.29. This matches the common human view that a bishop is worth a little more than a knight.
- **Caveat:** the run started from human values (3/3/5/9) and the masses were still moving when it stopped. Only the equal-mass control run (Q1) can show whether the universe *discovers* this ordering.

### F16 · The Law Ledger: the universe keeps its field, and two laws do most of the work (2026-10-24) · confirmed
- **What:** 10 SPRTs of 4,000 games each (depth 4), each comparing gen-001 with one law changed:
  - The field as a whole is worth **+330 Elo** over plain material counting.
  - **Spreading** (pull spread thin over the squares it reaches, like conserved flux; sigma) is worth **+129**.
  - **Held space** is worth **+111**.
- **Evidence:** `engine/results/ledger-gen001.md`, raw files in `engine/results/ledger-gen001/`.
- **Caveat:** one law at a time, no re-tuning, engine vs engine.

### F17 · The universe rejected accretion (2026-10-24) · confirmed
- **What:** switching accretion off (f_acc = 0) makes gen-001 **103 Elo stronger** [93, 113]. Captures making pieces heavier hurts play under real chess rules.
- **Evidence:** `ledger-gen001/ledger-gen001-acc0.json`.
- **Caveat:** f_acc was never tuned (it stayed at the hand-set 0.8), and gen-001's other constants were learned with accretion on. A smaller fraction might still help; an SPSA run on f_acc would show that.

### F18 · Pull needs only one or two moves of travel (2026-10-24) · confirmed
- **What:** light speed c = 2 beats c = 3 by **14 Elo** [4, 24], and c = 1 ties c = 2 (+3 [−7, +13]). Influence that takes three moves to arrive adds noise rather than information.
- **Evidence:** `ledger-gen001-c3.json`, `ledger-gen001-c1.json`.
- **Note:** c = 1 is cheaper to compute, so at equal *time* it may be the better choice. Not tested yet.

### F19 · Binding energy and the spec-1 Roche limit both hurt (2026-10-24) · confirmed
- **What:**
  - Removing binding energy makes gen-001 **15 Elo stronger** [5, 25].
  - Switching the Roche limit back on (beta = 2) makes it **71 Elo weaker** [61, 81].
  - Together with F14 (training drove beta to 0 twice), the Roche limit *as formulated in spec-1* does not help.
- **Evidence:** `ledger-gen001-bind0.json`, `ledger-gen001-roche-on.json`.
- **Open:** whether a different Roche formula (smoother, or normalised another way) earns its place. The idea isn't refuted; this formula is.

### F20 · Starting from equal masses, self-play recovered the human piece order (2026-10-24) · confirmed (two seeds, see F24)
- **What:**
  - The knight, bishop, rook and queen all started at 4.43, so the army's total equals the human-valued army's (39), with the other laws at Gen 0.
  - After 30 TDLeaf rounds (60,000 self-play games, depth 4, GitHub Actions) they settled at **N 2.65, B 3.30, R 4.44, Q 8.23**.
  - Ratios to the knight: **bishop 1.25, rook 1.68, queen 3.11**. The human convention is 1.00, 1.67 and 3.00.
  - The order queen > rook > bishop > knight emerged with no human values and no human games. The temperature cooled from 6.0 to 2.42, the same pattern as F13.
- **Evidence:** `engine/results/control-equal-masses-a/` (round files and candidate), run `control-equal-masses-a` in `r-baruah/kepler-64-compute`.
- **Caveats:**
  - One seed.
  - Not fully converged: the queen was still rising and the knight still falling.
  - The absolute scale is below human (knight 2.65 vs 3).
  - The masses are what this universe's laws need, not "true" piece values.
- **Confirmed:** a second seed reached the same values (F24); the strength check is F23.

### F21 · First calibrated strength: Gen 2 against Stockfish 19 at fixed strength levels (2026-10-24) · confirmed, with caveats
- **What:** 200 games per level, 30 s + 0.3 s, Stockfish 19 with `UCI_LimitStrength`. Kepler's score:

  | Stockfish UCI_Elo | Kepler W / D / L | score |
  |---|---|---|
  | 1320 | 194 / 1 / 5 | 97% |
  | 1500 | 162 / 0 / 38 | 81% |
  | 1700 | 105 / 5 / 90 | 54% |
  | 1900 | 106 / 3 / 91 | 54% |
  | 2100 | 75 / 4 / 121 | 39% |

  Gen 2 is roughly level with Stockfish limited to 1700–1900 and clearly below 2100, at this time control.
- **Evidence:** `engine/results/ladder-gen002.json` and per-shard games in `engine/results/ladder-gen002/`, from the `ladder` workflow in `r-baruah/kepler-64-compute`.
- **Caveats:**
  - **29 of the 1,000 games were lost to a Kepler crash** (a bug in reporting the search line, since fixed; F22). All were scored as Kepler losses, so the true scores are slightly higher.
  - Stockfish's UCI_Elo is calibrated on its own scale at longer time controls. It is **not** a human FIDE rating. The equal scores at 1700 and 1900 suggest the limited-strength levels are not evenly spaced at 30 s + 0.3 s.
  - Claims must name the setup ("scores 54% against Stockfish 19 limited to UCI_Elo 1900 at 30+0.3"), never "rated 1900".

### F22 · Bug found by the ladder: illegal moves in the reported search line (2026-10-24) · fixed
- **What:** non-PV search nodes didn't clear their stored line, so a stale tail from an earlier sibling could be appended to the principal variation. The UCI layer replayed it to print `info pv` and panicked.
- **Fix:** every node clears its line on entry, and the UCI layer stops at the first illegal move. All 29 crash positions now replay cleanly, with no illegal move in any reported line. Search results are unchanged (golden tests), and the regression test is `kepler-core/tests/pv_legal.rs`.

### F23 · The self-discovered universe plays at least as well (2026-10-24) · confirmed, not a fair comparison
- **What:** the equal-mass control candidate (F20: piece values learned from an all-equal start) beat gen-001 (trained from human values) by **+19 Elo [+10, +29]** over 4,000 games at depth 4.
- **Evidence:** `engine/results/control-a-vs-gen001.json`.
- **Caveat:** the control had about 10× more training (30 rounds × 2,000 games at depth 4) than gen-001 (11 rounds × 500 games at depth 3), and both still use accretion. This shows that starting from equal values costs nothing. It does **not** show that self-discovered values beat human values.

### F24 · A second universe with a different seed found the same piece values (2026-10-24) · confirmed
- **What:**
  - `control-equal-masses-b` repeated F20 exactly (all four masses starting at 4.43, 30 rounds × 2,000 games, depth 4) with a different self-play seed. Its games differ from round 1 on.
  - It settled at **N 2.63, B 3.28, R 4.43, Q 8.23**; seed a settled at N 2.65, B 3.30, R 4.44, Q 8.23.
  - Ratios to the knight: **bishop 1.25, rook 1.68, queen 3.12** (seed a: 1.25, 1.68, 3.11).
  - Both switched the Roche gain `beta` off (at its bound of 0) and drove binding `lambda_bind` close to zero, matching F14 and F19.
- **Evidence:** `engine/results/control-equal-masses-b/` (round files and candidate), compared with `engine/results/control-equal-masses-a/`.
- **Meaning:** the values are a property of the laws plus self-play, not luck of one run.
- **Caveats:**
  - Same laws, same starting point, same training settings. Only the seed differs.
  - The absolute scale (knight about 2.6) is this universe's, not human.


### F25 · Gen 3: more self-play from gen-002 gained another 53 Elo (2026-10-24) · confirmed
- **What:** TDLeaf from gen-002 (run `gen003-tdleaf-a`: 20 rounds × 2,000 games, depth 4, learned temperature, quiet leaves, capped games dropped). The candidate beat gen-002 by **+53 Elo [+43, +63]** over 4,000 games at depth 4 (SPRT H1). Accepted as `gen-003`.
- **Evidence:** `engine/results/gen003a-vs-gen002.json`, `engine/results/gen003-tdleaf-a/`, `engine/data/generations/gen-003.json`.
- **What moved:**
  - Queen 10.58 → 8.28, rook 5.29 → 4.83, knight 3.15 → 3.04, bishop 3.63 → 3.53. The queen now sits where both equal-start universes put it (8.23, F24).
  - `kappa` (the king's own pull) 1.29 → 1.88, `alpha` 0.30 → 0.46, space 0.043 → 0.118.
  - The Roche gain `beta` came back from 0 to 1.49, and binding from 0 to 0.014. In F14 training drove `beta` to 0; starting from gen-002 it grew it again.
- **Caveats:** engine-vs-engine at fixed depth. Whether the regrown Roche and binding terms help is being tested (Ledger rows `ledger-gen003-roche0`, `ledger-gen003-bind0`).

### F26 · Binding energy came back and now earns its place (2026-10-24) · confirmed
- **What:** at Gen 1 the Ledger cut binding energy (L4a): removing it gained 15 Elo (F19), so gen-002 had `lambda_bind` = 0. Self-play from gen-002 grew it back to 0.014, and at Gen 3 removing it **costs 35 Elo [+25, +45]** (4,000 games, depth 4, SPRT H1).
- **Evidence:** `engine/results/ledger-gen003.md`, `engine/results/ledger-gen003-bind0.json`.
- **Meaning:** a law's value depends on the rest of the universe. The other constants moved a lot between Gen 1 and Gen 3 (F25); why binding became useful is not yet known. A Ledger verdict holds for the generation it was measured on, not forever.
- **Caveat:** one generation, one weight. The spec still lists L4a as a law, so no spec change is needed; F19's "cut" was a weight of 0, not a removal.

### F27 · The Roche limit still does not earn its place, even after training regrew it (2026-10-24) · confirmed
- **What:** gen-003 has a Roche gain of 1.49 (it was 0 in gen-002). Switching it off changes strength by **-9 Elo [-19, +1]** (SPRT H0): no evidence that it helps, and the estimate leans the other way.
- **Evidence:** `engine/results/ledger-gen003.md`, `engine/results/ledger-gen003-roche0.json`.
- **Meaning:** training moving a constant away from 0 is not evidence that the law helps. The spec-1 Roche limit stays unproven (F14, F19). Public copy must not say Kepler's king safety comes from tidal disruption.
- **Next:** the Roche redesign question (open decision with the maintainer).

### F28 · A tactics suite from Kepler's own games barely separates generations (2026-10-24) · observed
- **What:** `engine/data/suites/tactics-own.epd`, 200 positions from the gen-002 ladder games where Stockfish 19 (depth 16, MultiPV 2) finds one move at least 2 pawns better than any other. Best-move hits:

  | depth | gen-000 | gen-001 | gen-002 | gen-003 |
  |---|---|---|---|---|
  | 4 | 154 | 149 | 157 | 160 |
  | 6 | 172 | 171 | 172 | 174 |

- **Evidence:** `engine/results/suite-tactics-own-gen00<k>-d<depth>.json`, `engine/results/suites-per-generation.md`.
- **Meaning:** the spread (about 6 positions) is about one standard error. Short tactics are found by search, so the learned constants change little here; the generations' gains (F13, F25) come from judgement in quiet positions, which this suite doesn't test. Mate-in-1 and mate-in-2 are 100/100 for every generation.
- **Bias:** the positions come from games gen-002 played against strength-limited Stockfish (UCI_Elo 1320 to 2100), so they are drawn from those games' mistakes, not from chess at large.
