# 04 · Training and gates (v1)

Status: agreed 2026-10-23.

Two rules govern this file:
- **The engine learns only from itself.** No human games, no Stockfish labels, no opening books. Stockfish is only the exam.
- **Every public number comes from a committed file in `engine/results/`** produced by the commands below.

## 1. Training: TDLeaf(λ) self-play

TDLeaf(λ) is the method used for KnightCap (Baxter, Tridgell & Weaver, 1998) to tune a small set of evaluation parameters by self-play. It fits v1 well: few constants, a smooth evaluation, exact gradients (`Dual<12>`, see `03_ENGINE_ARCHITECTURE.md` §3).

**One round** (`kepler-train tdleaf`):
1. **Self-play.** N games from the opening suite (§4), both colours, the current generation against itself.
   - Fixed search: depth 4 (milestone 3 start), raised to the deepest that fits the budget.
   - Small exploration: the first 8 plies play a random move among those within 0.15 ku of the best.
2. **Record the leaves.** For each ply `t`, store the principal-variation leaf position `l_t` (the position the search's score actually came from) with its carried mass state `A`.
3. **Temporal differences.** With `v_t = tanh(eval(l_t) / s)` from the mover's view (`s = 4 ku`, fixed), `d_t = v_{t+1}' − v_t`, where the prime means sign-adjusted to the same side. Game end replaces `v` with the result: +1 / 0 / −1.
4. **Update** for each A-tuned constant `θ`:
   ```
   Δθ = η · Σ_t ∂v_t/∂θ · Σ_{j≥t} λ^{j−t} d_j
   ```
   with `λ = 0.7`, an Adam optimiser on the averaged update, and projection onto the bounds of `02_LAWS_SPEC.md` §1.
5. **Report pinned constants.** Any constant pinned at a bound for more than 50% of the last 10 updates is logged as pinned.

**Outer-loop constants:**
- `f_acc` (accretion) is tuned by SPSA on match results: perturb by ±0.1, play a 200-game mini-match, step.
- `c` (light speed) is chosen by the ablation in §5.
- These are slow, so they run once per generation, not every round.

**Generation acceptance:** a candidate becomes **Gen k+1** only if it passes SPRT against Gen k (§2) at the match time control. It's written to `engine/data/generations/gen-{k+1:03}.json`:
```json
{ "generation": 1, "parent": 0, "spec": "v1.0-spec-1", "created": "2026-..",
  "constants": { "m_n": 3.1, "...": 0 },
  "pinned": ["..."],
  "training": { "rounds": 0, "games": 0, "depth": 4, "lambda": 0.7 },
  "acceptance": { "sprt": "results/sprt-gen001-vs-gen000.json" } }
```

**Compute budget (one laptop, 10 of 12 threads):**
- A self-play game at depth 4 should take ~1–2 s.
- A round of 2,000 games is about 5 minutes.
- A generation (20 rounds plus SPRT) fits overnight.
- `kepler-train` takes `--threads` (default 10) and `--max-minutes`, and checkpoints every round, so a run can stop and resume.

## 2. Statistics

- **SPRT** (generation acceptance and Law Ledger): `elo0 = 0`, `elo1 = 10`, `α = β = 0.05`, using the pentanomial model on game pairs (each opening played with both colours). Maximum 4,000 games. If the limit is hit, the result is "inconclusive".
- **Elo estimates** (ladder): logistic Elo from W/D/L, reported with a 95% interval. Never reported from fewer than 200 games.
- **Default time controls:** 10+0.1 for SPRT, 30+0.3 for the ladder. Fixed node or depth limits are allowed in milestone 1 only, and must be labelled.

## 3. Test suites (`engine/data/suites/`)

| suite | contents | pass bar |
|---|---|---|
| `perft.epd` | the six positions in `03_ENGINE_ARCHITECTURE.md` §9 | exact |
| `mate1.epd`, `mate2.epd` | built by `kepler-train suite build-mates` from the opening suite with Stockfish verifying, 100 each | 100% at depth 4 |
| `v0_closed_blunders.epd` | 12 positions where v0 lost ≥ 2 pawns (Stockfish 19 d18), each with `am` (v0's move) and `bm` | **v1 plays none of the `am` moves** (milestone 1 gate); v1's move loses < 1 pawn vs SF in ≥ 10/12 |
| `tactics.epd` | about 300 classic tactics positions (Win At Chess). Source and licence noted in the file header before use | tracked per generation; no pass bar, it's a progress number |

Each suite run writes `engine/results/suite-<suite>-gen<k>.json`.

## 4. Opening suite

`engine/data/openings/balanced-200.epd`: 200 positions after 4–8 plies, built by `kepler-train openings build`.
- Random legal moves from the start position.
- Kept only if Stockfish (depth 14) scores the position within ±0.5 pawns.
- Deduplicated.

The file header records how it was built. Every match plays each opening twice with colours swapped.

## 5. Law Ledger

For each law, play SPRT: **v1 (full) vs v1 with that law removed**, at the current generation.

| ablation | how "removed" is defined |
|---|---|
| L3 light speed | `c = 1` vs `c = 2` vs `c = 3` (pick the best; others reported) |
| L2 spreading | `sigma = 0` |
| L2 king emission | `kappa = 0` |
| L4a binding | `lambda_bind = 0` |
| L4b space | `lambda_space = 0` |
| L5 Roche | `beta = 0` |
| L6 accretion | `f_acc = 0` |
| L1 learned mass | masses reset to 3/3/5/9 (the human values) |
| all physics | material only (`T_bind = T_space = T_roche = 0`) |

**Outcome per law:**
- **kept:** removing it is significantly worse;
- **cut:** removing it is not worse (SPRT accepts H0). The law is removed from the next spec version;
- **inconclusive.**

Results go to `engine/results/ledger-gen<k>.md` with the raw SPRT JSON next to it. The Ledger is public. Cuts are reported as plainly as keeps.

The **material-only** row is the most important honesty check: **if the physics can't beat plain material counting with the same search, v1 has failed** and the docs must say so.

## 6. Stockfish ladder

`kepler-train ladder` plays each generation, full-length (no ply cap, no adjudication by mass), against:
- Stockfish with `UCI_LimitStrength=true` at `UCI_Elo` 1320, 1500, 1700, 1900, 2100 (stopping two rungs above the first rung scoring below 20%);
- below 1320, Stockfish `Skill Level` 0, 3 and 6. These are labelled as skill levels, never as Elo.

200 games per anchored rung, at 30+0.3. Results go to `engine/results/ladder-gen<k>.json` and a summary in `ladder.md`. Only anchored rungs may be turned into an Elo estimate, with its interval.

Adjudication allowed: Stockfish at depth 16 scoring beyond ±10 pawns for 8 consecutive plies (both engines are still playing, this only saves time), the 50-move rule, repetition, tablebase-free draw by insufficient material.

## 7. Milestone 1 go/no-go

All at fixed depth 3 for both engines (it's a laws comparison, not a speed comparison), 100 games (50 openings × 2 colours) against v0 (`KEPLER_V0_UCI`):

| check | pass bar |
|---|---|
| closed suite | no `am` move played |
| match vs v0 at equal depth | v1 scores ≥ 60% |
| match vs material-only (same Rust search, depth 3) | v1 scores ≥ 55% (informational in M1, a gate from M3) |
| invariants | `02_LAWS_SPEC.md` §11 all pass |

- **Go:** continue to milestone 2.
- **No-go:** revise the laws (`02_LAWS_SPEC.md`), log the change, rerun. The Rust scaffolding is reused either way.
