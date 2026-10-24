# 10 · Universe-inspired methods

Kepler-64 is a tiny universe. When a standard machine-learning recipe misbehaves here (it often does, see `09_FINDINGS.md` F10–F11), we look at how the real universe handles the same problem, and copy the mechanism literally.

Rules:
- Each idea names its **physical source**.
- Each idea gets a **literal mechanism** in Kepler: an exact quantity, not a metaphor.
- Each idea states **what it should fix**.
- Each idea has a **test**: an SPRT or a Law Ledger row.

Absurd ideas are welcome if they pass the same bar. Nothing here is active until its status says so.

Status: `idea` → `coded` (behind a flag, default off) → `tested` (with a result file) → `adopted` or `rejected`.

---

## Part A · How the universe learns (training methods)

### T1 · Temperature: the universe has a temperature
- **Source:** statistical mechanics. The probability of an outcome given an energy follows a Boltzmann-like curve with a temperature.
- **Mechanism:** TDLeaf predicts `v = tanh(eval / T)`. `T` is the **temperature of the universe**, a training-only quantity (not a law, not one of the 14 constants). It's learned alongside the constants, so overconfidence is paid by heating the universe, not by shrinking every mass.
- **Fixes:** F11's scale mismatch (fixed `s = 4` vs a best fit of about 6), which drove the collapse in F10.
- **Test:** TDLeaf round with T learned vs the rejected run; SPRT of the candidate against Gen 0.
- **Status:** coded (`--temperature <T0>`, `--learn-temperature` (default on), `--no-learn-temperature`; `tdleaf`, `tdleaf-shard`, `tdleaf-reduce`). `T` is stored in the checkpoint and in `candidate.json` under `training.temperature`.

### T2 · Conservation of mass during learning
- **Source:** conservation laws (Noether). A closed universe doesn't create or destroy its total mass.
- **Mechanism:** optionally, after every training step, rescale the four learned masses so the total rest mass of a full army (8 + 2·m_n + 2·m_b + 2·m_r + m_q) stays constant. Ratios can change freely; the total cannot drift.
- **Fixes:** uniform shrinking or inflation of all masses, and the scale degeneracy between masses and temperature.
- **Test:** with vs without, same seeds, SPRT.
- **Status:** coded (`--conserve-mass`, default off; `tdleaf` and `tdleaf-reduce`).

### T3 · Cosmological natural selection: universes compete
- **Source:** Smolin's cosmological natural selection (1992). Universes whose constants make them reproduce more come to dominate.
- **Mechanism:** around the current constants θ, spawn twin universes `θ + δ` and `θ − δ` with a random perturbation δ, and let them play each other. The universe that wins more pulls θ toward itself: `θ ← θ + a · (score₊ − score₋) · δ / |δ|²`. Mathematically this is SPSA, the method used to tune Stockfish's parameters. It optimises **playing strength directly**, not prediction, so F10's failure mode (fitting outcomes ≠ playing better) can't occur. It's naturally a work unit: one twin pair, one mini-match.
- **Also tunes** `f_acc` and the other non-differentiable or outer-loop constants.
- **Test:** an evolved candidate vs Gen 0, SPRT.
- **Status:** coded (`evolve-shard`, `evolve-reduce`, and the local loop `evolve`; `evolve-shard --iter K --seed S --shard-index k --pairs N`, `evolve-reduce --a0 --A --c0`).

### T4 · A cooling universe (annealing)
- **Source:** the early universe cools as it expands. Annealing in metallurgy, formalised as simulated annealing (Kirkpatrick 1983, from Metropolis 1953).
- **Mechanism:** in T3, both the perturbation size `c_k = c₀ / (k+1)^0.101` and the step `a_k = a₀ / (k+1+A)^0.602` cool with the iteration k. These are the standard SPSA gain schedules, read as a cooling curve. Self-play exploration noise cools the same way across rounds.
- **Fixes:** big early moves that never settle, or a search that freezes too early.
- **Status:** coded (`c_K` in `evolve-shard`, `a_K` in `evolve-reduce`, both from `--c0`, `--a0`, `--A`).

### T5 · Measure at equilibrium (quiet leaves)
- **Source:** you measure a system's energy when it's in equilibrium, not in the middle of a collision.
- **Mechanism:** TDLeaf records the position at the end of the quiescence line (all captures resolved), not the end of the main-line PV.
- **Fixes:** F11's 11% of mid-exchange training positions.
- **Status:** coded (`Searcher::quiet_leaf`, always used by `tdleaf` and `tdleaf-shard`; no flag).

### T6 · Unfinished histories don't count as outcomes
- **Source:** a measurement needs a finished process. A system cut off mid-evolution has no outcome yet.
- **Mechanism:** games stopped at the ply cap are **excluded** from TD targets (or, optionally, use the final search score as the target). They are never scored as draws.
- **Fixes:** F11's feedback loop (capped "draws" with material imbalances teaching that material doesn't matter).
- **Status:** coded (`--capped-target drop|score|draw`, default `drop`).

### T7 · Multiverse averaging (kept from v0, outside the core)
- **Source:** ensembles and posterior averaging.
- **Mechanism:** evaluate a move across several universes sampled near θ and play the move that's good in most of them. A play mode and an uncertainty band, not part of the default evaluation.
- **Status:** idea (later; a website universe-picker feature first).

---

## Part B · Candidate laws for spec-2

None of these is active. Each needs a spec change, then must earn its place in the Law Ledger.

### L7 · A pawn's light cone contains its promotion
- **Source:** retarded potentials and causality. What can happen within your light cone already influences you.
- **Mechanism:** let `h_prom` be the number of moves a pawn needs to reach its last rank along empty push squares, and `c_prom` a learned horizon. The pawn's emission and body mass interpolate toward its promoted form:
  `M_pawn = 1 + (m_q − 1) · alpha^(h_prom − 1)` if `h_prom ≤ c_prom`, otherwise 1.
  A pawn about to queen already pulls like part of a queen.
- **Fixes:** the predicted weakness with passed pawns and promotion races.
- **Ledger row:** `c_prom = 0` (off).
- **Status:** idea.

### L8 · Running couplings: constants change with the universe's energy
- **Source:** renormalisation. In quantum field theory, coupling constants "run": their value depends on the energy scale.
- **Mechanism:** the universe's energy scale `E` = the total rest mass on the board (high in the opening, low in the endgame). Chosen constants get two learned values, and the engine interpolates `θ(E) = θ_low + (θ_high − θ_low) · E / E_start`. Candidates are `kappa` (the king becomes a heavy hitter in the endgame), the piece masses, and `alpha`.
- **Fixes:** the opening and endgame needing different behaviour, e.g. king activity late in the game.
- **Ledger row:** high = low (no running).
- **Status:** idea.

### L9 · Hawking evaporation of accreted mass
- **Source:** black holes slowly radiate their mass away (Hawking 1974).
- **Mechanism:** absorbed mass decays every ply, `A ← A · (1 − ε_evap)`, with `ε_evap` learned. A capture's extra weight is real but temporary, which may give accretion a reason to survive in the Ledger.
- **Ledger row:** `ε_evap = 0`.
- **Status:** idea (absurd but testable).

### L10 · Dark matter: controlled space has weight
- **Source:** most of the universe's gravitating mass is unseen.
- **Mechanism:** each square a side holds (`Φ_own > Φ_enemy`) carries a small learned dark mass `μ_dark` that emits pull one hop, like a very light piece. Held territory itself pulls.
- **Fixes:** possibly long-term space and pressure in closed positions.
- **Ledger row:** `μ_dark = 0`.
- **Status:** idea (absurd but testable; cost: one extra field pass).

### L11 · Tidal load: the enemy's field landing on the king's body (Roche redesign)
- **Source:** the energy of an extended body in an external field is the sum of the field over the body. A body is torn apart (the Roche limit) by the external field acting across it, not by the ratio of totals spec-1 uses.
- **Mechanism:** for king `K` of colour `C` with body `S_K` (unchanged from L5):
  `U_K = Σ_{s∈S_K} Φ_{¬C}(s)`, and `T_tide = lambda_tide · (U_{K_B} − U_{K_W})`, with `lambda_tide` learned in [0, 2].
  Linear, with no threshold and no shape constants, so its gradient is never stuck at weight 0 (F30). The spec-1 L5 term stays as it is; the two can be tested separately or together.
- **Evidence for trying it:** F31. The enemy pull on the king's body carries information the rest of the evaluation lacks; the spec-1 denominator and gate waste it. An outcome fit puts the weight near 0.34 ku per unit of pull.
- **Fixes:** the Roche limit not earning its place (F14, F19, F27), so that "a king is torn apart" can be backed by a working law.
- **Honesty check (§9):** it uses only the field `Φ` and the L5 body, both already in the spec. It adds no new chess concept. It is the L5 slot's king danger, measured more directly.
- **Ledger row:** `lambda_tide = 0` (off).
- **Status:** idea, measured (F31). Coding waits until the attribution runs finish (they rebuild the engine every round); CI runs wait for the maintainer's go.

### Note: pawn structure may already be in the laws
Pawns only pull diagonally forward, so pawns defend each other through L4a binding. An isolated pawn gets no friendly pull and is "unbound". Doubled pawns block each other's field. If binding is kept (the F10 run zeroed it through an artefact, not evidence), pawn structure should partly emerge. Check this in ladder game reviews before adding anything.

---

## Part C · Laws proposed by anyone (planned, after v1.0)

Anyone can propose a law for the universe. The process is the same one every law here goes through, done in public:

1. **Proposal:** a GitHub issue using the "Propose a law" template. It covers the physical source; the literal mechanism as a quantity computed from the board (no chess heuristics: `02_LAWS_SPEC.md` §9 applies); what it should change; and how to switch it off.
2. **Review:** the maintainer checks that it's literal, computable and not a disguised chess rule, and tags it `law: accepted for testing` or explains why not.
3. **Implementation:** behind a constant whose value 0 turns it off, with invariant tests. One pull request per law.
4. **Verdict:** a Law Ledger row on the compute repository (SPRT, 4,000 games, depth 4) against the current generation. The issue gets the result, kept or cut, with the numbers and the run link, whatever the outcome.
5. **Record:** a findings entry. Kept laws enter the next spec version with credit to the proposer. Cut laws stay listed in a public "laws the universe rejected" table.

Every verdict is reproducible (`08_DISTRIBUTED_COMPUTE.md` §3a), so anyone can rerun it.
