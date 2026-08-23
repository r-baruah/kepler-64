# Design Note — True Causal Gravity (Field-Age Propagation)

**Status:** designed, not yet implemented. This is the highest-priority innovation from the 2026-08-23 audit (roadmap Part 4, innovation #1).

## The limitation today

`c` (the speed-of-light leaf) gates a *static* field: `sigmoid(c − d)` weights every piece's contribution by distance alone, every ply, forever. Physically this says "distant masses pull weakly," not "distant masses' *changes* haven't arrived yet." A queen sprinting to your kingside influences the enemy King on the very same ply it moves — regardless of `c`. The retarded-potential story is therefore a metaphor about magnitude, not about time.

## The upgrade

Make influence actually propagate. Maintain a per-square **field age**: when a mass moves, its old field contribution begins fading at its old square and a new one begins spreading from the new square, one Chebyshev ring per ply. Concretely:

- **State:** `age[sq] ∈ [0..∞]` per square — how many plies since that square's field was "refreshed" by local movement. `FastBoard` gains an `age` array updated in `apply_move`: squares within distance `min(c_int, k)` of both `from_sq` and `to_sq` reset to 0; all others increment.
- **Gate:** each source's reach gate becomes `sigmoid(c·(1 + age_growth)) − d_effective` where `d_effective = d − plies_since_move × c`. Equivalently: a mass moved t plies ago casts full influence only within radius `c·t`; beyond that frontier, the *pre-move* field still applies (kept as a slowly-decaying second field, or approximated by attenuating the mover's contribution by `sigmoid(c·t − d)`).
- **Evaluator change:** `force_field`/`potential_field` accept `(masses_now, masses_prev, ages)`; sources inside the light-cone use current mass, sources outside use their previous-tick mass. All terms downstream (η, force gauge, drift) automatically inherit genuine causality.

## Why this is the most on-vision change available

1. `c` becomes a **temporal** constant again: "my knight's influence arrives at f7 in ⌈d/c⌉ plies." Attack timing — the soul of chess — becomes a law of the universe instead of a search artifact.
2. Sibling moves finally separate *physically*: two quiet moves with equal static η differ in which enemy squares they've refreshed.
3. The Verlet drift term gains meaning: it projects along fields that are themselves evolving causally.

## Integration costs (be honest about these)

- **TT keys:** positions with different field-age states are physically different universes → the mass-checksum in `_hash` must be joined by an age checksum (quantize ages to int8 before hashing).
- **Search/perf:** two field evaluations per node (current + previous mass vectors). Mitigation: batch both in one vmap call; the 218-pad sweep already amortizes compile cost.
- **Training:** `training/data.py` must record age arrays per position; the loss vmaps over them unchanged.
- **Repetition rule interplay:** closed-orbit draws compare board identity only — decide whether identical boards with different ages are the "same universe" (recommended: yes for draw detection, no for TT).
- **Book/README claims:** update Ch.5 (Speed of Light) from "reach gate" to true retardation — the strongest public claim the project will have made.

## Acceptance criteria

1. Property test: a mass moved t plies ago has zero effect on squares with `d > c·t` and full effect within `c·t`.
2. Regression match vs current build (≥40 games): Elo delta reported either way.
3. Drift term stability: no NaN/divergence through a 100-game self-play harvest.
4. Book chapters 5, 7, 13 updated in the same PR.
