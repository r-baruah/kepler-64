# Milestone 1 go/no-go (2026-10-24): GO

v1 Gen 0 (`data/generations/gen-000.json`, spec v1.0-spec-1, hand-set and calibrated, no training) against the gates in
`docs/v1/04_TRAINING_AND_GATES.md` §7. Openings: `data/openings/balanced-200.epd`, each played with both colours.

| check | pass bar | result | file |
|---|---|---|---|
| closed suite (12 positions where v0 lost 2+ pawns) | no avoid-move played | **0/12 avoid-moves at depth 3 and depth 6** | `suite-v0_closed_blunders-gen000-d3.json`, `-d6.json` |
| same suite, move quality (Stockfish 19, depth 18) | n/a | depth 3: 12/12 moves lose under 1 pawn; depth 6: 11/12 | `suite-v0_closed_blunders-gen000-sfcheck.json` |
| vs v0, both at depth 3, 100 games | v1 scores at least 60% | **+100 =0 -0** (93 Stockfish-adjudicated at more than 10 pawns for 8 plies, 7 checkmates) | `m1-v1gen000-vs-v0-d3.json` |
| vs material only, same search, depth 3, 400 games | at least 55% (informational in M1) | **+259 =49 -92, 70.9%, Elo +155 [+121, +191]** | `m1-v1gen000-vs-material-d3.json` |
| vs material only, equal time (100 ms per move), 200 games | n/a | **+124 =27 -49, 68.8%, Elo +137 [+91, +188]** | `m1-v1gen000-vs-material-mt100.json` |
| law invariants (spec §11) | all pass | pass (`tests/laws.rs`, `tests/review.rs`); independent review found no deviation from the spec | |

Caveats, stated plainly:
- The match against v0 is a laws-and-correctness comparison at equal depth. The Elo printed by the harness for a 100% score is
  meaningless (it hits the clamp), so the result is reported as a game count only.
- No result here is a playing-strength claim against humans or Stockfish levels. That needs the ladder (M4).
