# Replay JSON — schema v2

Written by `kepler64/analysis/recorder.py` (`record_game`, `record_selfplay`)
and emitted by:

```bash
python scripts/generate_replay.py game.pgn --json-only --output out/       # a PGN game
python scripts/generate_replay.py --max-plies 80 --json-only --output out/ # self-play
```

Every game is replayed through `core/universe.GameState` with the trained
constants (`kepler64/training/trained_constants.json` by default), so mass
absorbed on a capture and the velocity behind the Lorentz boost carry forward
exactly as when the engine plays. All scores are **native Kepler units**, not
centipawns.

## Top level

| key | type | meaning |
|---|---|---|
| `schema_version` | `2` | v1 had no `physics` / `highlights`, and replayed every ply at rest mass |
| `generated_at` | ISO time | |
| `engine` | `{name, depth, constants}` | `constants` = every leaf the game was scored with |
| `game` | PGN headers | |
| `initial_fen` | FEN | |
| `result` | `"1-0"`, `"0-1"`, `"1/2-1/2"`, `"*"` | |
| `highlights` | object | narration anchors, see below |
| `plies` | array | one row per half-move |

### `highlights`

| key | meaning |
|---|---|
| `collapse_plies` | plies after which either king's Roche gauge is above 0.5 |
| `first_collapse_ply` | first of those, or `null` |
| `biggest_swing_ply` | ply with the largest absolute `eval_shift_white` |
| `captures` | number of captures in the recorded plies |
| `heaviest_piece_final` | `{square, mass}` heaviest non-king body at the end |

## Per ply

| key | meaning |
|---|---|
| `ply`, `move_number`, `mover` | 1-based ply, full-move number, `"white"`/`"black"` |
| `move_uci`, `move_san`, `fen_before`, `fen_after` | |
| `eval_before`, `eval_after` | `{score_white, score_side_to_move, terms}` static eval (carried masses) |
| `played_move_evaluation` | same shape, scored with the parent, so the move-sensitivity (delta) terms are live |
| `eval_shift_white`, `eval_shift_mover` | `eval_after - eval_before` |
| `selected_candidate_rank` | rank of the played move among the engine's candidates |
| `analysis` | `{best_move_uci, candidates[], legal_move_count, depth, nodes, qnodes, cutoffs, elapsed_ms}` |
| `physics` | see below |
| flags | `is_check`, `is_checkmate`, `is_capture`, `is_castling`, `is_en_passant`, `promotion` |

`terms` keys (they sum to `total`): `tidal_enemy`, `tidal_self`,
`force_enemy_king`, `force_own_king`, `binding`, `material`, `delta_tidal`,
`delta_com`, `delta_inertia`, `delta_entropy`, `drift`, `gw`, `schwarzschild`,
`total`. All are from White's perspective.

Each `analysis.candidates[]` row: `move_uci`, `move_san`, `rank`,
`search_score_mover`, `immediate_score_white`, `immediate_score_mover`,
`terms`, `depth`, `nodes`, `qnodes`, `cutoffs`, `elapsed_ms`.

### `physics`

| key | meaning |
|---|---|
| `masses_after` | 64 signed masses (index 0 = a1, 63 = h8; White +, Black −) after the move, including accretion and Lorentz boost. Kings weigh about 1000. |
| `excess_after` | 64 signed values: carried mass minus rest mass (what captures and motion added) |
| `kings.white`, `kings.black` | `{square, mass, eta, force, roche, gauge, collapsing}` |
| `capture` | `null`, or `{captor_piece, captor_color, victim_piece, from_square, to_square, captor_mass_before, victim_mass, captor_mass_after}` |

How to read `kings`:

- `eta`: tidal-disruption index the enemy army exerts on this king (stretch). It enters the score linearly as `tidal_enemy` / `tidal_self`.
- `force`: magnitude of the enemy army's pull at the king.
- `gauge` = `sigmoid(kgain * (force − roche))`. This is the term the learned Roche threshold gates. `collapsing` means `gauge > 0.5`, i.e. the pull crossed the threshold.

**Honesty note for presenters:** the learned `roche` threshold acts on the
pull (`force`), not on `eta`. A gauge above 0.5 is the engine's "this king is
breaking" signal. It is not chess checkmate; that is `is_checkmate`.
