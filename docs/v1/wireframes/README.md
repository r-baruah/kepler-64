# Play wireframes (stage C, 2026-10-25)

Low-fidelity wireframes for every Play state in `06_WEB_PLAN.md` §5e, desktop (1280×800) and phone (390×844). Status: **awaiting the maintainer's review.** No UI code in `web/` until they're approved.

- One file per screen (`*.dc.html`, a design-canvas component format); `Board.dc.html` draws the board and the field; `canvas.json` lays them out.
- The square tints are the real field computed by the gen-004 engine for each position (blue = White pulls harder, orange = Black; the hue pair is a placeholder). The king readouts use the engine's numbers for those positions.
- Bracketed values like `[N]` and `[STARS]` are filled in by the live game.

| # | file | state |
|---|---|---|
| 1 | `Main.dc.html` | first visit |
| 2 | `YourMove.dc.html` | your move, light cones on |
| 3 | `Thinking.dc.html` | Kepler thinking |
| 4 | `Moved.dc.html` | Kepler moved |
| 5 | `KingPull.dc.html` | king under heavy pull (no event) |
| 6 | `GameOver.dc.html` | game over |
| 7 | `EndCard.dc.html` | end-of-game card |
| 8 | `Universe.dc.html` | universe picker |
| 9 | `Explain.dc.html` | how does this work, card 2 of 4 |
| 10a–d | `Phone*.dc.html` | phone: first visit, thinking, end card, picker |
| 11 | `Loading.dc.html` | engine loading |
| 12 | `NoWasm.dc.html` | WebAssembly unavailable |
| 13 | `Shared.dc.html` | shared game `/g/<code>` |

## Alternative directions (added after the first review, 2026-10-25)

The maintainer found rows 1–3 close to the conventional chess-site layout (board left, panel right), which the old frontend also used. Two different directions, same moment (after 1. e4 e5 2. Nf3 Nc6), desktop and phone:

| file | direction |
|---|---|
| `AltUniverse.dc.html`, `AltUniversePhone.dc.html` | A · The board is the universe: one centred board, king readouts on the board edge by each king, a floating control dock, moves in a drawer |
| `AltNarrate.dc.html`, `AltNarratePhone.dc.html` | B · The universe narrates: board plus a plain-words commentary of what the field did on each move, every line a fact from the engine's breakdown |

Both use a colour-free field (White's pull lights squares, Black's pull shades them): `Board.dc.html` `palette="light"`.

## Visual concepts (2026-10-25, second review)

The maintainer asked for a space-themed, hypermodern, professional look, shown on one key screen before drawing every state. Two concepts, Play mid-game, desktop and phone (layout A with B's one-line commentary):

| files | concept |
|---|---|
| `VisObservatory*.dc.html` | 1 · Observatory: dark night-sky ground, lit-plate board, hairline instrument panels, monospace labels; the field as light (starlight for White, violet nebula for Black). Manrope + IBM Plex Mono |
| `VisMission*.dc.html` | 2 · Mission: bright, precise, one electric-blue accent; the field as a scientific dot plate (solid dots for White, rings for Black, size = strength). Geist + Geist Mono |

`BoardHi.dc.html` draws both themes from the same computed field.
