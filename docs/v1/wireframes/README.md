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
