# 06 · Website plan (v1)

Status: stage A revised 2026-10-25 after M2–M4 (the backend is done; champion gen-004). Stage B follows the maintainer's answers to §3b. The old `viz-web/` frontend is retired. Don't port code or design from it.

## How we work (stages, with a check-in after each)

| stage | output | who approves |
|---|---|---|
| A · Understand | audiences, jobs, constraints (this file §1–§3) | maintainer |
| B · Plan | information architecture, screens, states, data needs (§4–§7) | maintainer |
| C · Wireframes | low-fi wireframes for every Play state on desktop and phone, in `docs/v1/wireframes/` | **maintainer, before any UI code** |
| D · Visual build | design tokens, then components, then screens, built in `web/` | maintainer, after each screen |

No visual or UI code before stage C is approved. Engine-facing code (the worker and the mock engine) can start earlier.

---

## 1. Who arrives, and what they need in the first minute

1. **Curious general visitor** from a shared link, on a phone, no chess-engine or physics background. They need to see what this is in one look, and be playing within seconds.
2. **Chess player.** Wants to play it, see its style, and know honestly how strong it is.
3. **Technical visitor** (developer, ML, physics). Wants to check it's real: the laws, the evidence, the code.
4. **Portfolio visitor.** Wants a quick, credible picture of the work and a way to go deeper.

Lessons from earlier versions:
- The old site was dense and cluttered. People said it had potential but couldn't tell what it was.
- Some redesigns became generic landing pages.
- Others looked good but lost the ability to play.
- **Playing is the one thing that may never be lost.**

## 1b. What we can show, because it's true (as of 2026-10-25)

Ordered by strength of evidence and of hook. Every line has a committed result file behind it (`09_FINDINGS.md`).

1. **Rule-shaped space.** Pull travels only along the lines a piece can legally move and stops at the first piece in the way. This law is worth **+330 Elo** over counting material alone (F16). It's the signature visual: the field layer is the hero of the site.
2. **The universe found the piece values itself.** Starting with every piece equal, two independent self-play runs sorted them queen > rook > bishop > knight at nearly the human ratios (F20, F24).
3. **Laws have to earn their place.** The Law Ledger keeps the field, spreading, held space and (at Gen 3) binding energy; it rejected accretion and both versions of a Roche limit (F16, F26, F27, F35). Rejected laws can be switched back on: alternate universes.
4. **It got stronger by playing itself:** Gen 0 → 4 in four accepted steps (+132, +81, +53, +30, engine vs engine).
5. **Strength against a known yardstick:** Gen 3 scores 53% against Stockfish 19 limited to UCI_Elo 2100 at 30+0.3 (F29). This is the native engine on a clock, **not** the browser opponent (see §3b, decision 3).
6. **It runs in your browser:** depth 6 in 64–333 ms on a phone (F34).
7. **The honest v0 → v1 story:** the old physics did nothing; it was rebuilt so every law must win games (`01_V0_AUDIT.md`).

What the site must not say (claims register): that a Roche limit, tidal force or a king "torn apart" decides the current engine's games; any rating or human comparison; anything about temperature making training work (F32).

## 2. Principles

1. **The board is the page.** The first screen is a readable board you can play on immediately. The field drawn on it (pull along legal lines) is the visual identity.
2. **What you see is what the engine computes.** Every visual is drawn from `breakdown()` (`02_LAWS_SPEC.md` §10). There are no decorative effects that imply physics the engine doesn't do.
3. **Explanation on demand.** Short labels by default; a "How does this work?" card set on request; depth in Lab. Nobody is made to scroll or read before playing.
4. **Readable first.** Flat standard board, clear pieces, coordinates, high contrast. The space feel comes from the real field (pull along move lines, glow by mass, king bodies), never from decoration.
5. **Phone is first-class.** It's designed for the phone, not squeezed onto it.
6. **Honest numbers.** Strength claims only from `engine/results/`. Scores in Kepler units. The browser engine is labelled by its generation.
7. **Accessible.** Keyboard play, `prefers-reduced-motion`, contrast AA, and colour never the only signal (pull meters show numbers too).

## 3. Constraints

- **Static-first.** The engine runs client-side (WASM in a Web Worker). The only server-side piece is the preview-image route for share links.
- **Hosting:** Vercel at **`kepler.ripuranjan.com`** (decided 2026-10-25: Kepler lives under the maintainer's own name, linked from ripuranjan.com; Gambits gets one footer line, never the domain).
- **Replays** total about 5.5 MB; load one at a time, on demand.
- **Budget:** initial JS (excluding the engine) ≤ 150 KB gzip; engine WASM loaded in parallel; first board paint < 1.5 s on a mid-range phone over 4G.
- **No accounts, no tracking cookies.** Privacy-friendly page analytics only.

## 3b. Decisions taken before stage B (maintainer, 2026-10-25; decision log)

1. **Hero visual.** Recommended: the field layer (pull along legal lines, blocked at the first piece) is on by default at low intensity, and is what every share image shows. The king meter is not the hero.
2. **King readout.** The old plan had a "breaking point %" meter and a "king breaks" event (old states 5–6). The Roche gauge is still computed (gen-004 weight 2.75) but proven not to add strength, so presenting it as the mechanism breaks the claims register. Recommended: a quiet "pull on your king" readout, enemy pull on the king's body against its own side's support, both straight from `KingReport`; it is an honest early warning (F31: it separates kings about to be mated, AUC 0.93); no break event. Alternative: no king readout at all.
3. **Default opponent and its strength label.** The browser engine searches 10–30× fewer positions per move than the ladder engine did, so the ladder numbers don't describe it. Recommended: levels defined by **node budgets** (identical results to native, F34), a default of about one second on a phone, and one Stockfish ladder of that exact level before launch (needs a small backend change: per-engine limits in `match`). Default generation: gen-004, with the ladder run on it at the end. Until then the site's strength line names its setup and generation.
4. **Launch presets (alternate universes).** Decided: see §5c. Slow light left out (a tie in the Ledger, nothing to see).

Decisions 1–3 were taken as recommended.

## 4. Information architecture (stage B, 2026-10-25)

| route | milestone | purpose |
|---|---|---|
| `/` | M5 | **Play.** Board with the field, opponent card, pull-on-king readouts, move list, controls. |
| `/g/<code>` | M5 | A shared game: replays it on the board, then "Play this universe". |
| `/u/<slug>` | M5 | Play opens with that universe preset selected (`/u/equal-start`, `/u/accretion`...). |
| `/watch` and `/watch/<id>` | M6 | The ten showcase replays (`results/replays/index.json`), one loaded at a time. |
| `/explore` | M6 | Paste a FEN or PGN, see the field and every term in plain words. |
| `/lab` | M6 | The laws (plain words first, then exact), the Ledger with rejected laws, generations and their learned values, the ladder, the v0 → v1 story, citation, UCI download. |

**Header** (every page): "Kepler-64" · one line, "A tiny universe that learned to play chess." · Play / Watch / Lab · GitHub Star button with the live count.
**Footer:** Lab · GitHub · Cite · "Made by Ripuranjan Baruah" (links ripuranjan.com) · "also building Gambits (chess training)". Nothing else.

## 5. Play (M5)

### 5a. Layout
- **Desktop (≥ 960 px):** board on the left at up to 640 px; right column: opponent card, two pull-on-king readouts (theirs on top, yours below, mirroring the board), move list, controls.
- **Phone (portrait):** header collapses to the name and a Star icon. Board full width (16 px gutters). Directly under it: the opponent line ("gen-004 · Default · thinking 0.8 s") and both king readouts on one row. Below: move list as one horizontal scrolling strip. Controls in a bottom bar: New game, Undo, Flip, Field, Universe.
- No page scroll is needed to play on either.

### 5b. Elements
- **Board:** flat, standard colours with good contrast, coordinates on the edges, cburnett pieces (see §8). Moves by tap-tap or drag; legal-move dots.
- **Field layer** (the hero; toggle with three settings: Off / Pull / Light cones; default Pull at low intensity):
  - *Pull:* each square tinted by `Φ_W − Φ_B` (one hue per side, intensity by size, capped); no tint where both are ~0. Squares are tinted, nothing is animated.
  - *Light cones:* tap or hover a piece to draw its hop-1 squares solid and hop-2 squares faint, from `breakdown().pieces[i].hops`. This is where "stops at the first piece" is seen.
  - After each move, the moved piece's hop-1 rays fade in once (300 ms; skipped with reduced motion).
- **Pull-on-king readout** (one per king): "Pull on your king: enemy 14.2 · yours 9.8", with a small two-segment bar. Numbers always shown. It's a readout, not a meter with a threshold: no percentage, no red zone, no event.
- **Opponent card:** generation and universe name, level, and one honest strength line for that exact level once its ladder exists ("At this level, Kepler scored N% against Stockfish 19 limited to X, 30+0.3, 200 games"); until then "Strength at this level: not measured yet".
- **Move list:** SAN, with Kepler's score in ku after each of its moves (labelled "ku, Kepler units" on hover/tap).
- **Explanation cards** ("How does this work?", 4 cards, each one screen on a phone): 1 Pieces have weight; 2 Pull follows the moves (with a live mini-board); 3 Learned by playing itself (the equal-start story, real numbers); 4 Laws must earn their place (Ledger summary, link to Lab).

### 5c. Universe presets (launch)
Every preset is a committed generation file. Labels are fixed text from this table.

| slug | name | file | label shown |
|---|---|---|---|
| `champion` | Champion (Gen 4) | `generations/gen-004.json` | "The current strongest universe." |
| `equal-start` | Equal start | `ablations/control-equal-masses-a-candidate.json` | "Every piece started with the same weight. It found these values by playing itself. Trained with accretion on." |
| `gen-0` | Gen 0 | `generations/gen-000.json` | "Before any self-play: human piece values, hand-set laws." |
| `accretion` | Accretion | gen-004 with `f_acc = 0.8` | "Captures absorb weight. Rejected by the Ledger at Gen 1 (+103 for switching it off). Untrained variant of Gen 4." |
| `strong-roche` | Strong Roche limit | gen-004 with `beta = 10` | "A king can be torn apart. Two versions were tried; neither made the engine stronger. Untrained variant of Gen 4." |
| `tidal-load` | Tidal load | gen-004 with `lambda_tide = 0.1` | "Enemy pull on the king's body costs points. Lost 77 Elo at Gen 3. Untrained variant of Gen 4." |

The picker shows the name, the label, and the constants that differ from the champion. The website's generation selector (all of Gen 0–4) lives in Lab, not in Play.

### 5d. Levels (node budgets)
| level | nodes per move | about, on a phone (F34) |
|---|---|---|
| Quick | 60,000 | 0.15 s |
| Default | 400,000 | 1 s |
| Strong | 1,500,000 | 3.5 s |

Same budget, same move on every device (the search is deterministic for a given node count). The worker reports depth reached as progress; "Move now" stops it early.

### 5e. States to wireframe (stage C)
1. First visit: board in the start position, "Play White / Play Black", default universe and level, the field on at low intensity, a one-line hint "Coloured squares show each side's pull".
2. Your move (legal dots; light cone of the selected piece if that layer is on).
3. Kepler thinking (depth reached, "Move now").
4. Kepler moved (last-move highlight, the moved piece's rays fade in once).
5. King under heavy pull (the readout's bar fills; no event, no colour change beyond the bar).
6. Game over: checkmate, stalemate, threefold, 50-move, insufficient material, resignation.
7. End-of-game card: result, length, and up to three moments computed from the game (the move where Kepler's score turned, the biggest capture swing, the largest jump in pull on a king), each clickable to that position; "Share" and "Play again".
8. Universe picker open.
9. Explanation cards open.
10. Phone portrait for states 1, 3, 7, 8.
11. Engine loading (board usable, "Kepler is waking up").
12. WebAssembly unavailable (plain message, link to the UCI download).
13. Shared game `/g/<code>`: replay controls, then "Play this universe".

## 6. Data the site reads (copied into `web/static/data/` by `web/scripts/sync-data.mjs` at build time)
- `generations/index.json`, every file it lists, and the preset files of §5c.
- `ladder-gen<k>.json` and, before launch, the level ladder (§10).
- Ledger tables as JSON, generated from `results/ledger-gen*.md` by the sync script; the "laws rejected" table with a generation column.
- `replays/index.json` and the ten replay files (Watch, M6), fetched one at a time.
- The script fails the build if any number the site prints lacks a source file. Copy that quotes a number takes it from these files, never from hand-typed text.

## 7. Sharing
- **Link:** `/g/<code>`, `code` = base64url of a packed payload `{v:1, preset slug or constants diff, start FEN if not standard, moves as 12-bit from-to(+promo) codes}`. A 100-move game is about 200 bytes. No server storage; long games still fit in a URL.
- **Preview image:** `/og/g/<code>.png`, rendered by a Vercel function (`@vercel/og`): the final or key position with its pull tint, the result, the universe name. The only server-side code.
- **End card download:** the same image, rendered client-side from the board canvas, for people who'd rather post a picture.

## 8. Tech
- SvelteKit + TypeScript in `web/`, `adapter-vercel`; everything prerendered except `/og/*`.
- Board and field: one Canvas 2D layer for squares and tint, SVG pieces above; DOM for controls.
- Engine: `kepler-wasm` (`--target web`) in a Web Worker; API `03_ENGINE_ARCHITECTURE.md` §6.2. The search call gains a `nodes` limit (already in `LimitsIn`).
- Pieces: cburnett set (Colin M. L. Burnett, distributed under GFDL/BSD/GPL); use it under the BSD terms with credit in Lab. Verify the licence text from the original source in stage D before shipping.
- Fonts: chosen in stage D (one sans for UI, tabular figures for numbers; self-hosted).
- Analytics: Vercel Web Analytics (no cookies).
- Tests: Vitest for the payload codec, data sync and moment detection; Playwright for Play on a desktop and a phone viewport.
- Budgets (§3) checked in CI.

## 9. Resolved and open
- Resolved: domain; hero; king readout; levels; presets; share format (no server storage); moment card is both link and image; piece set.
- Open for stage D: fonts, exact colours (tokens, light and dark), the field's hue pair.

## 10. Backend items before launch (not website code)
1. `kepler-train match`: separate limits per engine (`--a-nodes` / `--b-tc`), so Kepler can play at a node budget while Stockfish plays on its clock. Then the ladder workflow takes a `kepler_limits` input.
2. Ladder of the Default level (400k nodes) for the generation that ships, on CI.
3. Optionally, the clock ladder of the shipped generation (for Lab).
