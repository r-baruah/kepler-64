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

## 3b. Decisions needed before stage B (check-in, 2026-10-25)

1. **Hero visual.** Recommended: the field layer (pull along legal lines, blocked at the first piece) is on by default at low intensity, and is what every share image shows. The king meter is not the hero.
2. **King readout.** The old plan had a "breaking point %" meter and a "king breaks" event (old states 5–6). The Roche gauge is still computed (gen-004 weight 2.75) but proven not to add strength, so presenting it as the mechanism breaks the claims register. Recommended: a quiet "pull on your king" readout, enemy pull on the king's body against its own side's support, both straight from `KingReport`; it is an honest early warning (F31: it separates kings about to be mated, AUC 0.93); no break event. Alternative: no king readout at all.
3. **Default opponent and its strength label.** The browser engine searches 10–30× fewer positions per move than the ladder engine did, so the ladder numbers don't describe it. Recommended: levels defined by **node budgets** (identical results to native, F34), a default of about one second on a phone, and one Stockfish ladder of that exact level before launch (needs a small backend change: per-engine limits in `match`). Default generation: gen-004, with the ladder run on it at the end. Until then the site's strength line names its setup and generation.
4. **Launch presets (alternate universes).** Recommended: Champion (gen-004); Equal start (the self-found piece values, F24; note it was trained with accretion on); Accretion on; Roche limit on (spec-1 at a strong weight); Tidal load on; Slow light (c = 1). Every preset is a real generation file; untrained variants are labelled "untrained variant of gen-004", and any Ledger number shown names the generation it was measured on.

## 4. Information architecture (draft; finalised in stage B)

| route | purpose |
|---|---|
| `/` | **Play** (default). The board, the two king meters, the universe picker, the move list. Tabs: Play · Watch · Explore |
| `/watch/<id>` | a curated real game (Replay v3) with per-move captions |
| `/explore` | paste FEN/PGN or set up a position, then see the field and the term breakdown in plain words |
| `/lab` | the laws (plain words first, then exact definitions), learned values per generation, the Law Ledger, the Stockfish ladder, the v0→v1 story, the Book, citation, install/UCI |
| `/g/<code>` | a shared game: replays it, then offers "play this universe" |
| `/u/<slug>` | a named universe preset (e.g. `slow-light`, `accretion`, `heavy-queens`) |

Header: name, one-line premise, the GitHub link with its star count. Footer: Lab, Book, GitHub, cite.

## 5. Play screen: states to wireframe in stage C

1. First visit (empty board, side picker, "Play as White / Black", universe = latest generation)
2. Your move (legal-move hints on tap/drag)
3. Kepler thinking (progress shown as depth reached, cancellable)
4. Kepler moved (last-move highlight; a field-layer pulse along the moved piece's lines)
5. (pending decision 2) King under heavy pull: the readout highlights, no event
6. (removed 2026-10-25: the "king breaks" moment presented a rejected law as the mechanism)
7. Game over: checkmate, stalemate, repetition, resignation, 50-move rule
8. End-of-game moment card (a summary built from real events: the move where Kepler's evaluation turned, the biggest capture swing, the largest change in pull on a king; with a share link)
9. Universe picker open (presets and sliders, with plain descriptions)
10. Explanation cards open
11. Phone portrait (board full width, meters under it)
12. Engine still loading (board usable for setup, "Kepler is waking up")
13. WASM unsupported (an honest message and a link to UCI)

**Field layer:**
- Off / pull / light cones.
- Pull = soft tint per square from `Φ_W − Φ_B`.
- Light cones = tap a piece to see its hop-1/2 squares.
- One layer at a time.

## 6. Data the site reads

- `engine/data/generations/index.json` and `gen-XXX.json` (constants and metadata)
- `engine/results/ladder.md` / `ladder-gen<k>.json`, `ledger-gen<k>.md` (Lab)
- Replay v3 files (Watch)
- These are copied into `web/static/data/` at build time by a script. The website never invents numbers.

## 7. Share format (draft)

- **Share link:** `/g/<code>`, where `code` is a base64url payload of `{ v: 1, gen, fen0, moves (UCI, packed), constants? (only if custom) }`. If it's longer than about 1.5 KB, fall back to a short id stored server-side (decide in stage B).
- **Preview image:** `/og/g/<code>.png`, rendered server-side at 1200×630: the board at the key moment, both meters, a one-line caption.

## 8. Tech stack (agreed direction; confirmed in stage B)

- SvelteKit + TypeScript in `web/`, `adapter-vercel`. Everything prerendered except `/og/*`.
- Canvas 2D for the board and field layers; DOM for controls. No 3D in M5.
- The engine is a Web Worker wrapping `kepler-wasm` (`03_ENGINE_ARCHITECTURE.md` §6.2).
- Until the WASM ships: `web/src/lib/engine/mock.ts` implements the same interface with canned data.
- Testing: Vitest (unit), Playwright (Play flows on desktop and phone viewports).

## 9. Open questions for stage B
- Domain.
- Board and piece set (licence-clean SVGs).
- Font choices.
- Whether the moment card is an image download, a link, or both.
- Short-id storage for long share links.
- Which universe presets ship at launch.
