# Pass 4: three visual concepts

Exploration only. Nothing here touches `web/`. Each concept is one self-contained HTML file with a light/dark toggle (top right; `?theme=light` or `?theme=dark` also works). Each was checked at 390 and 1440 wide. The boards are static drawings, not the engine.

| File | Direction | Sections rendered |
|---|---|---|
| `concept-a.html` | Orbital: mission plot | Hero, 01, phone dock |
| `concept-b.html` | Plate: astronomy plate and catalogue, one NASA photograph | Hero, 03, phone dock |
| `concept-c.html` | Editorial: big type, ticker, Kepler-64b drawing | Hero, 01, 02 (proof), phone dock |

Shared across all three:

- Fonts are Geist and Geist Mono, copied with their OFL licences into `assets/fonts/`.
- Pull tints are mixed onto the squares with `color-mix()`.
- Piece glyphs are Unicode chess symbols forced to text style (U+FE0E), so they never render as emoji.
- All motion stops under `prefers-reduced-motion: reduce`, and every page has a sensible resting frame.
- Copy is reused from the live site. New text, every piece of which needs the maintainer's yes before it moves into `web/`:
  - A: eyebrow "Chess engine, physics only"; plot labels "Aphelion, slowest" and "Equal areas in equal times"; column indices 01 to 04.
  - B: nav tagline "A field study of chess"; "Plate I", "Plate II", "Earth over the Moon", the photo caption and credit; "The field, illustrative"; ledger row numbers 1 to 5; universe captions, including Home's "The champion. Weights it learned itself."
  - C: "The idea", "The field, illustrative"; the Kepler-64b caption and "Second pair, far out".
  - All three: "Your move", "A slimmer dock.", the dock note, and the footer labels ("Concept X", "Static board, no engine").
- The board tint is illustrative in all three: ranks 1 to 3 White, 6 to 8 Black, 4 and 5 neutral. The live engine's start-position field is graded and subtler. The real build must draw the engine's values.

## A. Orbital

**Idea.** The hero is a large mission-design plot drawn on graph paper. The board sits at the focus of a Kepler ellipse, and one body orbits it. Two shaded wedges show the second law: the body sweeps equal areas in equal times. The aphelion and the empty second focus are marked. Section 01 is a four-column translation table (physics drawing, then the chess version), followed by the rook worked example. Light is warm paper with navy ink and a trajectory-orange accent. Dark is deep navy with the same orange, and no glow.

**Motion.** The body's 48 s orbit follows Kepler's equation exactly. Positions come from solving M = E - e sin E at 97 equal-time steps (e = 0.3), and they are baked into CSS keyframes. The body visibly speeds up near the board and slows far from it. The wedges are exact equal-time slices, and the perihelion clears the board's corners. Under reduced motion the body rests at aphelion, beside its label.

**Mark.** A ring around a close pair of stars, with a planet on it. It nods to Kepler-64b without any caption.

**Section 01 numbering.** The four columns are numbered 01 to 04, not "Law 1 to 4". Mass and light speed are learned constants, not laws, and the Ledger's laws are a different list.

| Token | Light | Dark |
|---|---|---|
| bg | #F5F5F1 | #0E1626 |
| ink | #121A2E | #E9ECF2 |
| ink-2 | #2E3A55 | #C3CAD8 |
| muted | #596377 | #949EB2 |
| line | #D9DBD6 | #26324A |
| accent | #B93C0B | #F08A50 |
| card | #FFFFFF | #131D31 |
| soft | #ECEDE7 | #192439 |
| board light / dark | #EAE8E1 / #C6CAD0 | #3A4660 / #2A354E |
| White pull (mixed 42%, dark 52%) | #E2A15E | #E3B98A |
| Black pull | #7E95BE | #6F92D6 |

**Contrast** (on bg / card / soft). Light: ink 15.8 / 17.3 / 14.7, muted 5.5 / 6.0 / 5.1, accent 5.2 / 5.7 / 4.8. Dark: ink 15.3 / 14.2 / 13.1, muted 6.7 / 6.3 / 5.8, accent 7.3 / 6.8 / 6.3.

**Hard to build.**
- The orbit SVG is sized as a percentage of the board wrapper, so it follows the real board. When the board resizes or the Play screen takes over, it needs its own layer and clipping.
- In dark mode the White-pull tint goes greige over the navy squares. It needs another pass.
- At 390 the hero is clipped, so most of the ellipse is off screen and only the arcs and the body show.

## B. Plate

**Idea.** A field-study document. Plate I is the Apollo 17 Earthrise photograph in a cyanotype duotone (Prussian blue in light, graphite in dark). The headline is set in the empty sky. A réseau grid of small crosses overlays the photo, as on real astronomical survey plates. Plate II is the board, and the weights sit below it as a ruled catalogue. Section 03 is the Ledger as an actual ledger table with verdict stamps, followed by six universe plates (Home, Genesis, First Light, Hunger, Roche, Undertow), each with a small line drawing. Mono labels, square corners, double rules.

**Motion.** The photograph drifts and zooms by 6% over 70 s. A thin scan line moves down the plate every 14 s, like a plate-measuring machine. Both stop under reduced motion.

**Image.**
- NASA ID: `AS17-152-23272`. Title: "Apollo 17, Earthrise over Moon".
- Page: https://images.nasa.gov/details/as17-152-23272 (checked, HTTP 200)
- File: https://images-assets.nasa.gov/image/as17-152-23272/as17-152-23272~medium.jpg (1280 x 1280, 113 KB). Saved as `assets/nasa/as17-152-23272-medium.jpg`.
- Licence: NASA imagery, generally public domain. It is credited on the page. No NASA insignia is used, and the photo is not mirrored.

| Token | Light | Dark |
|---|---|---|
| bg | #F1EEE6 | #10141C |
| ink | #121C33 | #ECE8DE |
| ink-2 | #2B3A5C | #CFC9BA |
| muted | #596276 | #9C998F |
| line | #D6D0C2 | #2A303C |
| accent | #1F3F8C | #93B2F2 |
| card | #F9F7F1 | #161B25 |
| soft | #E7E2D6 | #1C2230 |
| board light / dark | #EDE8DB / #C9CDD5 | #3C4351 / #2C323E |
| White pull | #C9974F | #D9AE72 |
| Black pull | #3E5FA8 | #7E9EE0 |
| duotone shadow / highlight | #142A5C / #E9EDF3 | #0D121B / #B3BDCC |

**Contrast** (on bg / card / soft).
- Light: ink 14.6 / 15.8 / 13.1, muted 5.3 / 5.7 / 4.7, accent 8.4 / 9.1 / 7.6.
- Dark: ink 15.1 / 14.1 / 13.0, muted 6.5 / 6.1 / 5.6, accent 8.7 / 8.1 / 7.5.
- Headline on the duotone sky: 12.0 (light), 15.3 (dark).

**Hard to build.**
- The headline sits on the photo, so the crop must keep the Earth clear of the text at every width. There are two object-position presets now; any new breakpoint needs checking.
- The duotone is an SVG `feComponentTransfer` filter, swapped per theme. It is cheap, but the drifting image is repainted through the filter. A pre-made pair of tinted JPEGs would be lighter.
- The réseau crosses sit behind the headline and add a little noise there.
- "A field study of chess" in the nav is new framing copy and needs the maintainer's yes.

## C. Editorial

**Idea.** Poster typography: a 142 px headline whose full stop is a tiny orbit diagram, with an accent body moving on a Kepler ellipse (the same keyframes as A, 16 s). Below it, three ruled columns hold the idea and buttons, the board, and the weights as big mono numerals. Next comes a full-bleed accent ticker of the four translations: Mass to Piece weight, and so on. Section 01 is set as huge struck-through rows ("~~Mass~~ Piece weight") with small boards. Section 02 gives the 53% figure poster size, with the proof line word for word, the caveat beside it, and the not-inside list struck through. Black-on-paper with signal red.

**Kepler-64b.** The empty space at the top right of the hero holds a line drawing. A close pair of stars turns every 7 s, the planet circles both every 48 s, and an arrow points to "second pair, far out". Caption: "Kepler-64b orbits a pair of stars, with a second pair far out. This project shares its name."
- The drawing deliberately shows the planet circling the pair, not sitting between them, because that is the real configuration.
- The animation periods are illustrative and do not claim the real ratio.
- At 390 the drawing becomes a small row after the weights, so the hero order stays heading, sub, buttons, board, weights.
- The same mark (a ring around two stars, with a planet) is used in the nav.
- **Naming is the maintainer's call.** The live site says the project is named after Johannes Kepler; the coordinator allowed "named after the planet". The caption uses "shares its name", which is true either way.
- **The arriving-field animation is illustrative.** Pull arriving row by row over seconds could read as light speed measured in time, but light speed is counted in moves. The real build should drop it or rethink it.

**Motion.**
- The full-stop orbit.
- The pull arriving on the board row by row from each side, holding, then fading, on a 9 s loop.
- The ticker, 60 s per loop.
- The binary and the planet.

All of it stops under reduced motion, where the field is shown fully on.

| Token | Light | Dark |
|---|---|---|
| bg | #F3F2EE | #141414 |
| ink | #0F0F10 | #F3F2EE |
| ink-2 | #2D2D30 | #D2D0CA |
| muted | #5C5C61 | #A09E98 |
| line | #D6D4CE | #2E2E2E |
| accent | #B42A19 | #FF6450 |
| card | #FFFFFF | #1B1B1B |
| soft | #E9E7E1 | #222222 |
| board light / dark | #ECEAE4 / #CDCAC2 | #4A4945 / #363532 |
| White pull (overlay 36%, dark 45%) | #E7705F | #FF7A66 |
| Black pull | #5A6FB8 | #8EA2F0 |

**Contrast** (on bg / card / soft).
- Light: ink 17.1 / 19.2 / 15.5, muted 5.9 / 6.7 / 5.4, accent 5.7 / 6.4 / 5.2.
- Dark: ink 16.5 / 15.4 / 14.2, muted 6.9 / 6.4 / 5.9, accent 6.3 / 5.9 / 5.5.
- Ticker text on accent: 5.7 (light), 6.3 (dark).

**Hard to build.**
- The headline orbit is an inline SVG inside an `h1`; the real full stop is visually hidden for screen readers. It must survive line wrapping at every width.
- Red and black in dark mode sit closest to the rejected "Halloween" look of anything here. The dark White-pull tint reads as brick.
- The ticker repeats its content four times, so it needs `aria-hidden` (it has it) and a pause control for WCAG 2.2.2, since it moves for more than 5 s.
- At 1440 the 01 rows run to about 1.5 screens.

## Phone Play dock (all three)

The current dock is a big pill with five text buttons. The proposal is a slim toolbar of five 20 px stroke icons with short labels: New, Undo, Field (with its mode, "Pull", shown inline), Universe, More. Each concept styles it its own way:

- **A:** a hairline-topped card bar, with Field in the accent.
- **B:** a mono-label bar with ruled dividers.
- **C:** a solid ink bar.

It is a static mock under a "Your move" strip.

## Checks run

- Contrast: a WCAG ratio script over every text token on bg, card and soft, in both themes. Every body-text pair is at least 4.5:1. C's accent was darkened from #C8301E to #B42A19 to pass on soft.
- Copy: headings are 8 words or fewer and bodies 25 or fewer. The one exception is the proof line, which is kept word for word. The rook caption is set as body text, as on the live site.
- Banned words, sci-fi words and emoji: none found.
- Layout: no horizontal scroll at 390 or 1440 (`scrollWidth` equals the viewport width), in both themes.
