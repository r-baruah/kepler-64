# README Rewrite — Narrative & Wording Playbook

Working document. Supersedes the narrative parts of the earlier review; keeps its fact-table and its correct calls (FERRITE numbers, SENTRY staying modest, TTV cut, no launch dates). This document covers the "why," not just the "what."

---

## 1. What changed since the last pass

- **PRAOP** still isn't built, so it goes back to being a single forward-looking line. There are no numbers or architecture details to give yet, and giving any would be inventing them, so it also loses the card of its own.
- **HeapGlass** is fully built at this point, which means it has earned promotion into the tier that gets a real card.
- **Gambits** should have its MVP out in about a week. It gets a "shipping soon" framing rather than a date, for reasons covered later.
- **N-Body Labs** is confirmed in, and one sentence is all it needs.
- **Banner** — working from the SVG you just sent (the 3:1 version). Converting it to 4:1, fixing the subtitle grammar, treating everything else as-is unless you say otherwise.

---

## 2. The narrative

### 2.1 What was wrong with the old one

You already named it: the Ryzen Standard, as it exists in the current README, is an AI artifact. Worth being specific about *why* it reads that way, because the same failure mode will happen again if we're not careful.

"The Ryzen Standard — Research-grade results on consumer-grade hardware" is a **declared standard**. It announces itself as a named framework before it's shown you anything. That's a very particular AI tic: invent a proper noun for a thing, capitalize it, and treat the capitalization as if it does the work of proving the thing exists. A human describing their own habit doesn't usually name it first and justify it after — they just say what they do, and the pattern becomes visible from the doing. "The Ryzen Standard" as a header works fine as a callback *after* the reader has already seen FERRITE survive a bit error and Kepler-64 replace piece values with mass — at that point naming the pattern is just a label for something already demonstrated. Leading with the name is the mistake, not the name itself.

### 2.2 The narrative to set

Not "I build on cheap hardware." That's a fact about your setup, not a reason anyone should care. The actual pattern across your work:

**You build things on the assumption that the environment is already hostile, and the system has to keep working anyway.**

- FERRITE assumes the transmission channel already has bit errors in it.
- Kepler-64 assumes the premise is already absurd, and builds something technically rigorous inside that absurdity anyway.
- HeapGlass assumes the program is already leaking or behaving badly, and shows you where.
- The Ryzen Standard, correctly repositioned, assumes the hardware is already the weakest plausible machine, and treats that as the honest test.
- Even the BAH hackathon story (not for the README, but worth naming here so you see the pattern) is the same shape: assume the team won't fully show up, keep the submission alive anyway.

This is a stronger narrative than a hardware flex for three reasons. It's true — every flagship project actually demonstrates it, nothing has to be stretched to fit. It's provable — you have numbers for FERRITE and a working repo for HeapGlass, so the claim isn't just asserted. And it scales — it explains why a chess engine, a satellite codec, and a memory visualizer belong on the same profile without you having to write a paragraph justifying the range.

### 2.3 The Ryzen Standard's new job

Not the philosophy. The **enforcement mechanism** for the philosophy. It's the one concrete rule that makes "I build for hostile conditions" checkable rather than aspirational: nothing ships until it's been run, honestly, on a Ryzen 5 5500U with 8GB of RAM. That's a testable claim, not a slogan. Put it after the reader has already seen one or two proof points, not before.

---

## 3. Revised one-line thesis options

**Confirmed: "I build things that keep working after something breaks."**

This is the position line. It goes right under the banner, one line, no `<br>` walls around it. Every card underneath has to earn its place against this sentence — if a project doesn't demonstrate something breaking and staying alive anyway, it's either reframed or it goes in "earlier work" without a hard sell.

---

## 4. Structural skeleton, with reasoning per block

```
[banner, 4:1]

one line of position (thesis, from section 3)

> Ryzen Standard blockquote — the rule, stated plainly, no capitalized-concept framing up front

## Proof
  FERRITE     — the numbers, [details] for full table
  HeapGlass   — what it does, [details] if there's more to show

## The one that's fun to click
  Kepler-64   — absurd premise, real rigor, one link

## Building now
  Gambits.in  — MVP framing, no date

## Earlier work
  SENTRY          — kept modest, as-is from the last review
  Exo-Checkmate   — under N-Body Labs
  Pyre            — one line, link
  (optional) N-Body Labs — one sentence, the org exists and holds the space projects

## About
  chess 1800+ · mountains, half-line if you still want it

## Footer
  email · Gambits · Codeforces · Lichess
```

Why FERRITE and HeapGlass sit together at the top instead of FERRITE alone: they're your two most concrete, most finished proof points right now, and putting the fun one next to the serious one gives a skimming reader a reason to keep scrolling instead of bouncing after the first card. Why PRAOP isn't a card: a card implies something to click into. A plan has nothing behind it yet, so it stays a single clause, most naturally tacked onto the Gambits or "building now" area as a "next up" mention, not given its own heading. That was the review's original instinct on adaptive optics and it turns out to still be correct, just for the current reason instead of the old one.

---

## 5. Project-by-project facts and example phrasing

These are illustrations of tone and shape, not final copy — you said you want the reasoning more than a drafted README right now, so treat the sentences below as calibration, not text to paste in.

**FERRITE** — everything from `note.txt`, nothing added:
> FERRITE splits orbital image frames into 64×64 tiles, each with its own CRC-16 check, so a bit error corrupts one tile instead of the whole frame. 2.40× compression, within 0.4% of fpack — but at a 10⁻⁵ channel error rate, FERRITE keeps 70% of the frame and fpack keeps effectively none. No floating point anywhere in the pipeline, because the RAD750 and LEON3 flight processors don't have an FPU. Abstract submitted to ADASS 2026, in review.

**HeapGlass** — confirmed, repo public at [r-baruah/HeapGlass](https://github.com/r-baruah/HeapGlass), numbers verified on Ubuntu 24.04 (WSL2):

> HeapGlass watches a running program's heap in real time. A C++17 `LD_PRELOAD` interceptor tracks every allocation and free and renders it live to a 256×256 grid — 65,536 cells — so you can watch memory shape itself, or catch a leak and a double-free as they happen. 1.23% overhead on the demo workload.

That's the visible card. The rest goes in `<details>` — full numbers below are real and worth keeping, but they're reference-density, not skim-density:

| Measurement | Value |
|---|---|
| Interceptor overhead (1 Hz live ticker, demo workload) | 1.23% |
| `demo_leaky` @ 30 MB/s, killed at 4s | 42,817 allocs · 212.6 MB live · Ghost window 4.2s |
| `demo_leaky` @ 25 MB/s, killed at 3s | 26,834 allocs · 133.1 MB live · Ghost window 3.1s |
| `demo_fixed`, clean 45s run | 326,292 allocs · 7.2 MB plateau (1,200 blocks) · Ghost window 45.1s |
| Relay self-test (`--smoke`) | exit 0, grid locked, 6 offenders caught |
| Grid geometry | 256×256 cells, 16 MiB min span, 8 GiB default ceiling |
| Event envelope size | HELLO 5B · ALLOC 37B · FREE 21B · DOUBLE_FREE 13B |

One thing to flag for your own writing, not for the README: the leak-rate knob (`HEAPGLASS_LEAK_MBPS`) undercounts real block size, so actual throughput runs about 1.7× the label. That's a good detail to know if you're demoing it live and someone asks why the numbers don't match the knob, but it's implementation trivia, not README material — the relay's own allocation counts are exact, only the demo's self-reported rate is approximate. Keep it out of the visible card; it's the kind of caveat that reads as noise to someone skimming a profile.

**PRAOP** — plan stage only, one line, no invented specifics:
> Next: adaptive optics wavefront reconstruction, in the research and planning stage.

That's it. No architecture claims, no numbers, because there aren't any yet. Resist the urge to make it sound further along than it is — that's exactly the kind of overclaim the "keep the leverage" instinct should be protecting you from, not producing.

**Kepler-64** — the absurd-but-rigorous one:
> A chess engine where evaluation is Newtonian gravity instead of piece tables. Pieces get mass proportional to their value, tidal force tensors score the position, JAX batches the moves. It plays weak chess. That was never the point.

**Gambits.in** — MVP-soon, no date:
> Chess training for club-level players, 1000–2200. Small bootstrapped team, MVP shipping soon.

"Shipping soon" survives longer than "launching in a week" without becoming a lie the day after it's true. Once the MVP is actually live, swap to "live" — that's a five-second edit and much safer than a countdown sitting in a public file.

**SENTRY, Exo-Checkmate, Pyre** — no changes from the earlier review's read. It got these right: SENTRY stays a small, honest hackathon-win line, Exo-Checkmate moves under N-Body Labs, Pyre stays a single line with the WoC 5.0 credit.

---

## 6. Wording rules, with actual examples

**Kill list, with the reason each one is a tell:** "leverage" (business jargon standing in for a specific verb), "delve" (nobody says this out loud), "robust" / "seamless" (adjectives doing the work a number should do), "not X, but Y" and "it's not just X, it's Y" (a rhetorical shape that shows up constantly in generated text because it's a cheap way to sound insightful without committing to a claim).

**Em dash — no formal rule, just a read-aloud test.** The earlier "one per block, project cards only" policy is itself the kind of rule you'd only write by reverse-engineering a complaint into a spec. Real test: read the sentence out loud. If the em dash is where you'd naturally pause anyway, keep it. If it's there because the sentence needed a joint and the em dash was the fastest way to add one, replace it with a period. Example — "FERRITE trades zero compression efficiency for real fault tolerance, which fpack cannot offer" reads fine as one sentence with a comma. Splitting it artificially with an em dash just to look punchier is the tell, not the punctuation itself.

**No second-person address in the body.** Covered in section 3. Applies everywhere, not just the thesis line — check every sentence for a hidden "you" that isn't part of an explicit closing invitation.

**Vary the sentence shape between cards.** If every project card is "Name — one clause — one stat," the sameness reads templated even though no individual sentence is wrong. Let FERRITE run three sentences because it has three real ideas. Let Kepler-64 be one blunt line because the joke is the whole point. Let PRAOP be a fragment because it genuinely is one.

**Numbers before adjectives, always.** "2.40× vs 2.41×, within 0.4%" beats "highly efficient." This one the earlier review already had right — keeping it because it's correct, not because it needs re-justifying.

---

## 7. Keeping the leverage, without the loopholes

This is the part your last version got wrong, and it's worth being precise about the fix, since you specifically flagged wanting this addressed.

The failure mode last time was **cagey wording** — trying to keep information ambiguous enough that a reader couldn't pin you down, while still implying more than was true. That approach has a structural problem: ambiguity is detectable. A careful reader (which is exactly who reads a systems-focused GitHub profile) notices when a sentence is working hard to avoid saying something plainly, and the noticing itself reads as evasive, which costs you more credibility than the thing you were hiding would have.

The actual leverage move is simpler and doesn't have that failure mode: **state exactly what's proven, with numbers and links, and say nothing about what isn't ready.** Silence isn't a claim. Vagueness is. "FERRITE: 2.40× compression, 70% survival at 10⁻⁵ BER, abstract in review" is a plain, checkable, confident statement. It doesn't need "promising early results" or "strong indications" bolted on to feel more impressive — those phrases are exactly the kind of hedge that invites the "prove it" reaction you're trying to avoid. If ADASS accepts the paper, the README gets one word changed. Until then, "in review" said flatly is not a weakness, it's just accurate, and accuracy read as confidence.

Concretely: don't write anything about PRAOP beyond the one plain line in section 5. Don't write anything about IITB or Innopolis at all, which you'd already correctly decided. Don't write "bootstrapped, well-funded" or similar padding around Gambits — "bootstrapped, small team" is already a complete, confident statement that invites no follow-up questions you don't want.

---

## 8. Banner: final instructions

Working from the SVG you sent (the 3:1, "FOUNDER · RESEARCH" version), since you didn't flag the provenance question, I'm taking that as confirmation to proceed with it as the base file.

- Convert canvas to **1200×300 (4:1)**. Scale the starfield, ridge paths, and star position proportionally — same visual language, wider frame, exactly like the v2 file from the last pass.
- Subtitle confirmed as **"BUILDER · RESEARCHER."** Swap to "FOUNDER · RESEARCHER" the day Gambits actually ships — one line, five seconds, no reason to do it early.
- `role="img"` and `aria-label` added.
- If "Star of the Hero" refers to one specific existing artwork you have in mind, keep this SVG as loose inspiration from it (dark sky, ridge line, single warm star) rather than something aiming to replicate that piece closely.

**Status: built.** `github-header-v3.svg` — 1200×300, corrected subtitle, `aria-label="Ripuranjan Baruah, Builder and Researcher"`. Ready to commit.

---

## 9. Status

All three blockers from the last pass are resolved: thesis line picked, HeapGlass facts in with a link, subtitle confirmed. The only thing still genuinely open is Gambits — once the MVP is live, the subtitle and the Gambits card both get their one-line updates.

Everything in this document is now specific enough to write the actual README from. Say the word and I'll assemble the full file — banner, thesis line, both proof cards, Kepler-64, Gambits, earlier work, footer — as one piece instead of examples.
