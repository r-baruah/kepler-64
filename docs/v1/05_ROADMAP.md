# 05 · Roadmap to v1.0

Status as of 2026-10-23. Tick items as they land and give the commit or result file next to each. A milestone is done only when every acceptance line is ticked. Development happens on the branch `v1`. `main` stays the public, stable branch until the release.

Legend: `[ ]` open · `[x]` done · `[~]` in progress · `[-]` dropped (say why)

---

## M0 · Housekeeping
- [x] Audit v0 and save the scripts (`scripts/audit/`, `01_V0_AUDIT.md`)
- [x] Back up uncommitted local work outside the repo
- [x] Rust toolchain (user-level rustup, `wasm32-unknown-unknown`, clippy, rustfmt)
- [x] Stockfish 19 available locally (`KEPLER_STOCKFISH`)
- [x] Docs: `CLAUDE.md`, `AGENTS.md`, `docs/v1/` 01–07
- [x] Merge `backend/improvements` into main (fast-forward, local only; nothing pushed)
- [ ] History cleanup per the maintainer's private runbook, with the force-push confirmed by the maintainer at that moment (blocked 2026-10-24: the rewrite step needs the maintainer to allow or run it)
- [ ] Old Pages site replaced by a redirect stub. `deploy-pages.yml` updated in the same push
- [~] `v1` branch created (from local main; it gets replayed onto the cleaned main after the rewrite; it never touches `viz-web/`)

**Accept when:** main is clean, the `v1` branch exists, and the v0 baseline still runs (`KEPLER_V0_UCI` answers `uciok`).

## M1 · Core + laws spike (go/no-go)
- [x] Workspace skeleton, CI for fmt, clippy, test and the wasm32 build (`.github/workflows/engine.yml`)
- [x] `Position` with accretion state (copy-make), repetition, 50-move rule, insufficient material
- [x] perft suite passes (§9 numbers); perft(5) from the start position takes 23 ms
- [x] Laws L1–L6 per spec-1, with the breakdown struct (independent Opus review: no deviation from the spec; per-piece §10 fields still missing, moved to M2)
- [x] Invariant tests pass (`tests/laws.rs`, `tests/review.rs`), including the French accretion regression
- [x] `Dual` scalar (12 slots), and the per-position gradient check passes
- [x] Gen 0 calibration done and recorded (`gen-000.json`, `results/calibration-gen000.md`)
- [x] Search: ID, PVS, quiescence, TT, mate scoring, null move, LMR, killers, history (about 510k nps)
- [x] UCI binary with the smoke test
- [x] `kepler-train match` (UCI client for any engine) and `suite`
- [x] Opening suite `balanced-200.epd` built (505 tried, 200 kept, SF19 d14 within ±50 cp)
- [x] Gate results: **GO** (`engine/results/m1-gate.md`). Closed suite 0/12 avoid-moves; vs v0 +100 =0 -0 at d3; vs material-only 70.9% at d3 and 68.8% at equal time

**Accept when:** every check in `04_TRAINING_AND_GATES.md` §7 passes and the result files are in `engine/results/`. **No-go** means revising the laws and repeating M1's gate (log it).

## M2 · Full search, speed, WASM
- [ ] PVS, null move, LMR, check extension, aspiration windows, killers and history
- [ ] Incremental field updates (recompute only affected pieces), equal to the from-scratch result in tests
- [ ] Time management (§5)
- [ ] ≥ 1M nps native; perf file committed
- [ ] `kepler-wasm` with the §6.2 interface, built and tested in Node
- [ ] Depth 6 in under 1 s in a desktop browser, plus a measured number on one real phone
- [ ] Replay v3 export (`kepler-train replay`)

**Accept when:** the speed budget is met (or the shortfall is recorded with a decision) and the WASM package passes an interface test.

## M3 · Training, Gen 1+, Law Ledger
- [ ] `kepler-train selfplay`, `tdleaf` (checkpoint and resume, `--threads`, `--max-minutes`)
- [ ] SPRT runner (pentanomial)
- [-] Light Gen 1 via TDLeaf (fixed scale): rejected by SPRT, Elo −80 (`results/gen001-light.md`, findings F10–F11)
- [~] Training fixes and universe-inspired methods coded (T1 temperature, T2 mass conservation, T3/T4 natural selection with cooling, T5 quiet leaves, T6 capped games; `10_UNIVERSE_METHODS.md`). Runs wait for the maintainer's go
- [x] Gen 1 accepted by SPRT vs Gen 0: +181 Elo at depth 3, +132 at depth 4 over 400 games each (`results/gen001.md`)
- [x] Law Ledger at Gen 1, all 10 rows (`results/ledger-gen001.md`): field, spreading and space kept; accretion, binding and spec-1 Roche cut
- [ ] Spec updated if any law is cut (new spec id, decision log)
- [x] v1 beats material-only by SPRT (the honesty gate): +330 Elo at depth 4, 4,000 games
- [ ] Gen 2+ while time allows

**Accept when:** Gen ≥ 1 exists, the Ledger is published in `engine/results/`, and the material-only gate passes. If it fails, record that honestly and decide in the log.

## M4 · Ladder and showcase data
- [ ] Stockfish ladder for the latest generation (`ladder-gen<k>.json`, `ladder.md`)
- [ ] Tactics suite tracked per generation
- [ ] 8–12 showcase games exported as Replay v3 for Watch mode (self-play, vs a ladder rung, famous human games judged by Kepler)
- [ ] Generations manifest for the website (`data/generations/index.json`)

## M5 · Website: Play first
Plan and stages are in `06_WEB_PLAN.md`. Design follows the staged check-ins there.
- [ ] Stage A (understand) and stage B (plan) written into `06_WEB_PLAN.md`
- [ ] Stage C: low-fi wireframes approved by the maintainer
- [ ] Stage D: visual build of Play (board, pull meters, field layer, universe picker, end-of-game moment, share link, Star)
- [ ] Hosted on Vercel at the chosen domain. Soft launch to 5–10 cold testers, findings written down

## M6 · Website complete + docs
- [ ] Watch, Explore, Lab (laws, Ledger, Generations, the v0→v1 story, cite)
- [ ] Preview-image route for share links; clip export
- [ ] README rewritten for v1; a new Book written from scratch for v1 (the v0 Book was removed from `v1` on 2026-10-24; git history keeps it)
- [ ] `kepler64/` moved to `legacy/`; Python CI replaced by cargo CI

## M7 · Release v1.0
- [ ] Tag `v1.0.0`, changelog, `CITATION.cff` version bump, GitHub release with UCI binaries (Linux, Windows, macOS)
- [ ] `v1` merged into `main`
- [ ] Launch, following the maintainer's private plan

---

## Dependencies
```
M0 ─▶ M1 ─▶ M2 ─▶ M3 ─▶ M4 ─▶ M6 ─▶ M7
            │                  ▲
            └──▶ M5 (stage A–C can start during M2, against the WASM mock) ┘
```
