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
- [ ] Merge `backend/improvements` into main (fast-forward)
- [ ] History cleanup per the maintainer's private runbook, with the force-push confirmed by the maintainer at that moment
- [ ] Old Pages site replaced by a redirect stub. `deploy-pages.yml` updated in the same push
- [ ] `v1` branch created from the cleaned main

**Accept when:** main is clean, the `v1` branch exists, and the v0 baseline still runs (`KEPLER_V0_UCI` answers `uciok`).

## M1 · Core + laws spike (go/no-go)
- [ ] Workspace skeleton (`03_ENGINE_ARCHITECTURE.md` §1), CI for fmt, clippy and test
- [ ] `GameState` with accretion state, make/unmake, repetition, 50-move rule
- [ ] perft suite passes (§9 numbers)
- [ ] Laws L1–L6 exactly per `02_LAWS_SPEC.md`, with the breakdown struct
- [ ] Invariant tests 1–8 pass (`02_LAWS_SPEC.md` §11), including the French accretion regression
- [ ] `Dual<12>` scalar, and the gradient check passes
- [ ] Gen 0 calibration done and recorded (`gen-000.json`, decision log)
- [ ] Search: iterative deepening, alpha-beta, quiescence, TT, mate scoring, basic ordering
- [ ] UCI binary with the smoke test
- [ ] `kepler-train match` (UCI client for any engine) and `suite`
- [ ] Opening suite `balanced-200.epd` built
- [ ] Gate results committed: closed suite, vs v0 at depth 3, vs material-only, perf numbers

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
- [ ] Gen 1 accepted by SPRT vs Gen 0
- [ ] Law Ledger at Gen 1, every row of `04_TRAINING_AND_GATES.md` §5 (results committed)
- [ ] Spec updated if any law is cut (new spec id, decision log)
- [ ] v1 beats material-only by SPRT (the honesty gate)
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
- [ ] README rewritten for v1; Book restructured for v1 (v0 chapters marked as history)
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
