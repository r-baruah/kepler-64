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
- [x] History cleanup per the maintainer's private runbook, force-push confirmed by the maintainer (done 2026-10-24)
- [x] Old Pages site replaced by a placeholder stub (`site-stub/`). `deploy-pages.yml` updated in the same push
- [x] `v1` branch created and published on the cleaned history (it never touches `viz-web/`)

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
- [x] PVS, null move, LMR, check extension, killers and history (aspiration windows not added: not needed for the targets)
- [-] Incremental field updates: superseded by the M2 optimisations (lean path, flood-fill hops, exact cache), which met the speed target with bit-identical results
- [x] Time management (§5): soft time/30 + inc·0.8, hard min(3× soft, 0.8× remaining)
- [x] about 1M nps native (985k startpos, 1,015k Kiwipete at depth 6); evaluations bit-identical, proven by golden tests (`results/perf-m2.md`)
- [x] `kepler-wasm` with the §6.2 interface, built and tested in Node (`crates/kepler-wasm/tests/smoke.mjs`)
- [x] Depth 6 in under 1 s: desktop browser 55–307 ms, phone 64–333 ms (F34)
- [x] Replay v3 export (`kepler-train replay`); samples in `engine/results/replays/`

**Accept when:** the speed budget is met (or the shortfall is recorded with a decision) and the WASM package passes an interface test.

## M3 · Training, Gen 1+, Law Ledger
- [x] `kepler-train selfplay`, `tdleaf` (checkpoint and resume, `--threads`, `--max-minutes`), plus shard/reduce work units and `evolve`
- [x] SPRT runner (pentanomial), sequential and batch (`sprt-shard` / `sprt-reduce`)
- [-] Light Gen 1 via TDLeaf (fixed scale): rejected by SPRT, Elo −80 (`results/gen001-light.md`, findings F10–F11)
- [~] Training fixes and universe-inspired methods coded (T1 temperature, T2 mass conservation, T3/T4 natural selection with cooling, T5 quiet leaves, T6 capped games; `10_UNIVERSE_METHODS.md`). Runs wait for the maintainer's go
- [x] Gen 1 accepted by SPRT vs Gen 0: +181 Elo at depth 3, +132 at depth 4 over 400 games each (`results/gen001.md`)
- [x] Law Ledger at Gen 1, all 10 rows (`results/ledger-gen001.md`): field, spreading and space kept; accretion, binding and spec-1 Roche cut
- [x] Spec handling of cut laws decided: they stay at weight 0 as playable alternate universes (decision log 2026-10-25); spec-2 adds L5b
- [x] v1 beats material-only by SPRT (the honesty gate): +330 Elo at depth 4, 4,000 games
- [x] Gen 2+ while time allows: gen-002 (+81 over gen-001, Ledger cuts applied), gen-003 (+53 over gen-002, self-play on GitHub Actions), gen-004 (+30 over gen-003; F36)

**Accept when:** Gen ≥ 1 exists, the Ledger is published in `engine/results/`, and the material-only gate passes. If it fails, record that honestly and decide in the log.

## M4 · Ladder and showcase data
- [x] Stockfish ladder for gen-002 (`results/ladder-gen002.json`): 97% vs SF-1320, 81% vs 1500, 54% vs 1700 and 1900, 39% vs 2100 at 30+0.3 (findings F21)
- [x] Stockfish ladder for gen-003 (`results/ladder-gen003.json`): 87% vs 1500, 79% vs 1700, 62% vs 1900, 53% vs 2100, 40% vs 2300; zero crashes (F29)
- [x] Tactics suite tracked per generation (`results/suites-per-generation.md`: own-games tactics, mate in 1 and 2, closed blunders; F28)
- [x] 10 showcase games exported as Replay v3 with gen-003 (`results/replays/index.json`): 3 self-play, 3 vs Stockfish 2100/2300 from the ladder, 4 famous human games (Immortal, Evergreen, Opera, Game of the Century)
- [x] Generations manifest for the website (`engine/data/generations/index.json`: champion, lineage, Elo vs parent with its result file)

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
