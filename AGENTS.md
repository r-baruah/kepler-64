# AGENTS.md

Guidelines for any coding agent in this repo. Claude-specific details (model choice per subagent) are in `CLAUDE.md`. Keep the two consistent.

## Read first
1. `docs/v1/README.md` (the index), then `docs/v1/05_ROADMAP.md` (status).
2. `docs/v1/02_LAWS_SPEC.md` before any evaluation code. It's normative.
3. `docs/v1/03_ENGINE_ARCHITECTURE.md` for `engine/`, `docs/v1/04_TRAINING_AND_GATES.md` for matches and published numbers, `docs/v1/06_WEB_PLAN.md` for the website.

These win over the README, the Book and `kepler64/`, which describe the old v0 engine.

## Rules
- The evaluation is physics only: the quantities in the laws spec, nothing else. No piece-square tables, pawn-structure rules, opening books, tablebases or chess heuristics. Search techniques are fine.
- Training uses self-play only. Stockfish is used to test, never to teach.
- Public numbers must come from committed files in `engine/results/`. Scores are Kepler units, not centipawns. No Elo without an anchored ladder of at least 200 games and its interval.
- `kepler64/` (v0) is frozen as a baseline. `viz-web/` is retired: don't port from it.
- Change a law only by updating the spec first and logging it in `docs/v1/07_DECISION_LOG.md`.

## Delegating work
If your tool can start helper agents, give each one the cheapest model that does the job well: a small fast model for finding, reading and summarising; a mid-size model for most code and analysis (the default); the strongest model only for planning, independent reviews of the math or search, and the hardest problems. Briefs must be self-contained (absolute paths, spec sections, acceptance test). Helpers in isolated worktrees can't see gitignored files.

## Commands
From `engine/`, with `export PATH=$HOME/.cargo/bin:$PATH`: `cargo build --release -j6`, `cargo test --release -j6`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`. Matches need `KEPLER_STOCKFISH` and, for v0 comparisons, `KEPLER_V0_UCI`.

## Machine limits
One laptop: 6 cores / 12 threads, about 4 GB of free RAM, no GPU. Build with `-j6`, and run one heavy job at a time. Local high-CPU sessions are capped at about 1 hour (6 threads or fewer, `--max-minutes`). Heavy runs go to remote compute as work units (`docs/v1/08_DISTRIBUTED_COMPUTE.md`).

## Git
- Work on `v1` or a branch from it, never on `main`.
- Commit or push only when asked. Never force-push without explicit approval.
- No attribution trailers. Never commit `Planning/`, private `docs/` files, home paths or secrets.
