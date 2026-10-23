# CLAUDE.md

Instructions for Claude when working in this repo. This file stands on its own. Other agents read `AGENTS.md`, so keep the two consistent.

## Start every session here
1. Read `docs/v1/README.md`, then `docs/v1/05_ROADMAP.md` to see where things stand.
2. Before touching any evaluation code, read `docs/v1/02_LAWS_SPEC.md`. It's normative: code that disagrees with it is a bug, unless the spec is changed first and the change logged in `docs/v1/07_DECISION_LOG.md`.
3. For engine work, follow `docs/v1/03_ENGINE_ARCHITECTURE.md`. For training, matches and any number that may be published, follow `docs/v1/04_TRAINING_AND_GATES.md`. For website work, follow `docs/v1/06_WEB_PLAN.md` and its staged check-ins.
4. Maintainer-only notes (narrative, claims register, launch, runbooks) live in `Planning/`. It's gitignored, exists only in the maintainer's checkout, and must never be committed or quoted in tracked files.

Don't redo the v0 audit. If a doc looks wrong, check the code or rerun the measurement, then fix the doc.

## What Kepler-64 is (one paragraph)
A chess engine whose only judgement is a physical field. The rules of chess define the shape of space: pull travels only along the lines pieces can legally move, and stops at the first piece in the way. A king is torn apart when enemy pull across its body outweighs its defenders' (a Roche limit). Captures absorb mass. About a dozen constants (piece masses, light speed in moves, the Roche limit and others) are learned by self-play. Nothing else about chess is written in. v1 is a Rust rewrite (`engine/`) after an audit showed v0's headline physics didn't influence its moves (`docs/v1/01_V0_AUDIT.md`).

## Subagent models: choose one every time you delegate
Whenever you start a subagent (the Agent tool or a workflow), set its `model` on purpose. Don't let it inherit yours by default.

| Model | Use it for | Kepler examples |
|---|---|---|
| **Haiku** (`haiku`) | Simple, low-complexity work | Finding files, reading code or docs and reporting what's there, summarising results, doc consistency checks, listing what a suite contains |
| **Sonnet** (`sonnet`), **the default** | Most work: medium complexity | Implementing a module to a written spec, UCI, the match harness, WASM bindings, tests, benches, the website build, data export scripts |
| **Opus** (`opus`) | Only where top quality really matters | An independent review of the law math, gradient correctness or search soundness; hard debugging; design direction |

- **The main session** owns the laws spec, reviews evaluation math, gives gate verdicts (go/no-go, Law Ledger outcomes) and orchestrates. Hand the legwork down.
- When asking for an independent review, give the reviewer the code and the spec, not your conclusion.
- A forked subagent always runs on your own model. Fork only when the task needs your full context; otherwise start a fresh subagent with the right model.
- Split big jobs so each piece goes to the cheapest model that does it well (for example, Haiku collects the result files and Sonnet analyses them).
- **Briefs must be self-contained:** absolute paths, the spec sections to follow, the acceptance test, and what not to touch.
- **Worktree isolation hides gitignored files.** A subagent in an isolated worktree can't see `Planning/` or the private `docs/` files. Don't isolate tasks that need them, and pass absolute paths to the main checkout when they do.
- **One heavy job at a time on this machine** (see Compute). Don't run two subagents that both compile release builds or both play matches.

## Hard rules
- **The evaluation is physics only.** No piece-square tables, pawn-structure rules, opening books, tablebases, or hand-written chess heuristics in the evaluation (`02_LAWS_SPEC.md` §9). Search techniques are fine. If an idea can't be written as a quantity from the laws spec, it doesn't go in the evaluation.
- **It learns only from itself.** No human games or Stockfish labels in training. Stockfish is the exam (ladder, suite verification, opening-balance filter), never the teacher.
- **Every public number comes from a committed file in `engine/results/`**, produced by the commands in `04_TRAINING_AND_GATES.md`. Never invent or round up results. No Elo without an anchored ladder of at least 200 games and its interval. Scores are Kepler units (ku), never centipawns.
- **Honest failure is a result.** If a law is cut by the Ledger or v1 fails a gate, write that down plainly.
- **v0 is frozen.** Don't change `kepler64/` except to keep it runnable as the baseline.
- **`viz-web/` is retired.** Don't port code or design from it.
- **Banned words** in public copy: game-changer, revolutionary, delve, unleash, seamless, testament to, "a symphony of", and the "it's not X, it's Y" shape. No emoji in code, docs or UI.

## How the maintainer likes to work
- Plain, natural language in replies and docs. Explain the reasoning so decisions can be pushed back on.
- When a decision is genuinely the maintainer's, ask. Otherwise recommend one option and proceed.
- Website design goes in stages with a check-in after each: understand, plan, low-fi wireframes, visual build. No UI code before the wireframes are approved.
- Record decisions in `docs/v1/07_DECISION_LOG.md` and tick `docs/v1/05_ROADMAP.md` as work lands.

## Repo layout
```text
engine/         v1: Rust workspace (kepler-core, kepler-uci, kepler-train, kepler-wasm), data/, results/
docs/v1/        v1 specs and plans (tracked). Other files in docs/ are private and untracked
web/            v1 website (SvelteKit), from milestone M5
kepler64/       v0 engine (Python + JAX), frozen baseline; moves to legacy/ at release
scripts/audit/  the v0 audit scripts
Kepler-64_Book/ the Book (describes v0 until it's restructured in M6)
```

## Commands
Engine (from `engine/`):
```bash
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release -j6
cargo test --release -j6
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo bench -p kepler-core
```
Matches and suites: `cargo run --release -p kepler-train -- <selfplay|tdleaf|match|ladder|ablate|suite|openings|calibrate|replay> --help`.

Environment for matches:
- `KEPLER_STOCKFISH`: the path to the Stockfish binary (the maintainer's copy is in `~/.local/opt/stockfish/`).
- `KEPLER_V0_UCI`: the command that starts the v0 engine, e.g. `/path/to/venv/bin/python -m kepler64.uci`, run from a v0 checkout.

v0 (baseline only): `python -m kepler64.uci` and `pytest kepler64/tests` inside its virtualenv.

## Compute (one laptop)
- Ryzen 5 5500U (6 cores, 12 threads), about 7 GB RAM with about 4 GB usually free, no GPU, limited free disk.
- Build with `-j6`. Run at most one release build or one match/training job at a time.
- **The maintainer allows at most about 1 hour of high CPU per local session.** Default to 6 threads or fewer, always pass `--max-minutes`, and rely on checkpoints to stop and resume. Never start multi-hour local runs.
- Heavy runs (full generations, the Law Ledger, the Stockfish ladder) go to remote compute as work units: `docs/v1/08_DISTRIBUTED_COMPUTE.md`.

## Git
- Work on the `v1` branch (or a branch from it), never directly on `main`. `main` is the public, stable branch until v1.0.
- Commit or push only when the maintainer asks. Never force-push without the maintainer's explicit yes at that moment.
- No co-author or attribution trailers in commits or pull requests.
- Never commit `Planning/`, private `docs/` files, absolute home paths, or secrets.
