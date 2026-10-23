# Kepler-64 v1 docs

These files are the source of truth for v1. They win over the README, the Book and anything in `kepler64/`, which describe v0.

| file | read it when |
|---|---|
| [01_V0_AUDIT.md](01_V0_AUDIT.md) | you want to know why v1 exists: what v0 actually computed, measured |
| [02_LAWS_SPEC.md](02_LAWS_SPEC.md) | **before touching any evaluation code.** The exact laws, constants, invariants |
| [03_ENGINE_ARCHITECTURE.md](03_ENGINE_ARCHITECTURE.md) | working in `engine/`: layout, search, interfaces (UCI, WASM, Replay v3), performance, build rules |
| [04_TRAINING_AND_GATES.md](04_TRAINING_AND_GATES.md) | training, matches, suites, the Law Ledger, the Stockfish ladder, and how a number becomes publishable |
| [05_ROADMAP.md](05_ROADMAP.md) | what's done and what's next (milestones with acceptance criteria) |
| [06_WEB_PLAN.md](06_WEB_PLAN.md) | any website work: stages, screens, data, stack |
| [08_DISTRIBUTED_COMPUTE.md](08_DISTRIBUTED_COMPUTE.md) | running training or tests anywhere but the local machine: work units, GitHub Actions, the mesh orchestrator, the local-compute policy |
| [09_FINDINGS.md](09_FINDINGS.md) | you learned something from the engine: the lab notebook of findings, with evidence and status |
| [07_DECISION_LOG.md](07_DECISION_LOG.md) | why something is the way it is; add an entry when you change a decision |

In one paragraph: Kepler-64 is a chess engine whose evaluation is a single physical field. The rules of chess define the shape of space (pull travels only along the lines pieces can legally move), and a dozen-odd constants (piece masses, light speed, the Roche limit and others) are learned by self-play. Nothing else about chess is written in. v1 is a ground-up rewrite in Rust after an audit showed that v0's headline physics didn't influence its moves.
