# Kepler-64 — *The Roche Engine*

### A Complete, Self-Contained Textbook of an Astrophysical Chess Engine

**One-paragraph summary.** Kepler-64 is a chess engine whose evaluation function is not a hand-tuned heuristic and not a black-box neural network — it is a **differentiable, JAX-compiled N-body gravitational simulation**. Chess pieces are masses placed on a 64-square lattice; the board's state is fed through Plummer-softened Newtonian gravity, a tidal-tensor (Hessian) computation at the King, and the resulting dimensionless disruption parameter η. A position is "won" when the enemy King is tidally disrupted past a learned Roche limit. Every constant of this chess-universe — the gravitational constant G, the softening ε, the speed of light c, the Roche threshold, and several scaling knobs including a Verlet tidal-drift gain — is a *learnable leaf* that gradient descent moves. A short symplectic (Leapfrog) rollout projects each King forward under the opponent's field to read the "impending collapse" $d\eta/dt$ signal. Layer 2 ("the Multiverse") draws these constants from a posterior, accretes captured mass, and lets the engine observe itself. This book teaches every piece of physics, mathematics, and systems engineering needed to understand the whole project from zero — and then critiques the plan itself.

---

## How to read this book

The book is organized in four Parts plus an Appendix:

- **Part 0 — Prelude** ([PRELUDE.md](PRELUDE.md)): chess, gravity, and engines explained from zero; the elevator pitch; reading paths for every audience. *Start here if the project is new to you.*
- **Part A — Foundations** (Chapters 1–6): the physics and math you must know *before* the architecture. Newtonian gravity, Plummer softening, the tidal tensor, Roche/Hill scaling, the speed of light as a gate, and automatic differentiation.
- **Part B — The Core Engine** (Chapters 7–11): how Kepler-64 turns a chess position into a mass vector and runs it through the gravity → tidal → η → score pipeline, plus the 218-pad vmap trick and the Verlet rollout.
- **Part C — Learning & Layer 2** (Chapters 12–13): training through physics and the Multiverse/Observer/Accretion mechanisms.
- **Part D — Critique & Strategy** (Chapters 14–16): an independent review of every gap, bug, misassumption, and missed innovation ([14_Review.md](14_Review.md)), the requirements mapping ([15_Mapping.md](15_Mapping.md)), and the empirical case study on the Alien Physics Fallacy ([16_The_Alien_Physics_Fallacy.md](16_The_Alien_Physics_Fallacy.md)).
- **Appendix**: glossary, formula sheet, references.

**Prerequisites.** A 12th-grade science background (basic algebra, vectors, Newtonian mechanics, a little statistics). No chess-engine experience required and no deep CS theory. Wherever a concept is first used it is defined once and linked back.

**Conventions.** Each concept follows the pattern: *Intuition* (plain language + analogy) → *Formalism* (the math, variable by variable) → *Project link* (how Kepler-64 uses it) → optionally a `⚠` review note showing the flaw at the moment you learn the concept. Every critique item is also collected and deepened in Part D.

---

## What you will be able to do after reading

- Derive the force field, tidal tensor, and η by hand for a toy position.
- Explain *why* the engine plays the way it does, from first principles.
- Read any file in `kepler64/` and know exactly which physics concept it implements.
- Argue, with evidence, what is rigorous versus what is an honest metaphor — and what should be fixed before this project is shown to a skeptical reviewer.

---

## Source documents this book is built from

- README.md, Planning/Kepler-64.md, Planning/Kapler-64 Blog Writeup.md
- Planning/Kepler-64 Code Review.md, Planning/Kepler-64 Code Review v2.md
- Planning/Kepler-64 Audit & Layer-2 Development.md
- Planning/Kepler-64 P II.md, Planning/Kepler-64 Extended Absurdity.md, Planning/Kapler-64 Scaffold.md
- Planning/rg-dynamic-future.md, Planning/verification_and_training_suggestions.md
- The full `kepler64/` codebase (core, search, training, multiverse, viz, bench, tests)

*Throughout, file:line references point at the actual implementation so the book can be read side-by-side with the code.*
