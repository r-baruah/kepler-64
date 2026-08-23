# Part 0 — Prelude: Everything You Need Before Page One

> **This chapter assumes nothing.** You don't need to know chess. You don't need to know physics. You don't need to code. Ten minutes here and every chapter that follows will make sense.

---

## 0.1 Chess in one page (only the parts Kepler uses)

Chess is played on an **8×8 grid of 64 squares** between two armies, White and Black. Each army starts with 16 pieces:

| Piece | How many | How it moves |
|---|---|---|
| **Pawn** | 8 | one square forward (two on its first move); captures diagonally |
| **Knight** | 2 | in an "L": two squares one way, then one sideways; can jump over pieces |
| **Bishop** | 2 | any distance diagonally |
| **Rook** | 2 | any distance horizontally or vertically |
| **Queen** | 1 | any direction, any distance — the strongest piece |
| **King** | 1 | one square any direction — the piece that must not be captured |

White moves first, then players alternate. You win when your opponent's King is attacked with no escape (**checkmate**). A few special rules exist — castling, *en passant*, pawn promotion — Kepler implements all of them faithfully; you only need to know they exist.

One word you'll see constantly: a **ply** is one single move by one side ("1. e4" is one ply; the pair "1. e4 e5" is two). Search depth and light-speed delays are both measured in plies.

> **The one chess fact that matters for Kepler:** every piece has a conventional *value* — pawn 1, knight/bishop 3, rook 5, queen 9. Normal engines use these numbers directly to decide who's winning. Kepler does something stranger with them, as we'll see next.

## 0.2 Gravity in one page (no equations, promise)

Newton discovered that every mass attracts every other mass, and the pull weakens with distance — quadruple the distance, quarter the pull. That is really all Kepler borrows:

- Every mass creates a **field** around itself — a region where other masses feel a pull.
- Several masses together: their fields simply **add up**. Each square feels the combined tug of everything else.
- Real astrophysics adds refinements; two matter here:
  - **Softening**: at tiny distances Newton's formula blows up to infinity. Real stars have size, so galaxies never hit that singularity. Kepler likewise slightly "fattens" each piece (Plummer softening) so forces stay finite and smooth.
  - **Finite speed**: gravity acts at the speed of light, not instantly. If the Sun vanished, Earth would keep orbiting for 8 unaware minutes. Kepler keeps this as a dial: its "speed of light" sets how far across the board a piece's influence has spread *so far*.

And one idea from Einstein's neighborhood: a heavy body inside someone else's field isn't just pulled — it is **stretched and squeezed**. Near a black hole this *tide* shreds falling objects. Remember the word **tidal**: the whole engine stands on it.

## 0.3 What a chess engine actually is

A chess engine plays by combining two parts:

1. **An evaluator** — looks at a position and returns a number: "who is ahead, and by how much?" This encodes judgment.
2. **A searcher** — mentally plays through candidate moves ("if I go here, then they go there…"), scoring the leaves of that tree of futures with the evaluator, and picks the line with the best outcome. This encodes foresight.

For forty years evaluators were lists of hand-written chess knowledge — "own a central pawn," "rooks belong on open files" — plus material values. Recently neural networks learned such judgments from millions of games instead. Both share one flaw from our standpoint: **nobody wrote down laws; experts or statistics did.**

## 0.4 The elevator pitch

> **Kepler-64 hands the machine only the laws of physics and the rules of chess — and lets the universe figure out how to play.**

Concretely: every piece becomes a **mass** on a 64-square lattice. Their combined gravity forms a field across the board. The enemy King isn't defeated by a checkmate checklist but by something older and colder — **tidal disruption**: pull on it hard enough, from enough directions at once, and the math says it tears past its **Roche limit**, the same threshold that rips moons apart around giant planets.

The twist that makes this more than a metaphor: **every constant of this universe — gravity's strength, the softening length, the speed of light, the Roche threshold — is a learnable number.** Training runs gradient descent *through an actual gravitational simulation* until the universe's judgment matches real game outcomes. Nobody tells it "rooks love open files." If it develops that preference, gravity discovered it.

Three mechanisms stack on top:
- **Accretion** — captures don't delete mass; your piece absorbs it, growing heavier (and more fragile).
- **The Multiverse** — moves are judged under many sampled variations of the constants; plans that survive in most universes win.
- **The Observer** — optionally, the engine nudges its own laws after each move, co-evolving them as the game unfolds.

All three are implemented, tested, and readable in `kepler64/` — no black boxes.

## 0.5 What kind of science is this? (an honesty note)

The board is flat and discrete; Kings are points, not stars; masses are chess values. Kepler-64 is **not** a literal simulation of the cosmos — it is a *differentiable N-body potential used as a chess evaluator*, and the project insists on saying so. The equations are real physics, computed exactly; what they mean here is an engineered analogy. This book keeps flagging the difference — including a full critique chapter (14) listing the project's own weaknesses.

## 0.6 Choose your path through this book

| You are… | Read | Skim | Skip |
|---|---|---|---|
| **Curious, no technical background** | 0 → 1 → 7 → 13 → 14 | 3, 4, 10, 12 | formalism boxes everywhere |
| **A chess player** | 0 → 7 → 10 → 12 → 14 | 1–6 for the physics | §0.1 |
| **A physicist / ML person** | 0.4 → 2 → 3 → 4 → 9 → 13 | 5, 8, 11 | §0.1, §0.2 |
| **A reviewer with one hour** | 0.4 → 7 → 14 | §12.4 (the ablation gate) | the rest |

Every concept chapter follows one rhythm: **plain words first**, then a worked picture, then the formalism, then exactly how the code uses it, then an honest ⚠ note about what is fragile there.

---

**Next:** [01_Gravity_on_a_Board.md](01_Gravity_on_a_Board.md) · Architecture map: `docs/backend_audit_and_roadmap_2026-08-23.md` (Part 1)