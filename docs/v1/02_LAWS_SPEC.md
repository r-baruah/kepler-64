# 02 · The v1 laws (normative spec)

Status: **draft v1.0-spec-1** (2026-10-23). This file is the single source of truth for what the v1 engine computes. Code that disagrees with it is a bug, unless this file is updated first and the change is logged in `07_DECISION_LOG.md`.

The principle: **the rules of chess are the shape of space. A field is the only law. Learning tunes the constants.**

Plain-words version, used everywhere public:
- Every piece has weight.
- Its pull travels only along the lines it can legally move, and stops at the first piece in the way.
- Pull that would need more moves to arrive is weaker, and beyond the light-speed limit it doesn't arrive at all.
- A king is torn apart when the enemy's pull across its body outweighs what its own side holds there.
- Whatever captures absorbs some of the victim's weight.
- How strong all of this is was learned by the engine playing itself.

Honesty note, to repeat in public material: this is a field on the board's move graph, not Newtonian gravity in Euclidean space. The words "weight", "pull", "light speed" and "Roche limit" name exact quantities defined below. None of them is a metaphor without a formula.

---

## 0. Notation

- Squares `s ∈ {0..63}`, with `a1 = 0`, `h1 = 7`, `a8 = 56`, file = `s mod 8`, rank = `s div 8`.
- Colours `C ∈ {W, B}`. `¬C` is the other colour.
- A piece `p` has a type `t(p) ∈ {P, N, B, R, Q, K}`, a colour `col(p)`, a square `s_p` and a carried excess mass `A_p ≥ 0`.
- `occ` is the occupancy bitboard of the position.
- All scores are in **Kepler units (ku)**. 1 ku = the rest mass of one pawn. Scores are never centipawns. UCI output may print `cp = round(100 · ku)` because protocol tools expect it, and docs must say that's a unit conversion, not a centipawn claim.

---

## 1. The constants

There are 14 constants. **Gen 0** is the hand-set universe; training produces Gen 1, Gen 2 and so on. "Grad" says how a constant is tuned (§7): `A` = analytic gradient (TDLeaf), `O` = outer loop (match-based), `X` = discrete ablation.

| # | id | meaning (plain) | law | Gen 0 | bounds | grad |
|---|---|---|---|---|---|---|
| 1 | `m_n` | knight rest mass | L1 | 3.0 | [1, 8] | A |
| 2 | `m_b` | bishop rest mass | L1 | 3.0 | [1, 8] | A |
| 3 | `m_r` | rook rest mass | L1 | 5.0 | [2, 12] | A |
| 4 | `m_q` | queen rest mass | L1 | 9.0 | [4, 20] | A |
| 5 | `kappa` | king emission mass (how hard the king itself pulls) | L2 | 1.0 | [0, 5] | A |
| 6 | `alpha` | pull kept per extra move of travel | L3 | 0.35 | [0, 1] | A |
| 7 | `sigma` | how much a piece's pull spreads thin over the squares it reaches (0 = no spreading, 1 = conserved flux) | L2 | 0.5 | [0, 1] | A |
| 8 | `c` | light speed: the most moves pull can travel | L3 | 2 | {1, 2, 3} | X |
| 9 | `lambda_bind` | weight of binding energy | L4 | calibrated (§8) | [0, 1] | A |
| 10 | `lambda_space` | weight of held space | L4 | calibrated (§8) | [0, 2] | A |
| 11 | `rho` | Roche limit (breaking point of the king's index) | L5 | 1.5 | [0.1, 10] | A |
| 12 | `k_roche` | sharpness of breaking | L5 | 3.0 | [0.1, 20] | A |
| 13 | `beta` | weight of a breaking king | L5 | calibrated (§8) | [0, 30] | A |
| 14 | `f_acc` | fraction of a victim's mass the captor absorbs | L6 | 0.8 | [0, 1] | O |

Fixed by definition (not constants): the pawn rest mass `m_p = 1` (the unit), and `tanh` scale 1 ku in L4b.

Every bound is a human prior. Each one is listed here with the reason it exists. A bound that training pins against must be reported in the Law Ledger (`04_TRAINING_AND_GATES.md` §5), never hidden.

- Mass bounds keep the unit meaningful (no piece lighter than a pawn) and stop runaway growth.
- `alpha` and `sigma` bounds are mathematical (fractions).
- `rho`, `k_roche` and `beta` bounds only stop numerical blow-up.

---

## 2. L1 · Mass

**Rest mass** `μ(t)`:
- `μ(P) = 1`
- `μ(N) = m_n`, `μ(B) = m_b`, `μ(R) = m_r`, `μ(Q) = m_q`
- Kings have no rest mass for material purposes. They are bodies that can be destroyed, not material.

**Carried mass** `M_p = μ(t(p)) + A_p` for non-kings. For kings, `M_K = kappa + A_K` (see L2 emission).

**Material term** (White's point of view):
```
T_mat = Σ_{p ∈ W, t≠K} μ(t(p))  −  Σ_{p ∈ B, t≠K} μ(t(p))
```
Material uses **rest mass only**. Accreted mass never counts as material. This makes trades zero-sum again and fixes the v0 laundering bug (`01_V0_AUDIT.md` §3). Accreted mass does count in the field (L2) and in energy (L4): a fat piece pulls harder and is more exposed.

---

## 3. L2 · Rule-shaped space

**Attack set** `Att(p, occ)`: the squares piece `p` attacks under the rules of chess.
- Pawn: the two forward-diagonal capture squares for its colour, on the board. **Pawn pushes do not emit field.**
- Knight, king: their step sets.
- Bishop, rook, queen: along each ray, every empty square and **the first occupied square, inclusive**, whatever its colour. The ray stops there. **No x-rays.**
- Squares holding friendly pieces are included. Pull on your own piece is support (defence).
- A piece never pulls on its own square.

**Relocation set** `Mov(p, occ)`, used only for multi-move travel (L3): the **empty** squares `p` could move to under the rules, with all other pieces frozen.
- Knight, bishop, rook, queen, king: `Att(p, occ) ∩ ¬occ`.
- Pawn: the single push square if empty, plus the double push square if on its start rank and both squares are empty.
- Captures, castling and en passant are not relocations, because they would change the board.
- Promotion relocations count, with the piece still treated as a pawn (no type change inside the field computation). In practice this adds nothing, since a pawn on the last rank attacks no square.
- Relocations are pseudo-legal: pins and checks are ignored. The field describes reach under the movement rules, not legality in this exact position. (spec-1 clarification, 2026-10-24)

**Hop distance** `h_p(s)` for `s ≠ s_p`:
- `h = 1` if `s ∈ Att(p, occ)`.
- `h = 2` if not hop 1, and there is some `r ∈ Mov(p, occ)` with `s ∈ Att(p at r, occ')`, where `occ' = occ` with `s_p` cleared and `r` set.
- `h = 3`: the same again from hop-2 relocations (only when `c = 3`). Relocations from a relocation are computed with all other pieces frozen.
- Otherwise `h = ∞`.

One hop means **one move**: a rook covers its whole open ray in one hop, because it moves the whole ray in one move. Distance is counted in moves, never in squares.

**Shell size** `n_h(p)` = the number of squares at hop distance exactly `h` from `p`.

**Emission mass** `E_p = M_p` for non-kings. For kings, `E_K = kappa + A_K`.

**Field of one piece:**
```
φ_p(s) = E_p · alpha^(h_p(s) − 1) / n_{h_p(s)}(p)^sigma      if h_p(s) ≤ c
φ_p(s) = 0                                                    otherwise
```
`sigma = 1` means a piece's total pull per shell is conserved and spreads thin over many squares (the discrete form of Gauss's law). `sigma = 0` means every reached square gets the full pull. Training chooses.

**Side field:** `Φ_C(s) = Σ_{p ∈ C} φ_p(s)`. This is the "pull" drawn on the board.

---

## 4. L3 · Light speed

Pull from piece `p` reaches square `s` after `h_p(s)` moves and is attenuated by `alpha` per extra move. It arrives only if `h_p(s) ≤ c`. This is a retarded potential on the move graph: influence takes moves to arrive, and `c` is literally the light-cone depth in moves.

**There is no other speed of light.** v0's velocity accumulator and Lorentz mass boost are removed.

---

## 5. L4 · Energy of bodies in the field

**L4a · Binding energy.** For every non-king piece `q` of colour `C` on square `s`:
```
e_q = M_q · ( Φ_{¬C}(s) − Φ_C(s) )
```
Positive `e_q` means the body is unbound: more enemy pull on it than friendly support. It's exposed (hanging, under-defended). Negative means it's held.
```
U_C    = Σ_{q ∈ C, t≠K} e_q
T_bind = lambda_bind · ( U_B − U_W )
```

**L4b · Held space.**
```
T_space = lambda_space · Σ_{s=0}^{63} tanh( Φ_W(s) − Φ_B(s) )
```
Each square belongs, softly, to whoever pulls on it harder. The `tanh` scale is fixed at 1 ku.

Note for reviewers: L4a is a static quantity. It doesn't price the order of captures (that's the search's job: quiescence resolves exchanges). It answers "how exposed is this arrangement", nothing more.

---

## 6. L5 · The Roche limit (when a king is torn apart)

**Body** of king `K` of colour `C` on square `k`: `S_K = {k} ∪ {the up-to-8 squares a king steps to from k, on the board}`. The body includes squares occupied by any piece. On an edge it has 6 squares, in a corner 4.

```
stress   τ_K = Σ_{s ∈ S_K} Φ_{¬C}(s)              enemy pull across the body
binding  b_K = Σ_{s ∈ S_K} Φ_C(s) + 1e−3           own side's pull on the body (the king's own emission included)
index    η_K = τ_K / b_K
gauge    g_K = sigmoid( k_roche · (η_K − rho) )    0 = intact, 1 = torn apart
T_roche  = beta · ( g_{K_B} − g_{K_W} )
```
In plain words: a king breaks when the enemy pulls across its body harder than its own side holds it, by more than the learned Roche limit `rho`.

- The king square is part of the body, so check raises stress directly.
- Checkmate is the rule-defined extreme case, and it's handled by search (§9), not by the gauge.
- Public UI shows `η_K / rho` as a percentage, where 100% means the breaking point.

---

## 7. L6 · Accretion

When a capture happens (including en passant), the captor absorbs a fraction of the victim's carried mass:
```
A_captor ← A_captor + f_acc · M_victim
```
- **Promotion:** the new piece keeps the pawn's `A` (plus any capture share). Its rest mass becomes that of the new type.
- **Castling:** the rook keeps its `A`.
- **King captures:** `A_K` grows. It raises the king's emission `E_K` only, never material.
- **State:** `A` lives in the game state, not in FEN. A position given only by FEN has `A = 0` everywhere. UCI `position ... moves ...` replays the moves, so `A` is exact in real games.
- **Hashing:** the transposition-table key covers piece placement only. `A` is evaluation state. Transpositions with different `A` are accepted as an approximation and documented in `03_ENGINE_ARCHITECTURE.md` §4.

**Differentiation:** when gradients are taken (§1, column A), the accreted masses `A` of a position are treated as frozen state, not as a function of the constants. TDLeaf trains on stored positions whose history is fixed, so this is the consistent choice. (spec-1 clarification, 2026-10-24)

`f_acc = 0` turns accretion off. If training drives it there, the Law Ledger reports "the universe rejected accretion".

---

## 8. Total evaluation and calibration

```
score_white = T_mat + T_bind + T_space + T_roche
score_stm   = score_white if White to move, else −score_white
```

**Calibration of Gen 0** (done once, in milestone 1, and recorded in the decision log): set `lambda_bind`, `lambda_space` and `beta` so that, over 300 seeded random-play positions (generated by the calibration tool, xorshift seed 12345, accretion carried; every legal child scored):
- the standard deviation across sibling moves of `T_bind + T_space + T_roche` is **≤ 0.5 ku**;
- its maximum is **≤ 2.5 ku**, except where a king's `η > rho`;
- a free minor piece (≥ 3 ku of material) therefore always outweighs positional terms in quiet positions.

Training may move these constants afterwards. That's its job.

---

## 9. Rules the search applies (not laws)

These are chess rules and are never scored by physics: legal moves, checkmate (score `±(MATE − ply)`), stalemate, 50-move rule, threefold repetition (draw, with the full game history from UCI), and insufficient material. The search may use standard, physics-neutral techniques (`03_ENGINE_ARCHITECTURE.md` §5).

**Forbidden in the evaluation:** piece-square tables, pawn-structure rules, opening books, endgame tablebases, mobility bonuses outside L2–L4, king-safety rules outside L5, and any hand-written chess heuristic. If a strength idea can't be expressed as a quantity from §2–§7, it's either a search technique or it's not allowed.

---

## 10. What the evaluator must expose

Every evaluation can return a full breakdown. It's used by tests, the trainer and the website, with the same numbers everywhere:
- `terms`: `T_mat`, `T_bind`, `T_space`, `T_roche`, `score_white`.
- `field`: `Φ_W[64]`, `Φ_B[64]`.
- Per piece: square, type, colour, `μ`, `A`, `M`, `e_q`, and the hop-1/2/3 bitboards (its light cone, for drawing).
- Per king: body squares, `τ`, `b`, `η`, `g`, and `η / rho`.
- `constants`: the 14 values and the generation id.

---

## 11. Required invariants (each is a unit test)

1. **Colour symmetry:** mirroring the board (swap colours, flip ranks, swap side to move) negates `score_white` exactly (within 1e-5).
2. **File symmetry:** mirroring files with no castling rights leaves the score unchanged.
3. **Material conservation:** the material sum over a capture-recapture sequence equals standard material arithmetic with learned `μ`, whatever `f_acc` is. Regression: the French line `Qd3 c4 Qxc4 dxc4 Bxc4` must give `T_mat` change ≈ `−(m_q + m_b) + 2·1 + m_b`, i.e. a queen for two pawns, negative for White.
4. **No x-ray:** a rook behind its own pawn on a file adds no hop-1 pull beyond the pawn.
5. **Blocking:** closing a file with a pawn strictly reduces the rook's hop-1 set.
6. **Locality:** pull from a piece with `h > c` is exactly 0.
7. **Gradient check:** for every A-tuned constant and **each** of 200 positions, the analytic derivative matches a central finite difference within `1e-3 · (1 + |g|)` (accreted masses frozen, see §7).
8. **Breakdown sums:** `T_mat + T_bind + T_space + T_roche == score_white` bit-for-bit in f32.

---

## 12. Versioning

- Any change to a formula in §2–§7 bumps the spec id (`v1.0-spec-N`), is logged in `07_DECISION_LOG.md`, and invalidates previously learned generations until the Law Ledger is rerun.
- Changing only constant values is a new generation, not a new spec.
