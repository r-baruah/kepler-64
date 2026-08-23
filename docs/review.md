Now I have a full picture. Let me analyze the backend and identify high-impact improvements for chess understanding and gameplay.

---

## Kepler-64 Backend Analysis: High-Impact Opportunities

### Current Architecture Summary

| Layer | What It Does | Key Files |
|-------|--------------|-----------|
| **Physics Core** | Plummer gravity → tidal tensor (Hessian) → Roche η → score | `gravity.py`, `tidal.py`, `evaluate.py` |
| **Board/State** | `FastBoard` (pure NumPy legal moves, velocity for Lorentz mass) | `fastboard.py`, `transitions.py` |
| **Search** | Alpha-beta + PVS, 218-pad vmap batch eval, TT, null-move, quiescence | `minimax.py` |
| **Training** | Outcome + policy loss, backprops through entire physics pipeline | `loss.py`, `train.py` |
| **Layer 2 (Multiverse)** | Bayesian model average over (G,ε,c) posterior | `evaluate.py`, `multiverse/*.py` |

---

### Critical Weaknesses (Directly Hurting Gameplay)

#### 1. **Material Dominates — Physics Is Cosmetic** (Chapter 14, Item 14)
```python
# evaluate.py:197
material = mat_gain * (jnp.sum(white_m) - jnp.sum(black_m))  # mat_gain=1.0
```
- Material range: ±20 (all pieces) → score swing ~±20
- Tidal η range: ~0.05–1.0 → score swing ~±1
- **Result**: Engine plays "material hoarding" not "gravity chess." The thesis (gravity as primary evaluator) is undermined.

#### 2. **No Positional Chess Understanding**
The physics evaluates **only King tidal stress + global binding energy**. Missing entirely:
- Pawn structure (doubled, isolated, passed, backward)
- Piece activity (knights on outposts, bishops vs pawns, rooks on open files)
- King safety beyond tidal (pawn shelter, attack zones)
- Center control, space, tempo
- Endgame knowledge (king activity, opposition, Lucena/Philidor)

#### 3. **Shallow Search (Depth 4–5 Effective)**
- No **LMR (Late Move Reductions)** — searches all moves equally
- No **singular extensions** — misses forced lines
- No **aspiration windows** — full-window at every node
- Quiescence: **no checks**, only captures + evasions → horizon blindness to forcing sequences
- Time management: simple budget, no move-overhead allocation

#### 4. **Training Data Too Small & Uncurated** (C9)
- 378 examples → `roche→0.05` (floor), `gamma→0.0`, accuracy **drops**
- No curriculum: outcome → policy → multiverse → self-play
- Ablation gate (C1) not run/published — credibility gap

#### 5. **Physics Bugs That Corrupt Signal**
| Bug | File | Impact |
|-----|------|--------|
| Silent King fallback to a1 | `evaluate.py:44` | Wrong η at edge/corrupted boards |
| King mass match `int==1000` fails under accretion | `evaluate.py:43` | King becomes unfindable |
| Accretion doesn't update `Rg` | `transitions.py` | Overextended pieces don't get tidally fragile |
| Lorentz mass unwired from eval | `lorentz.py` vs `evaluate.py` | Anti-repetition signal dead |
| Verlet rollout not in hot path | `verlet.py` | `dη/dt` (impending collapse) missing |

---

### Highest-Impact Improvements (Ranked)

#### 🥇 **P0: Fix Physics Bugs + Rebalance Material vs Gravity**
**Effort**: ~1 day | **Impact**: Restores thesis credibility, makes gravity the primary signal

```python
# evaluate.py — reduce mat_gain, boost physics terms
mat_gain: float = 0.15      # was 1.0 — material is a TIEBREAKER, not the signal
lambda_delta: float = 2.0   # was 1.0 — Δη is the move-sensitivity heart
com_gain: float = 40.0      # was 25.0 — center-of-mass advance matters more
inertia_gain: float = 12.0  # was 8.0 — attack concentration matters more
gamma: float = 0.5          # was 0.25 — binding energy (cohesion) stronger
```

**Also fix the bugs:**
- `_king_idx`: use `atol=5.0` and track King squares explicitly in `FastBoard`
- `transitions.py`: `Rg_new = Rg_old * (m_new/m_old)**(1/3)` after accretion
- Wire `velocity` → Lorentz mass into `evaluate.py` (replace `Rg` with dynamic per-piece extent)

---

#### 🥈 **P1: Add Chess-Native Positional Features (Physics-Compatible)**
**Effort**: ~2–3 days | **Impact**: Massive Elo gain; keeps physics-first identity

The insight: **chess concepts map to physics fields** — don't add heuristics, add *field-derived observables* that the physics already computes.

| Chess Concept | Physics Observable (already computable) | Implementation |
|---------------|----------------------------------------|----------------|
| **Pawn structure** | `∇²Φ` along file → pawn chain cohesion | Add `pawn_chain_gain * Σ |∂²U/∂file²|` on pawn squares |
| **Weak squares** | Low `|∇Φ|` + low `λ₁` in enemy territory | `weak_sq_gain * Σ enemy_empty_sq * sigmoid(-|F|)` |
| **Outposts** | Knight on square with high friendly `|F|`, low enemy `|F|` | `outpost_gain * Σ knight_sq * (|F_friendly| - |F_enemy|)` |
| **Open files** | Ray cast along file: no pawns → high `|F_rook|` propagation | `open_file_gain * Σ rook_sq * file_clearance` |
| **King shelter** | Pawn shield = mass in front of King → binds King's well | `shelter_gain * Σ pawn_mass * sigmoid(king_dist - d)` |

**Implementation pattern** (add to `_score_terms_body`):
```python
# All derived from U_w, U_b, F_w, F_b — NO new field passes
pawn_chain = pawn_chain_gain * jnp.sum(pawn_mask * jnp.abs(u_file))
weak_squares = weak_sq_gain * jnp.sum((1 - enemy_pawn_mask) * jax.nn.sigmoid(-jnp.linalg.norm(F_b, axis=1)))
# ... etc
```

**Why this preserves the thesis**: Every term is a *reading of the gravitational field*, not a hand-crafted chess heuristic. The physics "discovers" positional concepts because they're literally in the field.

---

#### 🥉 **P2: Search Upgrades for 200–300 Elo**
**Effort**: ~2 days | **Impact**: Deeper effective search, tactical sharpness**

| Upgrade | Code Location | Key Change |
|---------|---------------|------------|
| **LMR** | `minimax.py:258` | `if not first and not capture and depth >= 3: depth -= 1 + (move_idx // 4)` |
| **Singular Extensions** | `negamax` | If TT move refutes with huge margin, extend 1 ply |
| **Aspiration Windows** | `iterative_search` | Start with `alpha = prev_score - 50, beta = prev_score + 50` |
| **Quiescence Checks** | `_quiesce:338` | Add checking moves to evasion set (not just captures) |
| **SEE Pruning** | `_ordered_moves` | Static Exchange Evaluation for capture ordering |

---

#### 🏅 **P3: Training Pipeline Overhaul**
**Effort**: ~3 days | **Impact**: Constants that actually work, publishable ablation**

1. **Data**: 10k+ positions (5k puzzles + 5k GM games), 80/20 split
2. **Curriculum**:
   - Phase 1: Outcome only (learn `eps`, `c`, `roche`)
   - Phase 2: Policy loss (learn `bonus`, `kgain`, `gamma`, `mat_gain`)
   - Phase 3: Delta terms (learn `lambda_delta`, `com_gain`, `inertia_gain`, `entropy_gain`)
   - Phase 4: Multiverse posterior (learn `K`, `sigma`)
   - Phase 5: Self-play (RL fine-tune)
3. **Run & publish ablation** (`training/ablation.py`) — this is the credibility gate

---

#### 🎯 **P4: Wire the "Cool Physics" That's Already Built But Dead**
| Feature | File | Status | Fix |
|---------|------|--------|-----|
| **Verlet rollout → `dη/dt`** | `verlet.py` | Documented, not in eval | Add `delta_eta_rollout = rollout_eta(child) - rollout_eta(parent)` |
| **Lorentz mass (anti-repetition)** | `lorentz.py`, `fastboard.py` | Velocity tracked, not used in eval | Pass `velocity` to `evaluate.py`, compute per-piece `γ` |
| **Multiverse at root** | `evaluate.py:395` | Implemented, opt-in | Enable by default for `depth >= 3` |
| **Image seeding (diversity)** | `image_seed.py` | Broken (FFT of JPEG bytes) | Decode with Pillow → 8×8 → FFT → seed `G, eps` |

---

### One-Week Sprint Plan (Max Gameplay Impact)

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | **P0 bugs + rebalance** | Clean eval, King detection fixed, `mat_gain=0.15`, physics terms dominant |
| 2 | **P1 positional fields** | 4–5 field-derived terms added, tested on tactical suites |
| 3 | **P2 LMR + aspiration** | Search reaches depth 7–8 in same time, tactical test pass |
| 4 | **P3 data + curriculum** | 10k positions, 5-phase train script, ablation table in README |
| 5 | **P4 wire cool physics** | Verlet `dη/dt`, Lorentz mass, Multiverse default-on |
| 6 | **Integration + self-play** | 100 games vs Stockfish 1300, measure Elo delta |
| 7 | **Polish + docs** | Update README with ablation, physics terms explained |

---

### What NOT to Do (Traps)

| Trap | Why It Fails |
|------|--------------|
| Add Stockfish-style heuristics (PST, mobility, king safety tables) | Destroys the "physics-first" thesis; becomes a weird hybrid |
| Train a neural net policy on top | Same — the differentiator is *inspectable physics*, not black-box strength |
| Chase Elo via deeper search alone | Diminishing returns without better eval; depth 6 with bad eval = bad moves faster |
| Overfit constants on tiny data | C9: `roche→0.05`, `gamma→0.0` — the engine unlearns chess |

---

### Bottom Line

**The physics is genuinely novel and the differentiable pipeline is sound.** The engine loses because:
1. **Material drowns the physics signal** (easy fix: rebalance gains)
2. **No positional vocabulary** (fix: read the field you already compute)
3. **Search is shallow + blind to forcing lines** (fix: LMR, aspiration, quiescence checks)
4. **Training is too small to learn meaningful constants** (fix: 10k+ positions, curriculum)

Fix P0 + P1 + P2 and you get a **physics engine that plays credible positional chess** — the "romantic chess from first principles" becomes real, not anecdotal. The ablation table then proves the gravity isn't cosmetic.

Want me to implement any specific P0/P1 fix first?

Now I have a full picture. Let me analyze the backend and identify high-impact improvements for chess understanding and gameplay.

---

## Kepler-64 Backend Analysis: High-Impact Opportunities

### Current Architecture Summary

| Layer | What It Does | Key Files |
|-------|--------------|-----------|
| **Physics Core** | Plummer gravity → tidal tensor (Hessian) → Roche η → score | `gravity.py`, `tidal.py`, `evaluate.py` |
| **Board/State** | `FastBoard` (pure NumPy legal moves, velocity for Lorentz mass) | `fastboard.py`, `transitions.py` |
| **Search** | Alpha-beta + PVS, 218-pad vmap batch eval, TT, null-move, quiescence | `minimax.py` |
| **Training** | Outcome + policy loss, backprops through entire physics pipeline | `loss.py`, `train.py` |
| **Layer 2 (Multiverse)** | Bayesian model average over (G,ε,c) posterior | `evaluate.py`, `multiverse/*.py` |

---

### Critical Weaknesses (Directly Hurting Gameplay)

#### 1. **Material Dominates — Physics Is Cosmetic** (Chapter 14, Item 14)
```python
# evaluate.py:197
material = mat_gain * (jnp.sum(white_m) - jnp.sum(black_m))  # mat_gain=1.0
```
- Material range: ±20 (all pieces) → score swing ~±20
- Tidal η range: ~0.05–1.0 → score swing ~±1
- **Result**: Engine plays "material hoarding" not "gravity chess." The thesis (gravity as primary evaluator) is undermined.

#### 2. **No Positional Chess Understanding**
The physics evaluates **only King tidal stress + global binding energy**. Missing entirely:
- Pawn structure (doubled, isolated, passed, backward)
- Piece activity (knights on outposts, bishops vs pawns, rooks on open files)
- King safety beyond tidal (pawn shelter, attack zones)
- Center control, space, tempo
- Endgame knowledge (king activity, opposition, Lucena/Philidor)

#### 3. **Shallow Search (Depth 4–5 Effective)**
- No **LMR (Late Move Reductions)** — searches all moves equally
- No **singular extensions** — misses forced lines
- No **aspiration windows** — full-window at every node
- Quiescence: **no checks**, only captures + evasions → horizon blindness to forcing sequences
- Time management: simple budget, no move-overhead allocation

#### 4. **Training Data Too Small & Uncurated** (C9)
- 378 examples → `roche→0.05` (floor), `gamma→0.0`, accuracy **drops**
- No curriculum: outcome → policy → multiverse → self-play
- Ablation gate (C1) not run/published — credibility gap

#### 5. **Physics Bugs That Corrupt Signal**
| Bug | File | Impact |
|-----|------|--------|
| Silent King fallback to a1 | `evaluate.py:44` | Wrong η at edge/corrupted boards |
| King mass match `int==1000` fails under accretion | `evaluate.py:43` | King becomes unfindable |
| Accretion doesn't update `Rg` | `transitions.py` | Overextended pieces don't get tidally fragile |
| Lorentz mass unwired from eval | `lorentz.py` vs `evaluate.py` | Anti-repetition signal dead |
| Verlet rollout not in hot path | `verlet.py` | `dη/dt` (impending collapse) missing |

---

### Highest-Impact Improvements (Ranked)

#### 🥇 **P0: Fix Physics Bugs + Rebalance Material vs Gravity**
**Effort**: ~1 day | **Impact**: Restores thesis credibility, makes gravity the primary signal

```python
# evaluate.py — reduce mat_gain, boost physics terms
mat_gain: float = 0.15      # was 1.0 — material is a TIEBREAKER, not the signal
lambda_delta: float = 2.0   # was 1.0 — Δη is the move-sensitivity heart
com_gain: float = 40.0      # was 25.0 — center-of-mass advance matters more
inertia_gain: float = 12.0  # was 8.0 — attack concentration matters more
gamma: float = 0.5          # was 0.25 — binding energy (cohesion) stronger
```

**Also fix the bugs:**
- `_king_idx`: use `atol=5.0` and track King squares explicitly in `FastBoard`
- `transitions.py`: `Rg_new = Rg_old * (m_new/m_old)**(1/3)` after accretion
- Wire `velocity` → Lorentz mass into `evaluate.py` (replace `Rg` with dynamic per-piece extent)

---

#### 🥈 **P1: Add Chess-Native Positional Features (Physics-Compatible)**
**Effort**: ~2–3 days | **Impact**: Massive Elo gain; keeps physics-first identity

The insight: **chess concepts map to physics fields** — don't add heuristics, add *field-derived observables* that the physics already computes.

| Chess Concept | Physics Observable (already computable) | Implementation |
|---------------|----------------------------------------|----------------|
| **Pawn structure** | `∇²Φ` along file → pawn chain cohesion | Add `pawn_chain_gain * Σ |∂²U/∂file²|` on pawn squares |
| **Weak squares** | Low `|∇Φ|` + low `λ₁` in enemy territory | `weak_sq_gain * Σ enemy_empty_sq * sigmoid(-|F|)` |
| **Outposts** | Knight on square with high friendly `|F|`, low enemy `|F|` | `outpost_gain * Σ knight_sq * (|F_friendly| - |F_enemy|)` |
| **Open files** | Ray cast along file: no pawns → high `|F_rook|` propagation | `open_file_gain * Σ rook_sq * file_clearance` |
| **King shelter** | Pawn shield = mass in front of King → binds King's well | `shelter_gain * Σ pawn_mass * sigmoid(king_dist - d)` |

**Implementation pattern** (add to `_score_terms_body`):
```python
# All derived from U_w, U_b, F_w, F_b — NO new field passes
pawn_chain = pawn_chain_gain * jnp.sum(pawn_mask * jnp.abs(u_file))
weak_squares = weak_sq_gain * jnp.sum((1 - enemy_pawn_mask) * jax.nn.sigmoid(-jnp.linalg.norm(F_b, axis=1)))
# ... etc
```

**Why this preserves the thesis**: Every term is a *reading of the gravitational field*, not a hand-crafted chess heuristic. The physics "discovers" positional concepts because they're literally in the field.

---

#### 🥉 **P2: Search Upgrades for 200–300 Elo**
**Effort**: ~2 days | **Impact**: Deeper effective search, tactical sharpness**

| Upgrade | Code Location | Key Change |
|---------|---------------|------------|
| **LMR** | `minimax.py:258` | `if not first and not capture and depth >= 3: depth -= 1 + (move_idx // 4)` |
| **Singular Extensions** | `negamax` | If TT move refutes with huge margin, extend 1 ply |
| **Aspiration Windows** | `iterative_search` | Start with `alpha = prev_score - 50, beta = prev_score + 50` |
| **Quiescence Checks** | `_quiesce:338` | Add checking moves to evasion set (not just captures) |
| **SEE Pruning** | `_ordered_moves` | Static Exchange Evaluation for capture ordering |

---

#### 🏅 **P3: Training Pipeline Overhaul**
**Effort**: ~3 days | **Impact**: Constants that actually work, publishable ablation**

1. **Data**: 10k+ positions (5k puzzles + 5k GM games), 80/20 split
2. **Curriculum**:
   - Phase 1: Outcome only (learn `eps`, `c`, `roche`)
   - Phase 2: Policy loss (learn `bonus`, `kgain`, `gamma`, `mat_gain`)
   - Phase 3: Delta terms (learn `lambda_delta`, `com_gain`, `inertia_gain`, `entropy_gain`)
   - Phase 4: Multiverse posterior (learn `K`, `sigma`)
   - Phase 5: Self-play (RL fine-tune)
3. **Run & publish ablation** (`training/ablation.py`) — this is the credibility gate

---

#### 🎯 **P4: Wire the "Cool Physics" That's Already Built But Dead**
| Feature | File | Status | Fix |
|---------|------|--------|-----|
| **Verlet rollout → `dη/dt`** | `verlet.py` | Documented, not in eval | Add `delta_eta_rollout = rollout_eta(child) - rollout_eta(parent)` |
| **Lorentz mass (anti-repetition)** | `lorentz.py`, `fastboard.py` | Velocity tracked, not used in eval | Pass `velocity` to `evaluate.py`, compute per-piece `γ` |
| **Multiverse at root** | `evaluate.py:395` | Implemented, opt-in | Enable by default for `depth >= 3` |
| **Image seeding (diversity)** | `image_seed.py` | Broken (FFT of JPEG bytes) | Decode with Pillow → 8×8 → FFT → seed `G, eps` |

---

### One-Week Sprint Plan (Max Gameplay Impact)

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | **P0 bugs + rebalance** | Clean eval, King detection fixed, `mat_gain=0.15`, physics terms dominant |
| 2 | **P1 positional fields** | 4–5 field-derived terms added, tested on tactical suites |
| 3 | **P2 LMR + aspiration** | Search reaches depth 7–8 in same time, tactical test pass |
| 4 | **P3 data + curriculum** | 10k positions, 5-phase train script, ablation table in README |
| 5 | **P4 wire cool physics** | Verlet `dη/dt`, Lorentz mass, Multiverse default-on |
| 6 | **Integration + self-play** | 100 games vs Stockfish 1300, measure Elo delta |
| 7 | **Polish + docs** | Update README with ablation, physics terms explained |

---

### What NOT to Do (Traps)

| Trap | Why It Fails |
|------|--------------|
| Add Stockfish-style heuristics (PST, mobility, king safety tables) | Destroys the "physics-first" thesis; becomes a weird hybrid |
| Train a neural net policy on top | Same — the differentiator is *inspectable physics*, not black-box strength |
| Chase Elo via deeper search alone | Diminishing returns without better eval; depth 6 with bad eval = bad moves faster |
| Overfit constants on tiny data | C9: `roche→0.05`, `gamma→0.0` — the engine unlearns chess |

---

### Bottom Line

**The physics is genuinely novel and the differentiable pipeline is sound.** The engine loses because:
1. **Material drowns the physics signal** (easy fix: rebalance gains)
2. **No positional vocabulary** (fix: read the field you already compute)
3. **Search is shallow + blind to forcing lines** (fix: LMR, aspiration, quiescence checks)
4. **Training is too small to learn meaningful constants** (fix: 10k+ positions, curriculum)

Fix P0 + P1 + P2 and you get a **physics engine that plays credible positional chess** — the "romantic chess from first principles" becomes real, not anecdotal. The ablation table then proves the gravity isn't cosmetic.

Want me to implement any specific P0/P1 fix first?