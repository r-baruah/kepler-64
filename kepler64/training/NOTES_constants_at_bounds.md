# Why trained constants end up pinned at their bounds

Status: investigation notes from 2026-10-14. No training was run; the evidence
is code reading, the September Kaggle log, and a small gradient probe.

## What was observed

The sovereign self-play run (`notebooks/kaggle_gate.ipynb`, the "198-2" run):
680 examples, 800 Adam steps, lr 3e-3, tau 2.0. The trained leaves ended at
or near their bounds:

| leaf | base | trained | bound |
|---|---|---|---|
| roche | 1.00 | 0.05 | LO |
| kgain | 4.00 | 0.01 | LO |
| Rg | 1.00 | 0.10 | LO |
| eps | 0.50 | 0.80 | HI |
| entropy_gain | 4.00 | 6.00 | HI |
| dt_drift | 0.10 | 0.01 | LO |
| com_gain | 1.00 | 0.00 | LO |

The training loss fell from ~90–125 to ~3 between steps 250 and 350. Held-out
MRR moved from 0.451 to 0.460. About 3 is roughly ln(number of legal moves),
the cross-entropy of a uniform guess.

The shipped `kepler64/training/trained_constants.json` (round 3, Aug 20) is
**not** this degenerate set: none of the 14 leaves it stores is pinned at the
bound that collapsed the gauge (roche, kgain, Rg, eps, entropy_gain are all
interior). Two caveats, though:

- `gamma` is stored as 0.0, which **is** its lower bound (`LEAF_LO` gamma =
  0.0), so it is pinned, not interior. It has never left its starting value
  of 0.0 in that run.
- The JSON predates three leaves and omits `lambda_gw`, `lambda_sch` and
  `dt_drift`. `load_constants` fills them from the defaults (0.0, 0.0 and
  0.1). `lambda_gw` and `lambda_sch` default to 0.0, which is also their lower
  bound, so those two terms are simply switched off in the shipped artifact;
  `dt_drift` was never trained there. The shipped set is therefore a 14-leaf
  result, not a 17-leaf one.

The degenerate set above exists only in the Kaggle run's output.

## Gradient probe

18 positions from a part-random game, with the teacher being the default
engine at depth 2. This is the same policy loss and tau = 2 the run used.

| constants | policy loss |
|---|---|
| uniform guess | 3.36 |
| default | 262.8 |
| Kaggle-degenerate | 6.9 |

The gradient at the default constants points every pinned leaf toward its
observed bound: roche down, kgain down, Rg down, mat_gain down (stopped by
the 1.5 floor), eps up, dt_drift.

## Mechanism

1. **The disruption gauge dominates the score spread.**
   `bonus * sigmoid(kgain * (|F| - roche))` with bonus = 300 is close to a
   ±300 step across sibling moves. Whichever child tips the sigmoid gets a
   score hundreds of units above its siblings. With tau = 2, the softmax puts
   almost all its mass on that child.
2. **The teacher disagrees with those spikes.** The teacher is the same
   physics searched deeper. A one-ply gauge spike, such as a piece landing
   next to the king, is usually refuted a ply later. So the
   confidently-preferred child is often not the expert move, and the
   cross-entropy is huge (262 here).
3. **The cheapest way down is to switch the gauge off.**
   - kgain → 0.01 makes the sigmoid flat at about 0.5. The two gauge terms
     then cancel.
   - roche → 0.05 pushes it further toward saturation.
   - Rg → 0.1 shrinks eta by a factor of (0.1)³.
   - eps → 0.8 blurs the field.
   - Moving loss from 262 to about 7 needs no improvement in move ranking,
     which matches MRR staying flat.
4. **Adam moves every leaf at about lr per step, regardless of the leaf's
   scale.** In 800 steps a leaf can travel about 2.4 units. That is the whole
   range of kgain, roche and Rg, but under 1% of bonus (300). So the optimizer
   can only remove the gauge through its small-scale knobs.

### Scale is not the whole story

`loss(..., normalize_scores=True)` (added as an opt-in diagnostic)
standardizes each position's child scores, so only the ORDER matters. On the
same probe the loss was 7.63 at the default constants and 6.27 at the
degenerate constants. The gradient still points roche, kgain and Rg down.
On this data, the one-ply disruption gauge ranks the teacher's moves worse
than having no gauge at all. That is a finding about the physics as built,
not just about the optimizer. It matters for the narrative: the training
signal is saying the "Roche collapse" term, evaluated one ply ahead, does
not predict what deeper search wants.

## Proposed fixes, in order of risk

1. **Gate acceptance on held-out ranking, never on training loss.**
   `scripts/train.py` already accepts a round only if validation MRR does not
   drop; the notebook path should do the same. The degenerate set would have
   been accepted on +0.009 MRR, so also require a minimum gain and a
   head-to-head check (`scripts/benchmarks.py learned-vs-frozen`).
2. **Log-space parametrization for positive leaves.** Then Adam steps are
   relative: bonus and kgain move at the same rate.
3. **Anchor prior.** A small L2 penalty in log space toward the hand-set
   universe, like the Observer's KL anchor, so a leaf leaves its prior only
   when the data pays for it.
4. **Bounds that keep each mechanism alive.** kgain ≥ ~0.5 and Rg ≥ ~0.5,
   like the existing "Mass Law" floor on mat_gain. Without them the
   optimizer can delete the mechanism the project is named after.
   **This is a product decision, not a bug fix, so it is not implemented
   here.**
5. **Fix the physics, not the optimizer.** If the scaled run still turns the
   gauge off, the honest move is to rework the gauge: for example, apply it
   only at quiescent leaves, or feed it eta rather than |F|. Training the
   gauge away and still advertising it as the win condition is not honest.

## What the frontend and narrative should take from this

- In code, the learned "Roche limit" gates the enemy army's **pull** (|F|) at
  the king through a sigmoid. It does not gate eta. eta enters the score
  linearly. The README's "η > ρ_roche ⇒ disruption" is a simplification.
- The "198-2" result: every one of the 200 games stopped at the 60-ply cap
  and was adjudicated by army mass. The engines were deterministic and played
  from the start position, so the 200 games were very likely a handful of
  distinct games repeated. `scripts/benchmarks.py` fixes both problems for
  the pre-launch rerun.
