# Chapter 4 — The Roche / Hill Disruption Parameter η

## 4.1 Intuition: when does a moon break apart?

Around a massive planet (say Jupiter) there is a distance within which a moon cannot hold itself together: Jupiter's tidal stretching overcomes the moon's own gravity. Cross that distance and the moon is pulled into a ring. That distance is the **Roche limit.** A related idea is the **Hill sphere** — the region around a body where its own gravity dominates over the tidal pull of a neighbor.

In Kepler-64 we borrow this *scaling* (not the exact astronomy) to decide when a King's defensive formation "structurally collapses." The King has a binding strength (its own self-gravity, proportional to its mass squared). The enemy has a tearing strength (the tidal eigenvalue $\lambda_1$). When *tearing ÷ binding* exceeds a threshold, the King disrupts.

> **Intuition box:** A snowman in a strong wind: its own cohesion holds it together, but past a certain wind gradient it shears apart. η is "wind-gradient divided by snowman-cohesion." Above 1, it breaks.

## 4.2 The math: the dimensionless disruption parameter

The textbook Roche/Hill scaling says the disruption parameter scales as

$$\eta \;\propto\; \frac{\text{tidal stretch} \times (\text{size})^3}{\text{self-binding}} \;\sim\; \frac{\lambda_1\,R_g^3}{G\,M_{\text{king}}^2}$$

where:

- $\lambda_1$ — principal tidal eigenvalue (Chapter 3), the tearing rate.
- $R_g$ — **radius of gyration**, the King's effective spatial extent (base 1.0 lattice unit; now scaled as $(m/1000)^{1/3}$ under accretion, see §4.5).
- $M_{\text{king}}$ — the King's mass (1000).
- $G$ — gravitational constant.

$\eta$ is **dimensionless** (a pure number), which is why it is the right thing to threshold.

## 4.3 The math as implemented: the $M_{\text{king}}^2$ decision

Here is a subtle but important modeling choice in the actual code (`core/evaluate.py:41-58`, `core/tidal.py:47-60`). The implemented η is

$$\eta = \frac{R_g^3\,\lambda_1}{m_{\text{ref}}^2}, \qquad m_{\text{ref}} = 3.5$$

**$G$ is deliberately absent**, and the denominator is $m_{\text{ref}}^2$ (a fixed minor-piece-scale constant), **not** $M_{\text{king}}^2$.

Two reasons, both documented in `core/evaluate.py:46-52`:

1. **$G$-independence.** Because $\lambda_1$ is itself proportional to $G$ (it is the Hessian of a potential that scales with $G$), putting $G$ in the denominator would cancel it — making η independent of field strength. Physically the disruption *criterion* should be $G$-independent, so the cancellation is correct.
2. **King-mass blow-up.** If we divided by $M_{\text{king}}^2 = 1000^2 = 10^6$, η would be ~$10^{-6}$ everywhere and the Roche limit would *never* be reachable. Using a fixed reference scale $m_{\text{ref}} \approx$ a minor piece keeps η in a usable range $[0.05\ \text{quiet} \;\ldots\; 1+\ \text{under attack}]$.

> **Intuition box:** We don't compare the King to *itself* (that always looks safe); we compare the tear to a *fixed reference scale*. That's what makes "under attack" actually register as a large η.

**Worked example.** At the King from §3.3 we had $\lambda_1 = 3.618$, with $R_g=1$, $m_{\text{ref}}=3.5$:

$$\eta = \frac{1^3 \cdot 3.618}{3.5^2} = \frac{3.618}{12.25} = 0.295$$

Quiet-ish (below the learned Roche threshold, typically ~0.05–1.0). If the enemy piles two more queens nearby, $\lambda_1$ might rise to 15:

$$\eta = \frac{15}{12.25} = 1.22 > \rho_{\text{roche}}$$

→ disruption. This is exactly how the engine "sees" a winning attack.

## 4.4 The learned Roche threshold $\rho_{\text{roche}}$

The crossover $\eta = \rho_{\text{roche}}$ is **learned**, initialized at 1.0 so learning is physically interpretable (`core/constants.py:18`, `Kepler-64 Audit` §3). In training it is clamped to $[0.05, 20.0]$ (`training/loss.py:52`). The audit recommends initializing it at 1.0 and tracking η *over time* (see §4.6).

> **⚠ [ISSUE: TRAIN/INFERENCE-ROCHE] (P2, verification doc):** A past run drove `roche` to a *negative* value ($-0.247$, since clipped to $0.05$), which made the disruption sigmoid saturate to 1 for *every* position and silently killed two terms of the eval. The clip in `loss.py` now prevents this, but it shows how a single mis-learned constant can erase the physics. **Always sanity-check learned constants after a training run** (the `trained_constants.json` currently shows `roche=0.05`, `Rg=0.19`, `gamma=0.0` — see §13.5).

## 4.5 The radius of gyration $R_g$ (dynamic: mass-scaled)

$R_g$ lives in `Constants` as a learnable leaf, init 1.0 (`core/constants.py:23`), and is now **mass-scaled** in the evaluation. The implemented η uses an *effective* radius of gyration

$$R_{g,\text{eff}} = R_g \left(\frac{|m_{\text{king}}|}{1000}\right)^{1/3},$$

the constant-density self-gravitating relation (radius grows as the cube root of mass). A King that accretes captured mass grows heavier *and* more extended, so $R_{g,\text{eff}}$ rises and it tears more easily — the physics-native "overextended piece is fragile" coupling (review C13, now resolved). For an ordinary King at mass 1000 the factor is exactly 1, so the worked example in §4.3 is unchanged. See `_eta` in `core/evaluate.py`.

> **⚠ [ISSUE: Rg-DROPPED] (P2, Code Review v2 ISSUE 14):** An earlier `_eta` dropped $R_g$ entirely (used `Rg=1.0` implicitly). The current code keeps the $R_g^3$ factor. Confirm the factor is present wherever η is computed, including `tidal_disruption()` (`core/tidal.py:59`) and the visualizer's `_eta_from_U64` (`viz/glassbox.py:115`) — note the *visualizer* still divides by $G\,M_{\text{king}}^2$, which is the **old** formula and disagrees with the evaluator's $m_{\text{ref}}^2$ version. That mismatch is a visible inconsistency (see §15.4).

## 4.6 Time derivative $d\eta/dt$ (implemented via the Verlet drift term)

The audit and P II both call for tracking **whether η is *increasing*** across a short rollout — $d\eta/dt > 0$ is the real precursor to collapse, more than a single snapshot. This is now implemented as the **Verlet tidal-drift term** (Chapter 6): a short Leapfrog projection advances each King under the opponent's field and the change in tidal stress over the horizon is read into the score, gated by the `lambda_drift` leaf. See §6.5 for the math and `_eta_drift` in `core/evaluate.py`.

> **(Resolved)** The former [ISSUE: DETA-MISSING] — "$d\eta/dt$ is documented but not implemented" — is closed. The Verlet rollout (§6) now realizes the "impending collapse" signal the docs promise.

## 4.7 Forward link

We have the win condition: enemy η high ⇒ good for us; our η high ⇒ bad. The next chapter introduces the **speed of light $c$** as a "reach gate" — gravity propagates with finite delay, which is both a physics theme and a real differentiable input.

**Cross-references:** Tidal tensor / λ₁ → §3. Score assembly using η → §7.4. The Verlet rollout that realizes $dη/dt$ → §6.
