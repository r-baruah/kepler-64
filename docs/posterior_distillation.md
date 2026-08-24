# Posterior Distillation — Which Constants Does the Physics Constrain?

*One-at-a-time multiplicative sensitivity (×0.8 / ×1.25) of the Layer-1 score across a 6-position panel, base universe. Mean |Deltascore| per leaf, sorted. High sensitivity = the leaf does real work and training can learn it; near-zero = currently unconstrained by the physics.*

| Rank | Leaf | Mean |dScore| | Share of total sensitivity |
|---|---|---|---|
| 1 | `c` | 3.4949 | 31.2% |
| 2 | `roche` | 3.0046 | 26.8% |
| 3 | `G` | 1.9399 | 17.3% |
| 4 | `kgain` | 1.2012 | 10.7% |
| 5 | `bonus` | 0.8928 | 8.0% |
| 6 | `mat_gain` | 0.4241 | 3.8% |
| 7 | `eps` | 0.1189 | 1.1% |
| 8 | `inertia_gain` | 0.0587 | 0.5% |
| 9 | `com_gain` | 0.0516 | 0.5% |
| 10 | `Rg` | 0.0035 | 0.0% |
| 11 | `entropy_gain` | 0.0002 | 0.0% |
| 12 | `lambda_delta` | 0.0001 | 0.0% |
| 13 | `dt_drift` | 0.0000 | 0.0% |
| 14 | `lambda_drift` | 0.0000 | 0.0% |
| 15 | `gamma` | 0.0000 | 0.0% |
| 16 | `lambda_gw` | 0.0000 | 0.0% |
| 17 | `lambda_sch` | 0.0000 | 0.0% |

**Unconstrained at current defaults:** `gamma`, `lambda_gw`, `lambda_sch`
(expected for the zero-init gains `lambda_gw` / `lambda_sch` — they are
behavior-neutral until training or a hand raise switches them on.)
