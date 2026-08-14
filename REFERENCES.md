# Scientific References & Theoretical Foundations

This document catalogs the academic literature, astrophysical foundations, computational mechanics, and chess programming references foundational to the **Kepler-64 (The Roche Engine)** project.

---

## 1. Astrophysical N-Body & Potential Theory

1. **Plummer, H. C. (1911).** *On the problem of distribution in globular star clusters.* Monthly Notices of the Royal Astronomical Society, 71(5), 460–470.
   - Introduced the Plummer potential core $\Phi(r) = -\frac{GM}{\sqrt{r^2 + b^2}}$, preventing point-mass numerical divergence.
2. **Binney, J., & Tremaine, S. (2008).** *Galactic Dynamics* (2nd ed.). Princeton University Press.
   - Theoretical framework for gravitational potential theory, Poisson equation solvers, and stellar orbital mechanics.
3. **Roche, É. (1849).** *La figure d'une masse fluide soumise à l'attraction d'un point éloigné.* Académie des Sciences de Montpellier, 1, 243–262.
   - Original derivation of the tidal disruption threshold for self-gravitating fluid bodies orbiting massive primaries.
4. **Hill, G. W. (1878).** *Researches in the Lunar Theory.* American Journal of Mathematics, 1(1), 5–26.
   - Derivation of the Hill sphere radius and zero-velocity surfaces in the restricted three-body problem.

---

## 2. Differentiable Physics & Machine Learning

5. **Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018).** *JAX: Composable transformations of Python+NumPy programs.*
   - Autodiff and XLA compilation framework used to compute analytical gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ over physical constants $(G, \varepsilon, c, \rho_{\text{roche}})$.
6. **Degrave, J., Hermans, M., Dambre, J., & Wyffels, F. (2019).** *A Differentiable Physics Engine for Deep Learning in Robotics.* Frontiers in Neurorobotics, 13, 6.
   - Principles of inductive bias integration via differentiable physics simulators.

---

## 3. Combinatorial Games & Classical Chess Engines

7. **Shannon, C. E. (1950).** *Programming a Computer for Playing Chess.* Philosophical Magazine, Ser. 7, 41(314), 256–275.
   - Established evaluation function paradigms and minimax game-tree search.
8. **Silver, D., Hubert, T., Schrittwieser, J., et al. (2018).** *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science, 362(6419), 1140–1144.
   - AlphaZero architecture demonstrating self-learned positional dynamics and queen sacrifices.
9. **Stockfish Engine Development Team (2008–2026).** *Stockfish: A strong open-source chess engine.* Available at https://stockfishchess.org.
   - Benchmark reference for alpha-beta minimax search, move generation, and transposition tables.

---

## 4. How to Cite Kepler-64

```bibtex
@software{baruah2026kepler64,
  author       = {Baruah, Ripuranjan},
  title        = {Kepler-64: Differentiable N-Body Gravitational Chess Engine},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/r-baruah/kepler-64}}
}
```
