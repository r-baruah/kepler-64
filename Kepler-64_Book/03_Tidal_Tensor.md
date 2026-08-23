# Chapter 3 — The Tidal Tensor and the Largest Eigenvalue λ₁

> **Read me first (no background needed).** Here is the chapter where "attack" gets a physical definition. If ten enemies push your King from one side, the King simply slides away — annoying, not fatal. But if they pull from *opposite sides at once*, the King cannot move anywhere that relieves the stretch: it is being pulled apart. That difference-between-pulls is called a **tide** (the Moon raises Earth's oceans exactly this way), and its mathematical fingerprint is a small 2×2 table of numbers called the **tidal tensor**. From that table one number — the biggest **eigenvalue**, λ₁ ("lambda-one") — summarizes the worst-case stretching in any direction. λ₁ *is* Kepler's measure of attack.

## 3.1 Intuition: why "total force" is the wrong thing to measure

If we only measured the net gravitational force on the enemy King, we'd reward simply *having a lot of mass nearby* — but a King sitting between two equal friendly pieces feels nearly **zero net force** (they pull in opposite directions) while still being violently *stretched* apart. The right physical quantity is not the force, but how the force **changes across space** — its spatial derivative. That derivative is the **tidal tensor.**

Real-world analogy: the Moon raises tides on Earth because the side of Earth facing the Moon feels a stronger pull than the far side. Earth isn't *falling into* the Moon as a whole (it orbits); it is being **stretched** along the Earth–Moon line. That stretching, not the net force, is what causes tides. Kepler-64 measures the exact same stretching at the King.

> **Intuition box:** Tidal force = difference in pull across an object. The Moon doesn't yank Earth sideways; it pulls the near side harder than the far side, stretching it. The King is "stretched" by unequal pulls from your pieces.

## 3.2 The math: gradient, then Hessian

The gravitational potential $U(x,y)$ is a scalar field over the board. Its **gradient** is the force (pointing downhill):

$$\nabla U = \begin{pmatrix} \partial U/\partial x \\ \partial U/\partial y \end{pmatrix}$$

The **Hessian** is the gradient of the gradient — a $2\times 2$ matrix of second derivatives:

$$\mathbf A = \nabla\nabla U = \begin{pmatrix}
\dfrac{\partial^2 U}{\partial x^2} & \dfrac{\partial^2 U}{\partial x\,\partial y} \\[6pt]
\dfrac{\partial^2 U}{\partial y\,\partial x} & \dfrac{\partial^2 U}{\partial y^2}
\end{pmatrix}
= \begin{pmatrix} U_{xx} & U_{xy} \\ U_{yx} & U_{yy} \end{pmatrix}$$

Because mixed partials are equal ($U_{xy}=U_{yx}$), $\mathbf A$ is **real symmetric**. This matrix is the **tidal tensor** at the King's square.

- $U_{xx}, U_{yy}$ measure how the stretching varies along the $x$ and $y$ axes.
- $U_{xy}$ measures the shear (diagonal stretching).

## 3.3 The math: eigenvalues of a 2×2 symmetric matrix (closed form)

We want the **principal axes of stretch** — the directions in which the King is pulled apart most and least. Those directions are the **eigenvectors**, and the amount of stretch along them are the **eigenvalues** $\lambda_1 \ge \lambda_2$.

For a $2\times 2$ symmetric matrix there is no need for a heavy numerical library (LAPACK). The eigenvalues have a closed form (`core/tidal.py:16-24`):

$$\lambda_{1,2} = \frac{\operatorname{tr}(\mathbf A)}{2} \pm \sqrt{\left(\frac{\operatorname{tr}(\mathbf A)}{2}\right)^2 - \det(\mathbf A)}$$

where $\operatorname{tr}(\mathbf A) = U_{xx}+U_{yy}$ and $\det(\mathbf A) = U_{xx}U_{yy} - U_{xy}^2$.

**Why this is a big deal for speed.** This is ~10 floating-point operations. The engine computes it **218 times per move** (once per candidate, see §8). A LAPACK `eigh` call would allocate and be far slower. The closed form is the reason the sub-millisecond claim is plausible (`Kepler-64 Audit` §4 confirms this).

**Worked example.** Suppose at the King the tidal tensor is

$$\mathbf A = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$$

$\operatorname{tr}=5$, $\det = 3\cdot 2 - 1 = 5$. Then

$$\lambda_{1,2} = 2.5 \pm \sqrt{6.25 - 5} = 2.5 \pm \sqrt{1.25} = 2.5 \pm 1.118$$

$$\lambda_1 = 3.618,\qquad \lambda_2 = 1.382$$

So the principal stretching rate is $3.618$ along the eigenvector of $\lambda_1$, and the King is being compressed at $1.382$ along the perpendicular axis. The **eigenvector of $\lambda_1$ is the "line of failure"** the visualizer draws in red (`viz/glassbox.py`, Chapter 15).

## 3.4 Computing the Hessian on the lattice (finite differences)

The potential $U$ is only known at discrete squares, so we approximate second derivatives with **central finite differences**. For the King at row $r$, file $f$:

$$U_{xx} \approx U[r+1,f] - 2U[r,f] + U[r-1,f]$$
$$U_{yy} \approx U[r,f+1] - 2U[r,f] + U[r,f-1]$$
$$U_{xy} \approx \frac{U[r+1,f+1] - U[r+1,f-1] - U[r-1,f+1] + U[r-1,f-1]}{4}$$

This is exactly `core/tidal.py:41-43`. The factor 4 for the cross term is the correct $h_x h_y$ with unit spacing.

> **⚠ [ISSUE: BOUNDARY-HESSIAN] (P0, Code Review v2 BUG 3):** The naive central-difference stencil **collapses at the board edge.** If the King is on rank 1 ($r=0$), then `r-1` would be $-1$, which (before the fix) wrapped or repeated the King's own square, turning a *second* derivative into a *first* derivative — i.e. the tidal tensor became mathematically wrong for **~25% of possible King positions** (any King on ranks 1/8 or files a/h). The fix in `core/tidal.py:38` is to `jnp.pad(Ug, 1, mode="edge")` to a $10\times 10$ grid *before* taking differences, so every stencil has three distinct points. **Verify this is present before trusting the win condition.** The earlier Code Review v1 BUG 7 documents the same class of error with the denominator.

## 3.5 Project link: the tidal tensor is the win condition

In Kepler-64, **disruption is source-attributed** — a King is torn by the *opponent's* masses, never its own (its own gravity binds it). So we compute the tidal tensor of the *enemy's* potential at *our* King, and of *our* potential at the *enemy* King (see §7.4). The largest eigenvalue feeds directly into the disruption index η (Chapter 4).

> **Project link:** `tidal_tensor_at()` (`core/tidal.py:27`) and `eig2x2()` (`core/tidal.py:16`) are called from `_eta()` and from the visualizer. They are pure JAX and fully differentiable.

> **⚠ [ISSUE: POINT-MASS-TIDAL] (P2, Kepler-64 P II §1):** A literal tidal tensor measures stretching across a *physical volume*. A chess King is a point with no volume, so "literal tidal stress" is physically meaningless. The project's correct response (P II §1) is to **drop the literal interpretation** and treat $\lambda_1$ as a mathematically clean "principal axis of stress" — a heuristic, not a physical tearing. The book and docs should say this explicitly so a physics reviewer is not tempted to attack it.

## 3.6 Forward link

We can now compute, at any King, the principal stretch $\lambda_1$. The next chapter turns $\lambda_1$ into a **dimensionless disruption parameter** η by scaling it against the King's self-binding, giving the Roche/Hill criterion that decides victory.

**Cross-references:** Potential field $U$ → §2.4. Eigenvalues in the score → §7.4. The η formula and its constant choice → §4.
