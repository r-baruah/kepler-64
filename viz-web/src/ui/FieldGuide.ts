/**
 * Kepler-64 Interactive Research Compendium & Notebook
 * Split-view scientific guide with interactive chapter simulator.
 */

import { latex } from './katexUtil';

export interface Chapter {
  id: string;
  num: string;
  title: string;
  subtitle: string;
  summary: string;
  latexFormula: string;
  intuition: string;
  mathExplanation: string;
  demoFen: string;
  demoTitle: string;
  keyMetrics: { label: string; value: string }[];
}

export const CHAPTERS: Chapter[] = [
  {
    id: 'ch1-lattice-mass',
    num: '01',
    title: 'Gravity on a Discrete Lattice',
    subtitle: 'Every piece is an attractive point mass in 2D spacetime',
    summary: 'The 64 squares form a fixed coordinate lattice. Pieces carry mass based on standard material values, with the King assigned an immense central stellar mass (1000m) to anchor the coordinate spacetime.',
    latexFormula: `\\vec{F}_i = G \\sum_{j \\neq i} |m_j| \\, \\frac{\\vec{r}_i - \\vec{r}_j}{\\left(\\lVert \\vec{r}_i - \\vec{r}_j\\rVert^2 + \\varepsilon^2\\right)^{3/2}} \\cdot \\sigma(c - d_{ij})`,
    intuition: 'Instead of heuristic questions like "who controls d4?", Kepler-64 evaluates a physical state: "What is the net spacetime curvature, and is the enemy King sinking into an inescapable gravitational well?"',
    mathExplanation: 'Because the 8×8 geometry is invariant across the game, the pairwise Euclidean distance matrix is precomputed once. All 64-square force interactions execute as a vectorized linear algebra operation in sub-millisecond time.',
    demoFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    demoTitle: 'Initial Lattice (Symmetric Balance)',
    keyMetrics: [
      { label: 'King Mass M_k', value: '1000.0 m' },
      { label: 'Queen Mass M_q', value: '9.0 m' },
      { label: 'Pairwise Stencil', value: '64 × 64 Matrix' },
    ],
  },
  {
    id: 'ch2-plummer-softening',
    num: '02',
    title: 'Plummer Softening Length',
    subtitle: 'Taming infinite singularities on a discrete board',
    summary: 'Pure Newtonian 1/r² force diverges to infinity when distance r → 0. Plummer softening introduces a characteristic length scale ε that smoothly rounds the potential core while maintaining differentiability.',
    latexFormula: `\\Phi(p) = -G \\sum_{j=1}^{64} \\frac{|m_j| \\cdot \\sigma(c - \\lVert p - r_j \\rVert)}{\\sqrt{\\lVert p - r_j \\rVert^2 + \\varepsilon^2}}`,
    intuition: 'Think of heavy bowling balls on a trampoline. Without softening, the balls would puncture infinite pinprick tears. Softening ε acts like a protective sphere radius so the trampoline sheet remains smoothly curved everywhere.',
    mathExplanation: 'Smoothness guarantees that position evaluations are $C^\\infty$ differentiable. JAX automatic differentiation computes exact analytical gradients with respect to ε, allowing gradient descent to discover the optimal core geometry.',
    demoFen: 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
    demoTitle: 'Open Game Potential Wells (e4 vs e5 wells)',
    keyMetrics: [
      { label: 'Softening ε', value: '0.50 sq' },
      { label: 'Newtonian Limit', value: 'lim ε→0' },
      { label: 'Core Max Depth', value: '-G·m / ε' },
    ],
  },
  {
    id: 'ch3-tidal-tensor',
    num: '03',
    title: 'The Tidal Tensor & Line of Failure',
    subtitle: 'The 2D Hessian matrix measures directional stretching across the King',
    summary: 'Total force is the wrong metric: a King pinned between two equal pieces feels zero net force while still being violently torn apart. The tidal tensor measures the spatial derivative of the force (the 2×2 Hessian of potential).',
    latexFormula: `\\mathbf{A} = \\nabla\\nabla\\Phi\\big|_{\\text{King}} = \\begin{pmatrix} \\dfrac{\\partial^2 \\Phi}{\\partial x^2} & \\dfrac{\\partial^2 \\Phi}{\\partial x \\partial y} \\\\[6pt] \\dfrac{\\partial^2 \\Phi}{\\partial y \\partial x} & \\dfrac{\\partial^2 \\Phi}{\\partial y^2} \\end{pmatrix}`,
    intuition: 'The Moon raises ocean tides on Earth not because it pulls Earth sideways, but because it pulls the near side harder than the far side. Kepler-64 computes this exact differential strain at the enemy King square.',
    mathExplanation: 'The 2×2 real symmetric matrix has closed-form analytical eigenvalues λ₁ ≥ λ₂ computed in ~10 FLOPs. Eigenvalue λ₁ is the principal stretching rate, and its eigenvector v₁ defines the principal line of structural failure.',
    demoFen: 'r1bqk2r/pp1p1pbp/2n1pnp1/8/2BNP3/2N1B3/PPP2PPP/R2QK2R w KQkq - 2 8',
    demoTitle: 'Sicilian Dragon: Kingside & Queenside Tension',
    keyMetrics: [
      { label: 'Hessian Stencil', value: '2D Central Diff' },
      { label: 'Max Eigenvalue λ₁', value: 'Principal Strain' },
      { label: 'Eigenvector v₁', value: 'Failure Axis θ' },
    ],
  },
  {
    id: 'ch4-roche-disruption',
    num: '04',
    title: 'The Dimensionless Roche Limit η',
    subtitle: 'When tearing forces overcome defensive cohesion, the King collapses',
    summary: 'Derived from astronomical Roche/Hill sphere stability: when a satellite ventures inside the Roche radius, external tidal shear overcomes internal gravity and disintegrates it into a debris ring.',
    latexFormula: `\\eta = \\frac{R_g^3 \\cdot \\lambda_1}{m_{\\text{ref}}^2}, \\qquad \\text{Collapse Criterion: } \\eta > \\rho_{\\text{roche}}`,
    intuition: 'A snowman in a blizzard: its internal cohesion holds it together until the spatial wind gradient shears it in half. η is tearing force divided by cohesion. Above the threshold, checkmate is realized as physical rupture.',
    mathExplanation: 'The denominator uses a fixed minor-piece reference mass m_ref = 3.5 instead of M_king², preventing the King\'s 1000m mass from dampening η to 10⁻⁶. This scales η into a clean, interpretable [0.05 .. 1.5+] range.',
    demoFen: 'r1b1k2r/pppp1ppp/8/8/1b1q4/8/PPPP1PPP/R1B1KB1R b KQkq - 1 9',
    demoTitle: 'Extreme Attack: King in Critical Roche Strain',
    keyMetrics: [
      { label: 'Critical Limit ρ', value: '1.00' },
      { label: 'Ref Mass m_ref', value: '3.5 m' },
      { label: 'Self-Gravity R_g', value: '1.0 sq' },
    ],
  },
  {
    id: 'ch5-retarded-potentials',
    num: '05',
    title: 'Speed of Light & Propagation Reach',
    subtitle: 'Gravity waves propagate at finite speed c across the board',
    summary: 'The speed of light c acts as a differentiable reach gate. Distant pieces do not exert instantaneous attraction; their gravity waves arrive with a sigmoid distance horizon.',
    latexFormula: `\\text{gate}(d) = \\sigma(c - d) = \\frac{1}{1 + e^{-(c - d)}}, \\qquad c \\in [1.0, 10.0] \\text{ sq/ply}`,
    intuition: 'A Queen sacrifice on h8 does not instantly register as a threat to a King on a1 until several plies later, when the gravitational wave propagates across the 8-square diagonal.',
    mathExplanation: 'A monotonicity prior penalty keeps c bounded to [1.0, 10.0]. Without this prior, gradient descent pushes c → ∞ (instantaneous gravity is easier to optimize), collapsing the relativistic narrative.',
    demoFen: '8/8/8/8/8/8/4k3/K6Q w - - 0 1',
    demoTitle: 'Corner-to-Corner Distance Gate (a1 to h1)',
    keyMetrics: [
      { label: 'Speed of Light c', value: '4.00 sq/ply' },
      { label: 'Horizon Sigmoid', value: 'Smooth Gate' },
      { label: 'Max Reach', value: '8√2 ≈ 11.3 sq' },
    ],
  },
  {
    id: 'ch6-accretion-multiverse',
    num: '06',
    title: 'Accretion & The Multiverse (Layer 2)',
    subtitle: 'Captured mass is absorbed, and constants are drawn from a posterior',
    summary: 'Captured pieces do not vanish—they are accreted by the capturing piece with 80% mass retention. In Layer 2, moves are evaluated across K parallel universes sampled from a learned posterior over (G, ε, c).',
    latexFormula: `m_{\\text{new}} = m_{\\text{captor}} + 0.8 \\cdot |m_{\\text{captured}}|, \\qquad \\operatorname{Eval}(m) = \\frac{1}{K}\\sum_{k=1}^K \\operatorname{Eval}\\big(m; \\theta_k\\big)`,
    intuition: 'A piece that hoards captures becomes supermassive, creating a deep gravity well, but also becomes spatially extended and more vulnerable to tidal disruption. Power has an inherent physical cost.',
    mathExplanation: 'Bayesian model averaging over K=8 posterior realizations prevents the engine from over-relying on a single fragile constant configuration, creating strategic epistemic humility.',
    demoFen: 'r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5',
    demoTitle: 'Accreted Knight Attack on f7',
    keyMetrics: [
      { label: 'Accretion Rate', value: '80% Mass' },
      { label: 'Multiverse Worlds', value: 'K = 8' },
      { label: 'Prior Distribution', value: 'p(G, ε, c)' },
    ],
  },
];

export class FieldGuideComponent {
  private container: HTMLElement;
  private activeChapterIndex: number = 0;
  private onSelectDemo?: (fen: string, title: string) => void;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public onSelectPosition(cb: (fen: string, title: string) => void): void {
    this.onSelectDemo = cb;
  }

  public render(): void {
    const ch = CHAPTERS[this.activeChapterIndex];

    this.container.innerHTML = `
      <div class="field-guide-notebook shell" id="compendium">
        <!-- Header -->
        <div class="notebook-header">
          <span class="section-number">04 — Compendium</span>
          <span class="badge-tag">KEPLER-64 SCIENTIFIC COMPENDIUM</span>
          <h2>The Mathematical &amp; Physical Foundations</h2>
          <p class="section-lead">
            An interactive textbook companion explaining the differentiable tensor physics behind Kepler-64.
          </p>
        </div>

        <!-- Split View Notebook -->
        <div class="notebook-split">
          <!-- Left: Chapter Index Menu -->
          <nav class="notebook-index">
            ${CHAPTERS.map((c, idx) => `
              <button class="index-nav-item ${idx === this.activeChapterIndex ? 'active' : ''}" data-ch-idx="${idx}">
                <div class="nav-ch-num">${c.num}</div>
                <div class="nav-ch-info">
                  <div class="nav-ch-title">${c.title}</div>
                  <div class="nav-ch-sub">${c.subtitle}</div>
                </div>
              </button>
            `).join('')}
          </nav>

          <!-- Right: Chapter Content Display -->
          <article class="notebook-article">
            <div class="article-meta-badge">CHAPTER ${ch.num} OF 06</div>
            <h3 class="article-title">${ch.title}</h3>
            <p class="article-subtitle">${ch.subtitle}</p>

            <!-- KaTeX Math Box -->
            <div class="article-math-card">
              <div class="math-card-label">GOVERNING EQUATION</div>
              <div class="math-card-body">
                ${latex(ch.latexFormula, true)}
              </div>
            </div>

            <!-- Intuition Callout -->
            <div class="article-intuition-box">
              <strong>PHYSICAL INTUITION:</strong>
              <p style="margin-top:4px;">${ch.intuition}</p>
            </div>

            <!-- Deep Dive Text -->
            <div class="article-body-text">
              <p>${ch.summary}</p>
              <p style="margin-top:var(--space-md);">${ch.mathExplanation}</p>
            </div>

            <!-- Key Metrics Strip -->
            <div class="article-metrics-strip">
              ${ch.keyMetrics.map((km) => `
                <div class="metric-item">
                  <span class="metric-val">${km.value}</span>
                  <span class="metric-lbl">${km.label}</span>
                </div>
              `).join('')}
            </div>

            <!-- Interactive Board Loader Action -->
            <div class="article-demo-box">
              <div>
                <span style="font-family:var(--font-mono); font-size:0.75rem; font-weight:700; color:var(--color-muted);">INTERACTIVE BENCHMARK</span>
                <div style="font-weight:700; font-size:1.05rem;">${ch.demoTitle}</div>
              </div>
              <button id="btn-article-inject" class="action-primary" data-fen="${ch.demoFen}" data-title="${ch.demoTitle}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Load into Observatory Board
              </button>
            </div>
          </article>
        </div>
      </div>
    `;

    this.attachEvents();
  }

  private attachEvents(): void {
    // Chapter selection
    this.container.querySelectorAll('.index-nav-item').forEach((item) => {
      item.addEventListener('click', (e) => {
        const idx = parseInt((e.currentTarget as HTMLElement).getAttribute('data-ch-idx') || '0', 10);
        this.activeChapterIndex = idx;
        this.render();
        const activeNav = this.container.querySelector(`.index-nav-item[data-ch-idx="${idx}"]`);
        activeNav?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      });
    });

    // Inject demo button
    const injectBtn = this.container.querySelector('#btn-article-inject');
    injectBtn?.addEventListener('click', (e) => {
      const fen = (e.currentTarget as HTMLElement).getAttribute('data-fen');
      const title = (e.currentTarget as HTMLElement).getAttribute('data-title');
      if (fen && title && this.onSelectDemo) {
        this.onSelectDemo(fen, title);
        document.querySelector('#observatory')?.scrollIntoView({ behavior: 'smooth' });
      }
    });
  }
}
