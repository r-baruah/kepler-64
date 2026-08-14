/**
 * Kepler-64 Project Contributors & Scientific Attribution Component
 */

export class ContributorsSection {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public render(): void {
    this.container.innerHTML = `
      <div class="contributors-wrapper shell" id="contributors">
        <div class="section-badge-header">
          <span class="badge-tag">ORIGIN & RESEARCH TEAM</span>
          <h2>Project Leadership & Contributors</h2>
          <p class="section-lead">
            Kepler-64 is an open-source astrophysical chess research project founded to pioneer differentiable physical heuristics in combinatorial games.
          </p>
        </div>

        <div class="contributors-grid">
          <!-- Founder Card -->
          <div class="creator-card founder-card">
            <div class="creator-header">
              <div class="creator-avatar">RB</div>
              <div>
                <h3 class="creator-name">Ripuranjan Baruah</h3>
                <div class="creator-role">Original Creator & Lead Architect</div>
              </div>
              <div class="founder-badge">FOUNDER</div>
            </div>
            <p class="creator-bio">
              Conceived the Kepler-64 gravitational chess paradigm, mathematical formulation of the 2D lattice tidal tensor, and the JAX-based differentiable physics engine.
            </p>
            <div class="creator-links">
              <a href="https://github.com/r-baruah" target="_blank" rel="noreferrer" class="action-secondary">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
                GitHub Profile
              </a>
              <a href="https://github.com/r-baruah/kepler-64" target="_blank" rel="noreferrer" class="action-primary">
                ★ Star Repository
              </a>
            </div>
          </div>

          <!-- Open Source Community & Research Card -->
          <div class="creator-card">
            <div class="creator-header">
              <div class="creator-avatar" style="background:var(--color-plate);">OS</div>
              <div>
                <h3 class="creator-name">Open Source Community</h3>
                <div class="creator-role">Research, Benchmarks & Optimization</div>
              </div>
            </div>
            <p class="creator-bio">
              Contributions spanning Stockfish evaluation comparisons, self-play tournament harnesses, WebGL potential shaders, and documentation.
            </p>
            <div class="community-stats">
              <div class="c-stat">
                <span class="c-num">100%</span>
                <span class="c-label">Open Source</span>
              </div>
              <div class="c-stat">
                <span class="c-num">Apache 2.0</span>
                <span class="c-label">License</span>
              </div>
              <div class="c-stat">
                <span class="c-num">JAX</span>
                <span class="c-label">Auto-Diff Core</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Citation Box -->
        <div class="citation-container">
          <div class="citation-header">
            <span>HOW TO CITE THIS WORK</span>
            <button id="btn-copy-bibtex" class="action-secondary" style="padding:3px 8px; font-size:0.75rem;">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              Copy BibTeX
            </button>
          </div>
          <pre class="citation-code"><code>@software{baruah2026kepler64,
  author       = {Baruah, Ripuranjan},
  title        = {Kepler-64: Differentiable N-Body Gravitational Chess Engine},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\\url{https://github.com/r-baruah/kepler-64}}
}</code></pre>
        </div>
      </div>
    `;

    this.attachEvents();
  }

  private attachEvents(): void {
    const copyBtn = this.container.querySelector('#btn-copy-bibtex');
    copyBtn?.addEventListener('click', () => {
      const code = `@software{baruah2026kepler64,\n  author       = {Baruah, Ripuranjan},\n  title        = {Kepler-64: Differentiable N-Body Gravitational Chess Engine},\n  year         = {2026},\n  publisher    = {GitHub},\n  journal      = {GitHub Repository},\n  howpublished = {\\url{https://github.com/r-baruah/kepler-64}}\n}`;
      navigator.clipboard.writeText(code).then(() => {
        copyBtn.innerHTML = `✓ Copied!`;
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            Copy BibTeX
          `;
        }, 2000);
      });
    });
  }
}
