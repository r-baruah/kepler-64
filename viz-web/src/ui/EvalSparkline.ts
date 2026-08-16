/**
 * Kepler-64 Panoramic Game Timeline & Evaluation Barometer
 * Renders an interactive, annotated full-width gravitational trajectory.
 */

export interface SparklinePoint {
  ply: number;
  score: number; // White perspective
  moveSan: string;
}

export interface MultiverseSparkPoint {
  ply: number;
  moveSan: string;
  mean: number;        // white-perspective mean score
  sigma: number;       // volatility (population stddev)
  spaghetti: number[]; // individual universe scores (white perspective), up to 5
}

interface MultiverseCoord {
  x: number;
  y: number;
  ply: number;
  moveSan: string;
  mean: number;
  sigma: number;
  spaghetti: number[];
}

export class EvalSparkline {
  private container: HTMLElement;
  private points: SparklinePoint[] = [];
  private mvPoints: MultiverseSparkPoint[] | null = null;
  private currentPly: number = 0;
  private onSelectPlyCallback?: (ply: number) => void;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public setData(points: SparklinePoint[], currentPly: number): void {
    this.points = points;
    this.mvPoints = null;
    this.currentPly = currentPly;
    this.render();
  }

  public setMultiverseData(points: MultiverseSparkPoint[], currentPly: number): void {
    this.mvPoints = points;
    this.points = [];
    this.currentPly = currentPly;
    this.render();
  }

  public setCurrentPly(currentPly: number): void {
    this.currentPly = currentPly;
    this.updateMarker();
  }

  public onSelectPly(cb: (ply: number) => void): void {
    this.onSelectPlyCallback = cb;
  }

  private render(): void {
    if (this.mvPoints) {
      this.renderMultiverse();
    } else {
      this.renderSingle();
    }
  }

  private renderSingle(): void {
    if (!this.points.length) {
      this.container.innerHTML = '';
      return;
    }

    const width = 800;
    const height = 90;
    const padX = 20;
    const padY = 12;

    const clampSpan = this.computeSingleClampSpan();
    const zeroY = height / 2.0;

    // Build SVG paths
    const n = this.points.length;
    const lineD: string[] = [];
    const areaWhiteD: string[] = [];
    const areaBlackD: string[] = [];

    const coords: { x: number; y: number; ply: number; score: number; moveSan: string }[] = [];

    for (let i = 0; i < n; i++) {
      const p = this.points[i];
      const x = padX + (i / Math.max(1, n - 1)) * (width - 2 * padX);
      const y = this.yFor(p.score, clampSpan, height, padY);
      coords.push({ x, y, ply: p.ply, score: p.score, moveSan: p.moveSan });

      if (i === 0) {
        lineD.push(`M ${x.toFixed(1)} ${y.toFixed(1)}`);
      } else {
        lineD.push(`L ${x.toFixed(1)} ${y.toFixed(1)}`);
      }
    }

    // Build positive/negative area fills
    areaWhiteD.push(`M ${coords[0].x.toFixed(1)} ${zeroY.toFixed(1)}`);
    coords.forEach((c) => {
      const yClamped = Math.min(zeroY, c.y);
      areaWhiteD.push(`L ${c.x.toFixed(1)} ${yClamped.toFixed(1)}`);
    });
    areaWhiteD.push(`L ${coords[coords.length - 1].x.toFixed(1)} ${zeroY.toFixed(1)} Z`);

    areaBlackD.push(`M ${coords[0].x.toFixed(1)} ${zeroY.toFixed(1)}`);
    coords.forEach((c) => {
      const yClamped = Math.max(zeroY, c.y);
      areaBlackD.push(`L ${c.x.toFixed(1)} ${yClamped.toFixed(1)}`);
    });
    areaBlackD.push(`L ${coords[coords.length - 1].x.toFixed(1)} ${zeroY.toFixed(1)} Z`);

    const activeCoord = coords[this.currentPly] || coords[0];

    this.container.innerHTML = `
      <div class="panoramic-timeline-card">
        <div class="timeline-header-bar">
          <div style="display:flex; align-items:center; gap:var(--space-sm);">
            <span class="timeline-tag">GRAVITATIONAL TRAJECTORY</span>
            <span class="timeline-ply-badge" id="timeline-hover-info">Move: ${activeCoord.moveSan} · Score: ${(activeCoord.score >= 0 ? '+' : '') + activeCoord.score.toFixed(2)}</span>
          </div>
          <div class="timeline-legend">
            <span class="legend-dot white-dot"></span> White Pull (+ΔE)
            <span class="legend-dot black-dot"></span> Black Pull (-ΔE)
          </div>
        </div>

        <div class="timeline-svg-wrapper">
          <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" class="timeline-svg">
            <!-- Zero Equilibrium Axis -->
            <line x1="${padX}" y1="${zeroY}" x2="${width - padX}" y2="${zeroY}" stroke="var(--color-rule)" stroke-width="1.2" stroke-dasharray="4,4" />

            <!-- Shaded Advantage Areas (flat fills) -->
            <path d="${areaWhiteD.join(' ')}" fill="var(--color-plate)" fill-opacity="0.10" />
            <path d="${areaBlackD.join(' ')}" fill="var(--color-accent)" fill-opacity="0.10" />

            <!-- Trajectory Wave -->
            <path d="${lineD.join(' ')}" fill="none" stroke="var(--color-ink)" stroke-width="2.2" stroke-linejoin="round" />

            <!-- Active Cursor Vertical Line -->
            <line id="timeline-cursor-line" x1="${activeCoord.x.toFixed(1)}" y1="4" x2="${activeCoord.x.toFixed(1)}" y2="${height - 4}" stroke="var(--color-accent)" stroke-width="1.8" stroke-dasharray="2,2" />

            <!-- Active Cursor Marker -->
            <circle id="timeline-cursor-dot" cx="${activeCoord.x.toFixed(1)}" cy="${activeCoord.y.toFixed(1)}" r="5.5" fill="var(--color-accent)" stroke="var(--color-ink)" stroke-width="2" />
          </svg>
        </div>
      </div>
    `;

    this.attachEvents();
  }

  private renderMultiverse(): void {
    if (!this.mvPoints || !this.mvPoints.length) {
      this.container.innerHTML = '';
      return;
    }

    const width = 800;
    const height = 90;
    const padX = 20;
    const padY = 12;

    const { clampSpan, coords } = this.buildMultiverseCoords();
    const zeroY = height / 2.0;

    // Mean line
    const meanLineD: string[] = [];
    coords.forEach((c, i) => {
      if (i === 0) {
        meanLineD.push(`M ${c.x.toFixed(1)} ${c.y.toFixed(1)}`);
      } else {
        meanLineD.push(`L ${c.x.toFixed(1)} ${c.y.toFixed(1)}`);
      }
    });

    // Confidence envelope (top edge forward, bottom edge backward)
    const topPts: string[] = [];
    const bottomPts: string[] = [];
    coords.forEach((c) => {
      const sigma = Math.min(c.sigma, 4.0);
      topPts.push(`${c.x.toFixed(1)} ${this.yFor(c.mean + sigma, clampSpan, height, padY).toFixed(1)}`);
      bottomPts.push(`${c.x.toFixed(1)} ${this.yFor(c.mean - sigma, clampSpan, height, padY).toFixed(1)}`);
    });
    const envelopePath = `M ${topPts[0]} ${topPts.slice(1).map((p) => `L ${p}`).join(' ')} ${bottomPts.slice().reverse().map((p) => `L ${p}`).join(' ')} Z`;

    // Spaghetti lines (cap at 5 universes)
    const maxUniverseCount = Math.min(5, coords.reduce((max, c) => Math.max(max, c.spaghetti.length), 0));
    const spaghettiPaths: string[] = [];
    for (let u = 0; u < maxUniverseCount; u++) {
      const segments: string[] = [];
      let started = false;
      coords.forEach((c) => {
        if (u >= c.spaghetti.length) return;
        const s = c.spaghetti[u];
        const y = this.yFor(s, clampSpan, height, padY);
        if (!started) {
          segments.push(`M ${c.x.toFixed(1)} ${y.toFixed(1)}`);
          started = true;
        } else {
          segments.push(`L ${c.x.toFixed(1)} ${y.toFixed(1)}`);
        }
      });
      if (segments.length) {
        const stroke = u % 2 === 0 ? 'var(--color-plate)' : 'var(--color-accent)';
        const opacity = u % 2 === 0 ? '0.18' : '0.16';
        spaghettiPaths.push(`<path d="${segments.join(' ')}" fill="none" stroke="${stroke}" stroke-width="1" stroke-opacity="${opacity}" stroke-linejoin="round" />`);
      }
    }

    const activeCoord = coords[this.currentPly] || coords[0];
    const clampSigma = Math.min(activeCoord.sigma, 4.0);

    this.container.innerHTML = `
      <div class="panoramic-timeline-card">
        <div class="timeline-header-bar">
          <div style="display:flex; align-items:center; gap:var(--space-sm);">
            <span class="timeline-tag">GRAVITATIONAL TRAJECTORY</span>
            <span class="timeline-ply-badge" id="timeline-hover-info">Multiverse Volatility: ±${clampSigma.toFixed(2)} native (σ = ${activeCoord.sigma.toFixed(2)})</span>
          </div>
          <div class="timeline-legend">
            <span class="legend-dot white-dot"></span> White Pull (+ΔE)
            <span class="legend-dot black-dot"></span> Black Pull (-ΔE)
          </div>
        </div>

        <div class="timeline-svg-wrapper">
          <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" class="timeline-svg">
            <!-- Zero Equilibrium Axis -->
            <line x1="${padX}" y1="${zeroY}" x2="${width - padX}" y2="${zeroY}" stroke="var(--color-rule)" stroke-width="1.2" stroke-dasharray="4,4" />

            <!-- Confidence Envelope (flat fill) -->
            <path d="${envelopePath}" fill="var(--color-plate)" fill-opacity="0.12" />

            <!-- Spaghetti Universe Threads -->
            ${spaghettiPaths.join('\n            ')}

            <!-- Mean Trajectory -->
            <path d="${meanLineD.join(' ')}" fill="none" stroke="var(--color-ink)" stroke-width="2.2" stroke-linejoin="round" />

            <!-- Active Cursor Vertical Line -->
            <line id="timeline-cursor-line" x1="${activeCoord.x.toFixed(1)}" y1="4" x2="${activeCoord.x.toFixed(1)}" y2="${height - 4}" stroke="var(--color-accent)" stroke-width="1.8" stroke-dasharray="2,2" />

            <!-- Active Cursor Marker -->
            <circle id="timeline-cursor-dot" cx="${activeCoord.x.toFixed(1)}" cy="${activeCoord.y.toFixed(1)}" r="5.5" fill="var(--color-accent)" stroke="var(--color-ink)" stroke-width="2" />
          </svg>
        </div>
      </div>
    `;

    this.attachEvents();
  }

  private updateMarker(): void {
    const cursorLine = this.container.querySelector('#timeline-cursor-line') as SVGLineElement;
    const cursorDot = this.container.querySelector('#timeline-cursor-dot') as SVGCircleElement;
    const hoverInfo = this.container.querySelector('#timeline-hover-info') as HTMLElement;
    if (!cursorDot) return;

    if (this.mvPoints) {
      this.updateMultiverseMarker(cursorLine, cursorDot, hoverInfo);
    } else {
      this.updateSingleMarker(cursorLine, cursorDot, hoverInfo);
    }
  }

  private updateSingleMarker(cursorLine: SVGLineElement | null, cursorDot: SVGCircleElement, hoverInfo: HTMLElement | null): void {
    if (!this.points.length) return;

    const width = 800;
    const height = 90;
    const padX = 20;
    const padY = 12;
    const clampSpan = this.computeSingleClampSpan();

    const n = this.points.length;
    const i = Math.max(0, Math.min(n - 1, this.currentPly));
    const p = this.points[i];
    if (!p) return;

    const x = padX + (i / Math.max(1, n - 1)) * (width - 2 * padX);
    const y = this.yFor(p.score, clampSpan, height, padY);

    cursorLine?.setAttribute('x1', x.toFixed(1));
    cursorLine?.setAttribute('x2', x.toFixed(1));
    cursorDot.setAttribute('cx', x.toFixed(1));
    cursorDot.setAttribute('cy', y.toFixed(1));

    if (hoverInfo) {
      hoverInfo.textContent = `Move: ${p.moveSan} · Score: ${(p.score >= 0 ? '+' : '') + p.score.toFixed(2)}`;
    }
  }

  private updateMultiverseMarker(cursorLine: SVGLineElement | null, cursorDot: SVGCircleElement, hoverInfo: HTMLElement | null): void {
    if (!this.mvPoints || !this.mvPoints.length) return;

    const { coords } = this.buildMultiverseCoords();
    const n = coords.length;
    const i = Math.max(0, Math.min(n - 1, this.currentPly));
    const c = coords[i];
    if (!c) return;

    cursorLine?.setAttribute('x1', c.x.toFixed(1));
    cursorLine?.setAttribute('x2', c.x.toFixed(1));
    cursorDot.setAttribute('cx', c.x.toFixed(1));
    cursorDot.setAttribute('cy', c.y.toFixed(1));

    if (hoverInfo) {
      const clampSigma = Math.min(c.sigma, 4.0);
      hoverInfo.textContent = `Multiverse Volatility: ±${clampSigma.toFixed(2)} native (σ = ${c.sigma.toFixed(2)})`;
    }
  }

  private attachEvents(): void {
    const svgWrapper = this.container.querySelector('.timeline-svg-wrapper') as HTMLElement;
    const hoverInfo = this.container.querySelector('#timeline-hover-info');

    const handlePointerInteraction = (clientX: number, isCommit: boolean = false) => {
      if (!svgWrapper) return;
      const mvPoints = this.mvPoints;
      const points = this.points;
      const count = mvPoints ? mvPoints.length : points.length;
      if (!count) return;

      const rect = svgWrapper.getBoundingClientRect();
      const clickX = clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / rect.width));
      const targetIdx = Math.round(pct * (count - 1));

      if (hoverInfo) {
        if (mvPoints) {
          const p = mvPoints[targetIdx];
          if (p) {
            hoverInfo.textContent = `Ply ${p.ply + 1} (${p.moveSan}) · μ ${(p.mean >= 0 ? '+' : '') + p.mean.toFixed(2)} · σ ${p.sigma.toFixed(2)}`;
          }
        } else {
          const p = points[targetIdx];
          if (p) {
            hoverInfo.textContent = `Ply ${p.ply + 1} (${p.moveSan}) · Score: ${(p.score >= 0 ? '+' : '') + p.score.toFixed(2)}`;
          }
        }
      }

      if (isCommit && this.onSelectPlyCallback) {
        this.onSelectPlyCallback(targetIdx);
      }
    };

    svgWrapper?.addEventListener('mousemove', (e: MouseEvent) => {
      handlePointerInteraction(e.clientX, false);
    });

    svgWrapper?.addEventListener('click', (e: MouseEvent) => {
      handlePointerInteraction(e.clientX, true);
    });

    // Touch scrubber support
    let isTouchingTimeline = false;

    svgWrapper?.addEventListener('touchstart', (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isTouchingTimeline = true;
        handlePointerInteraction(e.touches[0].clientX, true);
      }
    }, { passive: true });

    svgWrapper?.addEventListener('touchmove', (e: TouchEvent) => {
      if (isTouchingTimeline && e.touches.length === 1) {
        if (e.cancelable) e.preventDefault();
        handlePointerInteraction(e.touches[0].clientX, true);
      }
    }, { passive: false });

    svgWrapper?.addEventListener('touchend', () => {
      isTouchingTimeline = false;
    });
  }

  private computeSingleClampSpan(): number {
    let minScore = -4.0;
    let maxScore = 4.0;
    this.points.forEach((p) => {
      if (p.score < minScore) minScore = p.score;
      if (p.score > maxScore) maxScore = p.score;
    });
    return Math.max(8.0, Math.max(Math.abs(minScore), Math.abs(maxScore)) * 2.0);
  }

  private computeMultiverseClampSpan(): number {
    let maxAbs = 0;
    this.mvPoints!.forEach((p) => {
      const sigma = Math.min(p.sigma, 4.0);
      const top = p.mean + sigma;
      const bottom = p.mean - sigma;
      const abs = Math.max(Math.abs(top), Math.abs(bottom));
      if (abs > maxAbs) maxAbs = abs;
    });
    return Math.max(8.0, maxAbs * 2.0);
  }

  private buildMultiverseCoords(): { clampSpan: number; coords: MultiverseCoord[] } {
    const width = 800;
    const height = 90;
    const padX = 20;
    const padY = 12;
    const clampSpan = this.computeMultiverseClampSpan();
    const n = this.mvPoints!.length;
    const coords: MultiverseCoord[] = this.mvPoints!.map((p, i) => {
      const x = padX + (i / Math.max(1, n - 1)) * (width - 2 * padX);
      const y = this.yFor(p.mean, clampSpan, height, padY);
      return { x, y, ply: p.ply, moveSan: p.moveSan, mean: p.mean, sigma: p.sigma, spaghetti: p.spaghetti };
    });
    return { clampSpan, coords };
  }

  private yFor(score: number, clampSpan: number, height: number, padY: number): number {
    const zeroY = height / 2.0;
    const norm = Math.max(-1, Math.min(1, score / (clampSpan / 2.0)));
    return zeroY - norm * (height / 2.0 - padY);
  }
}
