/**
 * Kepler-64 Panoramic Game Timeline & Evaluation Barometer
 * Renders an interactive, annotated full-width gravitational trajectory.
 */

export interface SparklinePoint {
  ply: number;
  score: number; // White perspective
  moveSan: string;
}

export class EvalSparkline {
  private container: HTMLElement;
  private points: SparklinePoint[] = [];
  private currentPly: number = 0;
  private onSelectPlyCallback?: (ply: number) => void;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public setData(points: SparklinePoint[], currentPly: number): void {
    this.points = points;
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
    if (!this.points.length) {
      this.container.innerHTML = '';
      return;
    }

    const width = 800;
    const height = 90;
    const padX = 20;
    const padY = 12;

    // Find min and max scores
    let minScore = -4.0;
    let maxScore = 4.0;
    this.points.forEach((p) => {
      if (p.score < minScore) minScore = p.score;
      if (p.score > maxScore) maxScore = p.score;
    });

    const clampSpan = Math.max(8.0, Math.max(Math.abs(minScore), Math.abs(maxScore)) * 2.0);
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
      const norm = Math.max(-1, Math.min(1, p.score / (clampSpan / 2.0)));
      const y = zeroY - norm * (height / 2.0 - padY);
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
            <defs>
              <linearGradient id="gradWhite" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="var(--color-plate)" stop-opacity="0.35" />
                <stop offset="100%" stop-color="var(--color-plate)" stop-opacity="0.0" />
              </linearGradient>
              <linearGradient id="gradBlack" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="var(--color-accent)" stop-opacity="0.0" />
                <stop offset="100%" stop-color="var(--color-accent)" stop-opacity="0.35" />
              </linearGradient>
            </defs>

            <!-- Zero Equilibrium Axis -->
            <line x1="${padX}" y1="${zeroY}" x2="${width - padX}" y2="${zeroY}" stroke="var(--color-rule)" stroke-width="1.2" stroke-dasharray="4,4" />

            <!-- Shaded Advantage Areas -->
            <path d="${areaWhiteD.join(' ')}" fill="url(#gradWhite)" />
            <path d="${areaBlackD.join(' ')}" fill="url(#gradBlack)" />

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

  private updateMarker(): void {
    const cursorLine = this.container.querySelector('#timeline-cursor-line') as SVGLineElement;
    const cursorDot = this.container.querySelector('#timeline-cursor-dot') as SVGCircleElement;
    const hoverInfo = this.container.querySelector('#timeline-hover-info') as HTMLElement;
    if (!cursorDot || !this.points.length) return;

    const width = 800;
    const height = 90;
    const padX = 20;
    const padY = 12;
    const clampSpan = 8.0;
    const zeroY = height / 2.0;

    const n = this.points.length;
    const i = Math.max(0, Math.min(n - 1, this.currentPly));
    const p = this.points[i];
    if (!p) return;

    const x = padX + (i / Math.max(1, n - 1)) * (width - 2 * padX);
    const norm = Math.max(-1, Math.min(1, p.score / (clampSpan / 2.0)));
    const y = zeroY - norm * (height / 2.0 - padY);

    cursorLine?.setAttribute('x1', x.toFixed(1));
    cursorLine?.setAttribute('x2', x.toFixed(1));
    cursorDot.setAttribute('cx', x.toFixed(1));
    cursorDot.setAttribute('cy', y.toFixed(1));

    if (hoverInfo) {
      hoverInfo.textContent = `Move: ${p.moveSan} · Score: ${(p.score >= 0 ? '+' : '') + p.score.toFixed(2)}`;
    }
  }

  private attachEvents(): void {
    const svgWrapper = this.container.querySelector('.timeline-svg-wrapper');
    const hoverInfo = this.container.querySelector('#timeline-hover-info');

    svgWrapper?.addEventListener('mousemove', (e: any) => {
      const rect = svgWrapper.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / rect.width));
      const targetIdx = Math.round(pct * (this.points.length - 1));
      const p = this.points[targetIdx];
      if (p && hoverInfo) {
        hoverInfo.textContent = `Ply ${p.ply + 1} (${p.moveSan}) · Score: ${(p.score >= 0 ? '+' : '') + p.score.toFixed(2)}`;
      }
    });

    svgWrapper?.addEventListener('click', (e: any) => {
      const rect = svgWrapper.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / rect.width));
      const targetIdx = Math.round(pct * (this.points.length - 1));
      if (this.onSelectPlyCallback) {
        this.onSelectPlyCallback(targetIdx);
      }
    });
  }
}
