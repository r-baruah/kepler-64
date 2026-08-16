/**
 * Kepler-64 Social GIF & Clip Export Studio
 * Generates high-resolution Observatory HUD GIFs & clips ready for Reddit / Hacker News.
 */

// @ts-ignore
import gifshot from 'gifshot';
import { KeplerBoard } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { BannerRenderer } from '../render/BannerRenderer';
import type { TrajectoryPoint } from '../render/BannerRenderer';
import { evaluatePosition } from '../core/evaluate';
import type { PresetGame } from '../core/presets';
import { Chess } from 'chess.js';

export class ExportModal {
  private container: HTMLElement;
  private game: PresetGame;
  private config: ConstantsConfig;
  private isRendering = false;

  constructor(game: PresetGame, config: ConstantsConfig) {
    this.game = game;
    this.config = config;
    this.container = document.createElement('div');
    this.container.className = 'modal-backdrop';
    this.container.style.display = 'none';
    document.body.appendChild(this.container);

    this.renderModalStructure();
  }

  public open(currentGame: PresetGame, currentConfig: ConstantsConfig): void {
    this.game = currentGame;
    this.config = currentConfig;
    this.container.style.display = 'flex';
    this.updatePreview();
  }

  public close(): void {
    this.container.style.display = 'none';
  }

  private esc(value: string): string {
    return value.replace(/[&<>"']/g, (ch) => {
      switch (ch) {
        case '&': return '&amp;';
        case '<': return '&lt;';
        case '>': return '&gt;';
        case '"': return '&quot;';
        case "'": return '&#39;';
        default: return ch;
      }
    });
  }

  private renderModalStructure(): void {
    this.container.innerHTML = `
      <div class="modal-content">
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:var(--space-md); border-bottom:var(--rule-thin) solid var(--color-rule); padding-bottom:var(--space-sm);">
          <div>
            <span class="badge-tag">SOCIAL OBSERVATORY STUDIO</span>
            <h3 style="font-size:1.4rem;">Export Master Observatory GIF</h3>
          </div>
          <button id="modal-close-btn" class="action-secondary" style="padding:4px 8px;">✕ Close</button>
        </div>

        <p style="font-size:0.88rem; color:var(--color-muted); margin-bottom:var(--space-md);">
          Renders a complete, professional widescreen Observatory HUD (Board + Vertical Barometer + Live Telemetry + Panoramic Trajectory Wave + Watermark) for <strong>${this.esc(this.game.title)}</strong>, formatted for Hacker News, Reddit, and Twitter/X.
        </p>

        <div class="export-options-grid">
          <div class="slider-group">
            <span class="slider-header">Export Format</span>
            <select id="export-layout" style="padding:8px; font-family:var(--font-mono); border:var(--rule-thin) solid var(--color-ink);">
              <option value="wide">Master Observatory HUD (840×520 HD)</option>
              <option value="compact">Compact HUD (700×430)</option>
            </select>
          </div>

          <div class="slider-group">
            <span class="slider-header">Move Speed / Duration</span>
            <select id="export-speed" style="padding:8px; font-family:var(--font-mono); border:var(--rule-thin) solid var(--color-ink);">
              <option value="0.8">0.8s / move (Standard Social)</option>
              <option value="0.5">0.5s / move (Fast Recap)</option>
              <option value="1.2">1.2s / move (Cinematic Analysis)</option>
            </select>
          </div>
        </div>

        <div id="export-status" style="margin-bottom:var(--space-md); font-family:var(--font-mono); font-size:0.8rem; display:none;">
          <div style="background:#e2e8f0; height:6px; margin-bottom:6px; overflow:hidden;">
            <div id="export-progress-bar" style="width:0%; height:100%; background:var(--color-accent); transition:width 0.2s ease;"></div>
          </div>
          <span id="export-status-text">Preloading piece assets...</span>
        </div>

        <div id="export-result-container" style="margin-bottom:var(--space-md); text-align:center; display:none;">
          <img id="export-result-img" style="max-width:100%; border:var(--rule-thin) solid var(--color-ink); max-height:280px;" />
        </div>

        <div class="modal-actions">
          <button id="render-btn" class="action-primary">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            Generate & Download GIF
          </button>
        </div>
      </div>
    `;

    this.container.querySelector('#modal-close-btn')?.addEventListener('click', () => this.close());
    this.container.addEventListener('click', (e) => {
      if (e.target === this.container) this.close();
    });

    this.container.querySelector('#render-btn')?.addEventListener('click', () => this.executeExport());
  }

  private updatePreview(): void {
    const statusBox = this.container.querySelector('#export-status') as HTMLElement;
    const resultBox = this.container.querySelector('#export-result-container') as HTMLElement;
    if (statusBox) statusBox.style.display = 'none';
    if (resultBox) resultBox.style.display = 'none';
  }

  private async executeExport(): Promise<void> {
    if (this.isRendering) return;
    this.isRendering = true;

    const statusBox = this.container.querySelector('#export-status') as HTMLElement;
    const progressBar = this.container.querySelector('#export-progress-bar') as HTMLElement;
    const statusText = this.container.querySelector('#export-status-text') as HTMLElement;
    const resultBox = this.container.querySelector('#export-result-container') as HTMLElement;
    const resultImg = this.container.querySelector('#export-result-img') as HTMLImageElement;
    const renderBtn = this.container.querySelector('#render-btn') as HTMLButtonElement;

    statusBox.style.display = 'block';
    renderBtn.disabled = true;
    statusText.textContent = 'Preloading vector chess pieces...';

    try {
      // 1. Ensure all piece SVG assets are 100% preloaded in memory
      await BannerRenderer.loadPieceImages();

      const chess = new Chess();
      chess.loadPgn(this.game.pgn);
      const moves = chess.history({ verbose: true });

      // Precompute game trajectory points
      const trajectoryBoard = new KeplerBoard();
      const trajectoryChess = new Chess();
      const trajectoryPoints: TrajectoryPoint[] = [];

      for (let i = 0; i < moves.length; i++) {
        const m = moves[i];
        trajectoryChess.move(m);
        trajectoryBoard.loadFen(trajectoryChess.fen());
        const evalRes = evaluatePosition(trajectoryBoard, this.config);
        trajectoryPoints.push({
          ply: i,
          score: evalRes.totalScoreWhite,
          moveSan: m.san,
        });
      }

      const layoutSelect = this.container.querySelector('#export-layout') as HTMLSelectElement;
      const isHD = (layoutSelect?.value || 'wide') === 'wide';

      const canvasWidth = isHD ? 840 : 700;
      const canvasHeight = isHD ? 520 : 430;

      const offCanvas = document.createElement('canvas');
      offCanvas.width = canvasWidth;
      offCanvas.height = canvasHeight;

      const replayBoard = new KeplerBoard();
      replayBoard.loadFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');

      const frameUrls: string[] = [];
      const replayChess = new Chess();

      // Render moves up to 24 plies for snappy GIF generation
      const totalPlies = Math.min(moves.length, 24);

      for (let i = 0; i < totalPlies; i++) {
        const move = moves[i];
        replayChess.move(move);
        replayBoard.loadFen(replayChess.fen());

        const lastMoveObj = {
          from: (move.from.charCodeAt(1) - 49) * 8 + (move.from.charCodeAt(0) - 97),
          to: (move.to.charCodeAt(1) - 49) * 8 + (move.to.charCodeAt(0) - 97),
        };

        const moveSan = `${Math.floor(i / 2) + 1}. ${move.color === 'b' ? '...' : ''}${move.san}`;

        BannerRenderer.renderFrame(
          offCanvas,
          replayBoard,
          this.config,
          this.game.title,
          moveSan,
          i,
          moves.length,
          trajectoryPoints,
          lastMoveObj
        );

        frameUrls.push(offCanvas.toDataURL('image/png'));

        const pct = Math.round(((i + 1) / totalPlies) * 45);
        progressBar.style.width = `${pct}%`;
        statusText.textContent = `Rendered composite frame ${i + 1} of ${totalPlies}...`;
      }

      statusText.textContent = 'Encoding animated GIF with gifshot...';
      const speedSelect = this.container.querySelector('#export-speed') as HTMLSelectElement;
      const intervalSec = parseFloat(speedSelect?.value || '0.8');

      gifshot.createGIF(
        {
          images: frameUrls,
          interval: intervalSec,
          gifWidth: canvasWidth,
          gifHeight: canvasHeight,
          progressCallback: (captureProgress: number) => {
            const pct = 45 + Math.round(captureProgress * 55);
            progressBar.style.width = `${pct}%`;
          },
        },
        (obj: { error: boolean; image: string }) => {
          if (!obj.error) {
            resultImg.src = obj.image;
            resultBox.style.display = 'block';
            statusText.textContent = '✓ High-fidelity GIF generated successfully!';

            // Download
            const a = document.createElement('a');
            a.href = obj.image;
            a.download = `kepler64_${this.game.id}_observatory.gif`;
            a.click();
          } else {
            statusText.textContent = 'Error assembling GIF.';
          }
          this.isRendering = false;
          renderBtn.disabled = false;
        }
      );
    } catch (err) {
      console.error(err);
      statusText.textContent = 'Render failed: ' + String(err);
      this.isRendering = false;
      renderBtn.disabled = false;
    }
  }
}
