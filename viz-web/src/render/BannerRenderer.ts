/**
 * Kepler-64 Composite Social Banner & GIF Frame Renderer
 * Renders a full high-fidelity Observatory HUD (Board + Vertical Barometer +
 * Live Telemetry + Candidate Move Matrix + Panoramic Trajectory Wave + Watermark)
 * onto a single canvas for social sharing.
 */

import { Chess } from 'chess.js';
import { KeplerBoard } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { potentialOnGrid } from '../core/gravity';
import { evaluatePosition } from '../core/evaluate';
import { generateContourLines } from './ContourRenderer';
import { drawTidalStressEllipse } from './TidalRenderer';

export interface TrajectoryPoint {
  ply: number;
  score: number;
  moveSan: string;
}

export class BannerRenderer {
  private static pieceCache: Map<string, HTMLImageElement> = new Map();
  private static preloadPromise: Promise<Map<string, HTMLImageElement>> | null = null;

  public static async loadPieceImages(): Promise<Map<string, HTMLImageElement>> {
    if (this.pieceCache.size === 12) return this.pieceCache;
    if (this.preloadPromise) return this.preloadPromise;

    const pieces = [
      'bb', 'bk', 'bn', 'bp', 'bq', 'br',
      'wb', 'wk', 'wn', 'wp', 'wq', 'wr',
    ];

    const base = (import.meta.env.BASE_URL || './').replace(/\/$/, '') + '/';
    this.preloadPromise = new Promise((resolve) => {
      let loaded = 0;
      pieces.forEach((name) => {
        const img = new Image();
        img.src = `${base}pieces/${name}.svg`;
        img.onload = () => {
          this.pieceCache.set(name, img);
          loaded++;
          if (loaded === pieces.length) {
            resolve(this.pieceCache);
          }
        };
        img.onerror = () => {
          loaded++;
          if (loaded === pieces.length) resolve(this.pieceCache);
        };
      });
    });

    return this.preloadPromise;
  }

  public static renderFrame(
    canvas: HTMLCanvasElement,
    board: KeplerBoard,
    config: ConstantsConfig,
    gameTitle: string,
    moveSan: string,
    plyNum: number,
    totalPlies: number,
    trajectoryPoints: TrajectoryPoint[],
    lastMove: { from: number; to: number } | null = null
  ): void {
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Background paper
    ctx.fillStyle = '#ebf1f6';
    ctx.fillRect(0, 0, width, height);

    const pad = 14;
    const topHeight = height - 104; // Top section for Board + Telemetry

    // 1. Draw Vertical Gravitational Barometer
    const barometerX = pad;
    const barometerY = pad;
    const barometerW = 16;
    const barometerH = topHeight - 2 * pad;
    this.drawVerticalBarometer(ctx, board, config, barometerX, barometerY, barometerW, barometerH);

    // 2. Draw Board
    const boardX = barometerX + barometerW + 10;
    const boardY = pad;
    const boardSize = barometerH;
    this.drawBoard(ctx, board, config, boardX, boardY, boardSize, lastMove);

    // 3. Draw Telemetry HUD + Candidate Moves Matrix
    const hudX = boardX + boardSize + 14;
    const hudY = pad;
    const hudWidth = width - hudX - pad;
    const hudHeight = boardSize;
    this.drawTelemetryHUD(ctx, board, config, hudX, hudY, hudWidth, hudHeight, gameTitle, moveSan, plyNum, totalPlies);

    // 4. Draw Panoramic Gravitational Trajectory Strip
    const timelineX = pad;
    const timelineY = topHeight - 6;
    const timelineW = width - 2 * pad;
    const timelineH = 74;
    this.drawTrajectoryWave(ctx, timelineX, timelineY, timelineW, timelineH, trajectoryPoints, plyNum);

    // 5. Draw Watermark & Attribution Footer
    const footerY = height - 8;
    ctx.font = '600 9px Sometype Mono, monospace';
    ctx.fillStyle = '#64748b';
    ctx.fillText('KEPLER-64 · Differentiable Astrophysical Chess · Created by Ripuranjan Baruah', pad, footerY);

    ctx.textAlign = 'right';
    ctx.fillStyle = '#2448b8';
    ctx.font = '700 9px Sometype Mono, monospace';
    ctx.fillText('github.com/r-baruah/kepler-64', width - pad, footerY);
    ctx.textAlign = 'left';
  }

  private static drawVerticalBarometer(
    ctx: CanvasRenderingContext2D,
    board: KeplerBoard,
    config: ConstantsConfig,
    bx: number,
    by: number,
    bw: number,
    bh: number
  ): void {
    const breakdown = evaluatePosition(board, config);
    const scoreW = breakdown.totalScoreWhite;
    const pct = Math.max(5, Math.min(95, 50 + (scoreW / 8.0) * 50));

    ctx.save();
    ctx.translate(bx, by);

    // Track
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, bw, bh);
    ctx.strokeStyle = '#172235';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(0, 0, bw, bh);

    // Liquid Fill
    const fillH = (bh * pct) / 100;
    ctx.fillStyle = scoreW >= 0 ? '#2448b8' : '#f06426';
    ctx.fillRect(1, bh - fillH, bw - 2, fillH);

    // Center Zero tick
    ctx.strokeStyle = '#172235';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, bh / 2);
    ctx.lineTo(bw, bh / 2);
    ctx.stroke();

    ctx.restore();
  }

  private static drawBoard(
    ctx: CanvasRenderingContext2D,
    board: KeplerBoard,
    config: ConstantsConfig,
    bx: number,
    by: number,
    bsize: number,
    lastMove: { from: number; to: number } | null
  ): void {
    ctx.save();
    ctx.translate(bx, by);

    // Outer Board Container
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, bsize, bsize);
    ctx.strokeStyle = '#172235';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, bsize, bsize);

    // CLIPPING: Keep all contours & tidal ellipses strictly within board boundaries
    ctx.beginPath();
    ctx.rect(0, 0, bsize, bsize);
    ctx.clip();

    const sqSize = bsize / 8.0;

    // Squares
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const isLight = (r + f) % 2 === 0;
        ctx.fillStyle = isLight ? '#e9eff4' : '#b8c9d9';
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    // Last move highlight
    if (lastMove) {
      ctx.fillStyle = 'rgba(240, 100, 38, 0.35)';
      for (const sq of [lastMove.from, lastMove.to]) {
        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    // Continuous Potential Heatmap & Contours
    const masses = board.massVector();
    const { grid, n, p3, p97 } = potentialOnGrid(masses, config.eps, config.G, config.c, 48);
    const span = p97 - p3 || 1.0;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = n;
    offCanvas.height = n;
    const offCtx = offCanvas.getContext('2d');
    if (offCtx) {
      const imgData = offCtx.createImageData(n, n);
      const data = imgData.data;

      for (let gy = 0; gy < n; gy++) {
        const invGy = n - 1 - gy;
        for (let gx = 0; gx < n; gx++) {
          const val = grid[invGy * n + gx];
          const t = Math.max(0, Math.min(1, (val - p3) / span));

          const r = Math.round(18 + t * (230 - 18));
          const g = Math.round(26 + t * (140 - 26));
          const b = Math.round(75 + t * (45 - 75));
          const alpha = Math.round(35 + t * 45);

          const idx = (gy * n + gx) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = alpha;
        }
      }
      offCtx.putImageData(imgData, 0, 0);
      ctx.drawImage(offCanvas, 0, 0, bsize, bsize);
    }

    // Contours
    const levels: number[] = [];
    for (let k = 1; k < 12; k++) {
      levels.push(p3 + (span * k) / 12);
    }
    const contours = generateContourLines(grid, n, levels);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = 1.2;
    contours.forEach((segments) => {
      ctx.beginPath();
      segments.forEach((seg) => {
        const x1 = ((seg.x1 + 0.5) / 8.0) * bsize;
        const y1 = bsize - ((seg.y1 + 0.5) / 8.0) * bsize;
        const x2 = ((seg.x2 + 0.5) / 8.0) * bsize;
        const y2 = bsize - ((seg.y2 + 0.5) / 8.0) * bsize;
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      });
      ctx.stroke();
    });

    // Tidal Tensors
    const breakdown = evaluatePosition(board, config);
    const wk = board.findKingSquare('w');
    const bk = board.findKingSquare('b');

    if (wk !== null && breakdown.whiteKingTidal) {
      const kx = (wk % 8 + 0.5) * sqSize;
      const ky = (7 - Math.floor(wk / 8) + 0.5) * sqSize;
      drawTidalStressEllipse(ctx, kx, ky, sqSize, breakdown.whiteKingTidal);
    }
    if (bk !== null && breakdown.blackKingTidal) {
      const kx = (bk % 8 + 0.5) * sqSize;
      const ky = (7 - Math.floor(bk / 8) + 0.5) * sqSize;
      drawTidalStressEllipse(ctx, kx, ky, sqSize, breakdown.blackKingTidal);
    }

    // Pieces
    for (let sq = 0; sq < 64; sq++) {
      const piece = board.squares[sq];
      if (!piece) continue;
      const f = sq % 8;
      const r = Math.floor(sq / 8);
      const px = f * sqSize;
      const py = (7 - r) * sqSize;

      const imgKey = `${piece.color}${piece.type}`;
      const img = this.pieceCache.get(imgKey);
      if (img) {
        const pad = sqSize * 0.08;
        ctx.drawImage(img, px + pad, py + pad, sqSize - 2 * pad, sqSize - 2 * pad);
      }
    }

    // Board Coordinate labels (a-h, 1-8)
    ctx.font = '600 8px Sometype Mono, monospace';
    ctx.fillStyle = 'rgba(23, 34, 53, 0.6)';
    for (let f = 0; f < 8; f++) {
      ctx.fillText(String.fromCharCode(97 + f), f * sqSize + 2, bsize - 2);
    }
    for (let r = 0; r < 8; r++) {
      ctx.fillText((r + 1).toString(), bsize - 8, (7 - r) * sqSize + 9);
    }

    ctx.restore();
  }

  private static drawTelemetryHUD(
    ctx: CanvasRenderingContext2D,
    board: KeplerBoard,
    config: ConstantsConfig,
    hx: number,
    hy: number,
    hwidth: number,
    hheight: number,
    gameTitle: string,
    moveSan: string,
    plyNum: number,
    totalPlies: number
  ): void {
    const breakdown = evaluatePosition(board, config);

    ctx.save();
    ctx.translate(hx, hy);

    // Card background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, hwidth, hheight);
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, hwidth, hheight);

    const pad = 12;
    let currY = pad + 8;

    // Header brand & URL
    ctx.font = '700 10px Sometype Mono, monospace';
    ctx.fillStyle = '#2448b8';
    ctx.fillText('KEPLER-64 // ROCHE ENGINE', pad, currY);

    ctx.textAlign = 'right';
    ctx.font = '600 9px Sometype Mono, monospace';
    ctx.fillStyle = '#64748b';
    ctx.fillText(`Ply ${plyNum + 1}/${totalPlies}`, hwidth - pad, currY);
    ctx.textAlign = 'left';

    currY += 15;
    ctx.font = '600 10px Familjen Grotesk, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.fillText(gameTitle, pad, currY);

    currY += 22;
    // Big Move Text
    ctx.font = '700 18px Familjen Grotesk, sans-serif';
    ctx.fillStyle = '#172235';
    ctx.fillText(moveSan, pad, currY);

    // Status Badge
    const isDisrupted = (breakdown.blackKingTidal?.isDisrupted || breakdown.whiteKingTidal?.isDisrupted);
    const statusText = isDisrupted ? '⚠ ROCHE DISRUPTION' : '● TIDAL EQUILIBRIUM';
    ctx.font = '700 9px Sometype Mono, monospace';
    ctx.fillStyle = isDisrupted ? '#b91c1c' : '#1e6091';
    ctx.textAlign = 'right';
    ctx.fillText(statusText, hwidth - pad, currY - 3);
    ctx.textAlign = 'left';

    currY += 15;
    // Roche Disruption Gauge
    const activeKingTidal = board.turn === 'w' ? breakdown.blackKingTidal : breakdown.whiteKingTidal;
    const etaVal = activeKingTidal ? activeKingTidal.eta.toFixed(3) : '0.000';
    const etaRatio = activeKingTidal ? Math.min(100, Math.round((activeKingTidal.eta / config.roche) * 100)) : 0;

    ctx.font = '600 9px Sometype Mono, monospace';
    ctx.fillStyle = '#172235';
    ctx.fillText(`Roche Disruption (η = ${etaVal} / ρ = ${config.roche.toFixed(2)})`, pad, currY);

    currY += 4;
    const trackW = hwidth - 2 * pad;
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(pad, currY, trackW, 4);
    ctx.fillStyle = isDisrupted ? '#dc2626' : '#2448b8';
    ctx.fillRect(pad, currY, (trackW * etaRatio) / 100, 4);

    currY += 16;
    // Waterfall Ledger
    ctx.font = '700 9px Familjen Grotesk, sans-serif';
    ctx.fillStyle = '#172235';
    ctx.fillText('EVALUATION DECOMPOSITION', pad, currY);

    currY += 11;
    const rows = [
      { name: 'Enemy King Tide', val: breakdown.enemyKingTide, max: 2.5 },
      { name: 'Own King Tide', val: breakdown.ownKingTide, max: 2.5 },
      { name: 'Disruption Force', val: breakdown.forceEnemyKing, max: 50.0 },
      { name: 'Internal Binding', val: breakdown.bindingEnergy, max: 1.5 },
      { name: 'Material Edge', val: breakdown.materialBalance, max: 6.0 },
    ];

    rows.forEach((r) => {
      ctx.font = '500 8.5px Sometype Mono, monospace';
      ctx.fillStyle = '#475569';
      ctx.fillText(r.name, pad, currY);

      const valStr = (r.val >= 0 ? '+' : '') + r.val.toFixed(2);
      ctx.textAlign = 'right';
      ctx.fillText(valStr, hwidth - pad, currY);
      ctx.textAlign = 'left';

      // Bar
      const barX = pad + 95;
      const barW = hwidth - 2 * pad - 140;
      ctx.fillStyle = '#e2e8f0';
      ctx.fillRect(barX, currY - 6, barW, 3.5);

      const fillW = Math.min(barW, Math.round((Math.abs(r.val) / r.max) * barW));
      ctx.fillStyle = r.val >= 0 ? '#16a34a' : '#dc2626';
      ctx.fillRect(barX, currY - 6, fillW, 3.5);

      currY += 13;
    });

    currY += 2;
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, currY);
    ctx.lineTo(hwidth - pad, currY);
    ctx.stroke();

    currY += 14;
    ctx.font = '700 9.5px Sometype Mono, monospace';
    ctx.fillStyle = '#172235';
    ctx.fillText('NET POSITION SCORE:', pad, currY);

    ctx.textAlign = 'right';
    ctx.font = '700 12px Sometype Mono, monospace';
    ctx.fillStyle = '#2448b8';
    ctx.fillText(`${(breakdown.totalScoreMover >= 0 ? '+' : '') + breakdown.totalScoreMover.toFixed(2)} native`, hwidth - pad, currY);
    ctx.textAlign = 'left';

    // 5. CANDIDATE MOVES MATRIX (BELOW NET SCORE)
    currY += 14;
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(pad, currY - 10, hwidth - 2 * pad, 62);
    ctx.strokeStyle = '#e2e8f0';
    ctx.strokeRect(pad, currY - 10, hwidth - 2 * pad, 62);

    ctx.font = '700 8.5px Sometype Mono, monospace';
    ctx.fillStyle = '#64748b';
    ctx.fillText('TOP CANDIDATE MOVES (EVAL)', pad + 6, currY);

    const tempChess = new Chess(board.toFen());
    const legalMoves = tempChess.moves({ verbose: true });
    const moveEvaluations: { san: string; score: number }[] = [];
    const evalBoard = new KeplerBoard();

    const sampleMoves = legalMoves.slice(0, 3);
    sampleMoves.forEach((m) => {
      tempChess.move(m);
      evalBoard.loadFen(tempChess.fen());
      const res = evaluatePosition(evalBoard, config);
      moveEvaluations.push({
        san: m.san,
        score: res.totalScoreMover,
      });
      tempChess.undo();
    });

    moveEvaluations.sort((a, b) => b.score - a.score);

    currY += 13;
    moveEvaluations.forEach((m, idx) => {
      ctx.font = '700 8.5px Sometype Mono, monospace';
      ctx.fillStyle = '#2448b8';
      ctx.fillText(`#${idx + 1} ${m.san}`, pad + 6, currY);

      ctx.textAlign = 'right';
      ctx.font = '600 8.5px Sometype Mono, monospace';
      ctx.fillStyle = '#172235';
      ctx.fillText(`${(m.score >= 0 ? '+' : '') + m.score.toFixed(2)}`, hwidth - pad - 6, currY);
      ctx.textAlign = 'left';

      currY += 12;
    });

    ctx.restore();
  }

  private static drawTrajectoryWave(
    ctx: CanvasRenderingContext2D,
    tx: number,
    ty: number,
    tw: number,
    th: number,
    points: TrajectoryPoint[],
    activePly: number
  ): void {
    if (!points || !points.length) return;

    ctx.save();
    ctx.translate(tx, ty);

    // Card background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, tw, th);
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, tw, th);

    // Label
    ctx.font = '700 8px Sometype Mono, monospace';
    ctx.fillStyle = '#2448b8';
    ctx.fillText('GRAVITATIONAL TRAJECTORY HORIZON', 10, 11);

    const padX = 12;
    const padY = 14;
    const waveH = th - padY - 5;
    const zeroY = padY + waveH / 2.0;

    let minScore = -4.0;
    let maxScore = 4.0;
    points.forEach((p) => {
      if (p.score < minScore) minScore = p.score;
      if (p.score > maxScore) maxScore = p.score;
    });

    const clampSpan = Math.max(8.0, Math.max(Math.abs(minScore), Math.abs(maxScore)) * 2.0);
    const n = points.length;
    const coords: { x: number; y: number }[] = [];

    for (let i = 0; i < n; i++) {
      const p = points[i];
      const x = padX + (i / Math.max(1, n - 1)) * (tw - 2 * padX);
      const norm = Math.max(-1, Math.min(1, p.score / (clampSpan / 2.0)));
      const y = zeroY - norm * (waveH / 2.0 - 3);
      coords.push({ x, y });
    }

    // Zero equilibrium line
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padX, zeroY);
    ctx.lineTo(tw - padX, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Area fills
    // White advantage (above zero)
    ctx.fillStyle = 'rgba(36, 72, 184, 0.18)';
    ctx.beginPath();
    ctx.moveTo(coords[0].x, zeroY);
    coords.forEach((c) => {
      ctx.lineTo(c.x, Math.min(zeroY, c.y));
    });
    ctx.lineTo(coords[coords.length - 1].x, zeroY);
    ctx.closePath();
    ctx.fill();

    // Black advantage (below zero)
    ctx.fillStyle = 'rgba(240, 100, 38, 0.18)';
    ctx.beginPath();
    ctx.moveTo(coords[0].x, zeroY);
    coords.forEach((c) => {
      ctx.lineTo(c.x, Math.max(zeroY, c.y));
    });
    ctx.lineTo(coords[coords.length - 1].x, zeroY);
    ctx.closePath();
    ctx.fill();

    // Trajectory wave stroke
    ctx.strokeStyle = '#172235';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    coords.forEach((c, idx) => {
      if (idx === 0) ctx.moveTo(c.x, c.y);
      else ctx.lineTo(c.x, c.y);
    });
    ctx.stroke();

    // Active Ply Indicator
    const activeCoord = coords[activePly] || coords[0];
    ctx.strokeStyle = '#f06426';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(activeCoord.x, padY);
    ctx.lineTo(activeCoord.x, padY + waveH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f06426';
    ctx.strokeStyle = '#172235';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(activeCoord.x, activeCoord.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }
}
