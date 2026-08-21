/**
 * Kepler-64 Composite Social Banner & GIF Frame Renderer
 * Renders a full high-fidelity Observatory HUD (Board + Vertical Barometer +
 * Live Telemetry + Candidate Move Matrix + Panoramic Trajectory Wave + Watermark)
 * onto a single canvas for social sharing.
 *
 * Layout mirrors the on-screen Observatory: board on the left, and the
 * gravitational trajectory + telemetry stacked on the right.
 */

import { Chess } from 'chess.js';
import { KeplerBoard } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { potentialOnGrid, forceField } from '../core/gravity';
import { evaluatePosition } from '../core/evaluate';
import { generateContourLines } from './ContourRenderer';
import { drawTidalStressEllipse } from './TidalRenderer';

export interface TrajectoryPoint {
  ply: number;
  score: number;
  moveSan: string;
}

export interface BannerRenderOptions {
  showHeatmap?: boolean;
  showContours?: boolean;
  showVectors?: boolean;
  showTidalStress?: boolean;
  showAccretion?: boolean;
  showWavefronts?: boolean;
  showLorentz?: boolean;
  showLagrange?: boolean;
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
    lastMove: { from: number; to: number } | null = null,
    options: BannerRenderOptions = {}
  ): void {
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Background paper
    ctx.fillStyle = '#f5f4ef';
    ctx.fillRect(0, 0, width, height);

    const isHD = width >= 800;
    const padX = isHD ? 14 : 12;
    const padTop = isHD ? 12 : 10;
    const footerHeight = isHD ? 26 : 22;
    const gapFooter = isHD ? 8 : 6;

    // Calculate available height for the board, barometer & telemetry columns
    const boardSize = height - padTop - footerHeight - gapFooter;

    // 1. Vertical Gravitational Barometer
    const barometerX = padX;
    const barometerY = padTop;
    const barometerW = isHD ? 16 : 14;
    const barometerH = boardSize;
    this.drawVerticalBarometer(ctx, board, config, barometerX, barometerY, barometerW, barometerH);

    // 2. Board
    const boardX = barometerX + barometerW + (isHD ? 10 : 8);
    const boardY = padTop;
    this.drawBoard(ctx, board, config, boardX, boardY, boardSize, lastMove, options);

    // 3. Right column: Trajectory (top) + Telemetry (bottom)
    const rightX = boardX + boardSize + (isHD ? 12 : 10);
    const rightW = width - rightX - padX;

    const trajectoryY = padTop;
    const trajectoryH = isHD ? 72 : 62;
    this.drawTrajectoryWave(ctx, rightX, trajectoryY, rightW, trajectoryH, trajectoryPoints, plyNum);

    const hudY = trajectoryY + trajectoryH + (isHD ? 8 : 6);
    const hudH = boardSize - trajectoryH - (isHD ? 8 : 6);
    this.drawTelemetryHUD(ctx, board, config, rightX, hudY, rightW, hudH, gameTitle, moveSan, plyNum, totalPlies);

    // 4. Dedicated Watermark & Attribution Footer Bar
    const footerDividerY = height - footerHeight;
    ctx.strokeStyle = '#dcdad2';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padX, footerDividerY);
    ctx.lineTo(width - padX, footerDividerY);
    ctx.stroke();

    const footerTextY = height - Math.round(footerHeight / 2) + 3.5;
    const fontSize = isHD ? '9.5px' : '8.5px';

    ctx.font = `600 ${fontSize} Sometype Mono, monospace`;
    ctx.fillStyle = '#62615a';
    ctx.fillText('KEPLER-64 · Differentiable Astrophysical Chess · Created by Ripuranjan Baruah', padX, footerTextY);

    ctx.textAlign = 'right';
    ctx.fillStyle = '#0069ff';
    ctx.font = `700 ${fontSize} Sometype Mono, monospace`;
    ctx.fillText('github.com/r-baruah/kepler-64', width - padX, footerTextY);
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
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(0, 0, bw, bh);

    // Liquid Fill
    const fillH = (bh * pct) / 100;
    ctx.fillStyle = scoreW >= 0 ? '#0069ff' : '#ff5a00';
    ctx.fillRect(1, bh - fillH, bw - 2, fillH);

    // Center Zero tick
    ctx.strokeStyle = '#111111';
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
    lastMove: { from: number; to: number } | null,
    options: BannerRenderOptions = {}
  ): void {
    ctx.save();
    ctx.translate(bx, by);

    // Outer Board Container
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, bsize, bsize);
    ctx.strokeStyle = '#111111';
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
      ctx.fillStyle = 'rgba(255, 90, 0, 0.30)';
      for (const sq of [lastMove.from, lastMove.to]) {
        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    // Continuous Potential Heatmap & Contours
    const showHeatmap = options.showHeatmap !== false;
    const showContours = options.showContours !== false;
    const showVectors = !!options.showVectors;
    const showTidalStress = options.showTidalStress !== false;
    const showAccretion = !!options.showAccretion;
    const showWavefronts = !!options.showWavefronts;
    const showLorentz = !!options.showLorentz;
    const showLagrange = !!options.showLagrange;

    const masses = board.massVector();

    if (showHeatmap || showContours) {
      const { grid, n, p3, p97 } = potentialOnGrid(masses, config.eps, config.G, config.c, 48);
      const span = p97 - p3 || 1.0;

      if (showHeatmap) {
        const offCanvas = document.createElement('canvas');
        offCanvas.width = n;
        offCanvas.height = n;
        const offCtx = offCanvas.getContext('2d');
        if (offCtx) {
          const imgData = offCtx.createImageData(n, n);
          const data = imgData.data;

          const HEAT_PALETTE: ReadonlyArray<readonly [number, number, number]> = [
            [18, 26, 75],
            [58, 45, 86],
            [98, 66, 88],
            [140, 88, 77],
            [185, 112, 60],
            [230, 140, 45],
          ];
          for (let gy = 0; gy < n; gy++) {
            const invGy = n - 1 - gy;
            for (let gx = 0; gx < n; gx++) {
              const val = grid[invGy * n + gx];
              const t = Math.max(0, Math.min(1, (val - p3) / span));
              const band = Math.min(5, Math.floor(t * 6));
              const [r, g, b] = HEAT_PALETTE[band];
              const alpha = 35 + band * 9;

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
      }

      if (showContours) {
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
      }
    }

    // Force Vectors Streamlines
    if (showVectors) {
      const forces = forceField(masses, config.eps, config.G, config.c);
      ctx.save();
      ctx.strokeStyle = 'rgba(240, 100, 38, 0.70)';
      ctx.fillStyle = 'rgba(240, 100, 38, 0.70)';
      ctx.lineWidth = 1.3;

      for (let sq = 0; sq < 64; sq++) {
        const fx = forces.fx[sq];
        const fy = forces.fy[sq];
        const mag = Math.sqrt(fx * fx + fy * fy);
        if (mag < 0.05) continue;

        const f = sq % 8;
        const r = Math.floor(sq / 8);
        const cx = (f + 0.5) * sqSize;
        const cy = (7 - r + 0.5) * sqSize;

        const normX = fx / (mag + 1e-6);
        const normY = fy / (mag + 1e-6);
        const len = Math.min(sqSize * 0.40, sqSize * 0.15 * Math.log10(1 + mag * 5));

        const endX = cx + normX * len;
        const endY = cy - normY * len;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        const angle = Math.atan2(-normY, normX);
        const head = 3.5;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - head * Math.cos(angle - Math.PI / 6), endY - head * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(endX - head * Math.cos(angle + Math.PI / 6), endY - head * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }

    // Retarded Light Cone Wavefronts
    if (showWavefronts && lastMove) {
      const of = lastMove.from % 8;
      const or = Math.floor(lastMove.from / 8);
      const cx = (of + 0.5) * sqSize;
      const cy = (7 - or + 0.5) * sqSize;
      const cSq = config.c;

      ctx.save();
      const ripples = [1, 0.62, 0.3];
      ripples.forEach((f, idx) => {
        ctx.beginPath();
        ctx.arc(cx, cy, cSq * f * sqSize, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 105, 255, ${0.55 - idx * 0.15})`;
        ctx.lineWidth = idx === 0 ? 1.6 : 1.2;
        ctx.setLineDash(idx === 0 ? [5, 4] : []);
        ctx.stroke();
      });
      ctx.restore();
    }

    // Accretion Halos
    if (showAccretion) {
      for (let sq = 0; sq < 64; sq++) {
        const p = board.squares[sq];
        if (!p) continue;
        const boost = board.massBoost[sq] || 0;
        if (boost > 0.05) {
          const f = sq % 8;
          const r = Math.floor(sq / 8);
          const cx = (f + 0.5) * sqSize;
          const cy = (7 - r + 0.5) * sqSize;
          ctx.save();
          const rad = sqSize * 0.44;
          const grad = ctx.createRadialGradient(cx, cy, sqSize * 0.25, cx, cy, rad);
          grad.addColorStop(0, 'rgba(255, 170, 0, 0.65)');
          grad.addColorStop(0.7, 'rgba(255, 120, 0, 0.30)');
          grad.addColorStop(1, 'rgba(255, 120, 0, 0)');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, rad, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = 'rgba(255, 180, 0, 0.85)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.restore();
        }
      }
    }

    // Lorentz Relativistic Dilation
    if (showLorentz) {
      for (let sq = 0; sq < 64; sq++) {
        const p = board.squares[sq];
        if (!p) continue;
        const boost = board.massBoost[sq] || 0;
        if (boost > 0) {
          const f = sq % 8;
          const r = Math.floor(sq / 8);
          const cx = (f + 0.5) * sqSize;
          const cy = (7 - r + 0.5) * sqSize;
          ctx.save();
          ctx.strokeStyle = '#0069ff';
          ctx.lineWidth = 1.8;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.arc(cx, cy, sqSize * 0.46, 0, Math.PI * 2);
          ctx.stroke();
          ctx.font = '700 8px Sometype Mono, monospace';
          ctx.fillStyle = '#0069ff';
          ctx.fillText(`γ`, cx + sqSize * 0.26, cy - sqSize * 0.26);
          ctx.restore();
        }
      }
    }

    // Lagrange Points L1-L5
    if (showLagrange) {
      const ranked: { sq: number; m: number }[] = [];
      for (let sq = 0; sq < 64; sq++) {
        if (board.squares[sq]) ranked.push({ sq, m: Math.abs(masses[sq]) });
      }
      ranked.sort((a, b) => b.m - a.m);
      if (ranked.length >= 2 && ranked[0].m > 0 && ranked[1].m > 0) {
        const A = ranked[0];
        const B = ranked[1];
        const ax = (A.sq % 8) + 0.5;
        const ay = Math.floor(A.sq / 8) + 0.5;
        const bx = (B.sq % 8) + 0.5;
        const by = Math.floor(B.sq / 8) + 0.5;
        const dx = bx - ax;
        const dy = by - ay;
        const d = Math.hypot(dx, dy) || 1;
        const ux = dx / d;
        const uy = dy / d;
        const mu = B.m / (A.m + B.m);
        const s60 = 0.8660254;

        let points: { x: number; y: number; label: string }[];
        if (Math.abs(mu - 0.5) < 1e-6) {
          points = [
            { x: ax + ux * d * 0.5, y: ay + uy * d * 0.5, label: 'L1' },
            { x: bx + ux * d * 0.55, y: by + uy * d * 0.55, label: 'L2' },
            { x: ax - ux * d * 0.55, y: ay - uy * d * 0.55, label: 'L3' },
            { x: ax + (ux * 0.5 - uy * s60) * d, y: ay + (uy * 0.5 + ux * s60) * d, label: 'L4' },
            { x: ax + (ux * 0.5 + uy * s60) * d, y: ay + (uy * 0.5 - ux * s60) * d, label: 'L5' },
          ];
        } else {
          const alpha = Math.cbrt(mu / 3);
          points = [
            { x: ax + ux * d * (1 - alpha), y: ay + uy * d * (1 - alpha), label: 'L1' },
            { x: bx + ux * d * alpha, y: by + uy * d * alpha, label: 'L2' },
            { x: ax - ux * d * (1 - (7 * mu) / 12), y: ay - uy * d * (1 - (7 * mu) / 12), label: 'L3' },
            { x: ax + (ux * 0.5 - uy * s60) * d, y: ay + (uy * 0.5 + ux * s60) * d, label: 'L4' },
            { x: ax + (ux * 0.5 + uy * s60) * d, y: ay + (uy * 0.5 - ux * s60) * d, label: 'L5' },
          ];
        }

        ctx.save();
        points.forEach((p) => {
          if (p.x >= 0 && p.x <= 8 && p.y >= 0 && p.y <= 8) {
            const px = p.x * sqSize;
            const py = (8 - p.y) * sqSize;
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = '#0050c0';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.font = '700 8px Sometype Mono, monospace';
            ctx.fillStyle = '#0050c0';
            ctx.fillText(p.label, px + 5, py - 3);
          }
        });
        ctx.restore();
      }
    }

    // Tidal Tensors
    if (showTidalStress) {
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
    ctx.fillStyle = 'rgba(17, 17, 17, 0.55)';
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
    ctx.strokeStyle = '#dcdad2';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, hwidth, hheight);

    const pad = 12;
    let currY = pad + 8;

    // Header brand & ply
    ctx.font = '700 10px Sometype Mono, monospace';
    ctx.fillStyle = '#0050c0';
    ctx.fillText('KEPLER-64 // ROCHE ENGINE', pad, currY);

    ctx.textAlign = 'right';
    ctx.font = '600 9px Sometype Mono, monospace';
    ctx.fillStyle = '#62615a';
    ctx.fillText(`Ply ${plyNum + 1}/${totalPlies}`, hwidth - pad, currY);
    ctx.textAlign = 'left';

    currY += 15;
    ctx.font = '600 10px Space Grotesk, sans-serif';
    ctx.fillStyle = '#62615a';
    ctx.fillText(gameTitle, pad, currY);

    currY += 22;
    // Big Move Text
    ctx.font = '700 18px Space Grotesk, sans-serif';
    ctx.fillStyle = '#111111';
    ctx.fillText(moveSan, pad, currY);

    // Status Badge
    const isDisrupted = (breakdown.blackKingTidal?.isDisrupted || breakdown.whiteKingTidal?.isDisrupted);
    const statusText = isDisrupted ? '⚠ ROCHE DISRUPTION' : '● TIDAL EQUILIBRIUM';
    ctx.font = '700 9px Sometype Mono, monospace';
    ctx.fillStyle = isDisrupted ? '#d92f00' : '#0050c0';
    ctx.textAlign = 'right';
    ctx.fillText(statusText, hwidth - pad, currY - 3);
    ctx.textAlign = 'left';

    currY += 15;
    // Roche Disruption Gauge
    const activeKingTidal = board.turn === 'w' ? breakdown.blackKingTidal : breakdown.whiteKingTidal;
    const etaVal = activeKingTidal ? activeKingTidal.eta.toFixed(3) : '0.000';
    const etaRatio = activeKingTidal ? Math.min(100, Math.round((activeKingTidal.eta / config.roche) * 100)) : 0;

    ctx.font = '600 9px Sometype Mono, monospace';
    ctx.fillStyle = '#111111';
    ctx.fillText(`Roche Disruption (η = ${etaVal} / ρ = ${config.roche.toFixed(2)})`, pad, currY);

    currY += 4;
    const trackW = hwidth - 2 * pad;
    ctx.fillStyle = '#eae8e0';
    ctx.fillRect(pad, currY, trackW, 4);
    ctx.fillStyle = isDisrupted ? '#ff3b00' : '#0069ff';
    ctx.fillRect(pad, currY, (trackW * etaRatio) / 100, 4);

    currY += 16;
    // Waterfall Ledger
    ctx.font = '700 9px Space Grotesk, sans-serif';
    ctx.fillStyle = '#111111';
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
      ctx.fillStyle = '#62615a';
      ctx.fillText(r.name, pad, currY);

      const valStr = (r.val >= 0 ? '+' : '') + r.val.toFixed(2);
      ctx.textAlign = 'right';
      ctx.fillText(valStr, hwidth - pad, currY);
      ctx.textAlign = 'left';

      // Bar
      const barX = pad + 88;
      const barW = hwidth - 2 * pad - 132;
      ctx.fillStyle = '#eae8e0';
      ctx.fillRect(barX, currY - 6, barW, 3.5);

      const fillW = Math.min(barW, Math.round((Math.abs(r.val) / r.max) * barW));
      ctx.fillStyle = r.val >= 0 ? '#0069ff' : '#ff5a00';
      ctx.fillRect(barX, currY - 6, fillW, 3.5);

      currY += 13;
    });

    currY += 2;
    ctx.strokeStyle = '#dcdad2';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, currY);
    ctx.lineTo(hwidth - pad, currY);
    ctx.stroke();

    currY += 14;
    ctx.font = '700 9.5px Sometype Mono, monospace';
    ctx.fillStyle = '#111111';
    ctx.fillText('NET POSITION SCORE:', pad, currY);

    ctx.textAlign = 'right';
    ctx.font = '700 12px Sometype Mono, monospace';
    ctx.fillStyle = '#0069ff';
    ctx.fillText(`${(breakdown.totalScoreMover >= 0 ? '+' : '') + breakdown.totalScoreMover.toFixed(2)} native`, hwidth - pad, currY);
    ctx.textAlign = 'left';

    // Candidate Moves Matrix
    currY += 14;
    ctx.fillStyle = '#f5f4ef';
    ctx.fillRect(pad, currY - 10, hwidth - 2 * pad, 62);
    ctx.strokeStyle = '#dcdad2';
    ctx.strokeRect(pad, currY - 10, hwidth - 2 * pad, 62);

    ctx.font = '700 8.5px Sometype Mono, monospace';
    ctx.fillStyle = '#62615a';
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
      ctx.fillStyle = '#0050c0';
      ctx.fillText(`#${idx + 1} ${m.san}`, pad + 6, currY);

      ctx.textAlign = 'right';
      ctx.font = '600 8.5px Sometype Mono, monospace';
      ctx.fillStyle = '#111111';
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
    ctx.strokeStyle = '#dcdad2';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, tw, th);

    // Label
    ctx.font = '700 8px Sometype Mono, monospace';
    ctx.fillStyle = '#0050c0';
    ctx.fillText('GRAVITATIONAL TRAJECTORY HORIZON', 10, 11);

    const padX = 12;
    const padY = 16;
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
    ctx.strokeStyle = '#dcdad2';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padX, zeroY);
    ctx.lineTo(tw - padX, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Area fills (flat, no gradients)
    ctx.fillStyle = 'rgba(0, 105, 255, 0.12)';
    ctx.beginPath();
    ctx.moveTo(coords[0].x, zeroY);
    coords.forEach((c) => {
      ctx.lineTo(c.x, Math.min(zeroY, c.y));
    });
    ctx.lineTo(coords[coords.length - 1].x, zeroY);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = 'rgba(255, 90, 0, 0.12)';
    ctx.beginPath();
    ctx.moveTo(coords[0].x, zeroY);
    coords.forEach((c) => {
      ctx.lineTo(c.x, Math.max(zeroY, c.y));
    });
    ctx.lineTo(coords[coords.length - 1].x, zeroY);
    ctx.closePath();
    ctx.fill();

    // Trajectory wave stroke
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    coords.forEach((c, idx) => {
      if (idx === 0) ctx.moveTo(c.x, c.y);
      else ctx.lineTo(c.x, c.y);
    });
    ctx.stroke();

    // Active Ply Indicator
    const activeCoord = coords[activePly] || coords[0];
    ctx.strokeStyle = '#ff5a00';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(activeCoord.x, padY);
    ctx.lineTo(activeCoord.x, padY + waveH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#ff5a00';
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(activeCoord.x, activeCoord.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }
}
