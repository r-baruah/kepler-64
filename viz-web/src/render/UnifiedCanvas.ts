/**
 * Kepler-64 Unified Canvas Renderer
 * Renders the fused Chessboard + Topographic Gravity Field + Force Vectors + Tidal Tensors
 */

import { KeplerBoard } from '../core/board';
import type { Piece } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { DEFAULT_CONSTANTS } from '../core/constants';
import { potentialOnGrid, forceField, DIST_64 } from '../core/gravity';
import { evaluatePosition } from '../core/evaluate';
import type { ScoreBreakdown } from '../core/evaluate';
import { generateContourLines } from './ContourRenderer';
import { drawTidalStressEllipse } from './TidalRenderer';
import { drawAccretionHalos, drawCaptureStream } from './AccretionRenderer';

export interface RenderLayers {
  showHeatmap: boolean;
  showContours: boolean;
  showVectors: boolean;
  showTidalStress: boolean;
  showAccretion: boolean;
  showWavefronts: boolean;
  showLorentz: boolean;
  showLagrange: boolean;
}

export interface SquareHoverInfo {
  square: number;
  fileChar: string;
  rankNum: number;
  piece: Piece | null;
  mass: number;
  potential: number;
  forceMag: number;
}

export class UnifiedCanvas {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private board: KeplerBoard;
  private config: ConstantsConfig;
  private layers: RenderLayers = {
    showHeatmap: true,
    showContours: true,
    showVectors: false,
    showTidalStress: true,
    showAccretion: true,
    showWavefronts: false,
    showLorentz: false,
    showLagrange: false,
  };

  private pieceImages: Map<string, HTMLImageElement> = new Map();
  private isLoaded = false;
  private lastMove: { from: number; to: number } | null = null;
  private accretionExcess: Record<number, number> = {};
  private captureStream: { fromSq: number; toSq: number; startTime: number } | null = null;
  private animFrame: number | null = null;

  // Hover state
  private hoveredSquare: number | null = null;
  private onHoverCallback?: (info: SquareHoverInfo | null) => void;

  // Selection & Tap-to-move state
  private selectedSquare: number | null = null;

  // Drag-and-drop state
  private isDragging = false;
  private dragPiece: Piece | null = null;
  private dragFromSq: number | null = null;
  private dragStartX = 0;
  private dragStartY = 0;
  private dragLastClientX = 0;
  private dragLastClientY = 0;
  private dragCanvasX = 0;
  private dragCanvasY = 0;
  private onMoveCallback?: (fromSq: number, toSq: number) => void;
  private onEvaluateCallback?: (breakdown: ScoreBreakdown) => void;

  constructor(
    canvas: HTMLCanvasElement,
    board: KeplerBoard,
    config: ConstantsConfig = DEFAULT_CONSTANTS
  ) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not obtain 2D canvas context');
    this.ctx = ctx;
    this.board = board;
    this.config = config;

    this.preloadPieces();
    this.attachEventListeners();
  }

  public setBoard(board: KeplerBoard, lastMove: { from: number; to: number } | null = null): void {
    this.board = board;
    this.lastMove = lastMove;
    this.selectedSquare = null;
    this.captureStream = null;
    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
    this.render();
  }

  public setConfig(config: ConstantsConfig): void {
    this.config = config;
    this.render();
  }

  public setLayers(layers: Partial<RenderLayers>): void {
    this.layers = { ...this.layers, ...layers };
    this.render();
  }

  public setAccretion(excess: Record<number, number>): void {
    this.accretionExcess = excess;
  }

  public playCaptureStream(fromSq: number, toSq: number): void {
    this.captureStream = { fromSq, toSq, startTime: performance.now() };
    if (this.animFrame === null) {
      this.tickAnimation();
    }
  }

  private tickAnimation = (): void => {
    if (!this.captureStream) {
      this.animFrame = null;
      return;
    }
    const t = (performance.now() - this.captureStream.startTime) / 300;
    if (t >= 1) {
      this.captureStream = null;
      this.animFrame = null;
      this.render();
      return;
    }
    this.render();
    this.animFrame = requestAnimationFrame(this.tickAnimation);
  };

  public onMove(cb: (fromSq: number, toSq: number) => void): void {
    this.onMoveCallback = cb;
  }

  public onEvaluate(cb: (breakdown: ScoreBreakdown) => void): void {
    this.onEvaluateCallback = cb;
  }

  public onHover(cb: (info: SquareHoverInfo | null) => void): void {
    this.onHoverCallback = cb;
  }

  private preloadPieces(): void {
    const pieces = [
      'bb', 'bk', 'bn', 'bp', 'bq', 'br',
      'wb', 'wk', 'wn', 'wp', 'wq', 'wr',
    ];
    let loadedCount = 0;

    const base = (import.meta.env.BASE_URL || './').replace(/\/$/, '') + '/';
    pieces.forEach((name) => {
      const img = new Image();
      img.src = `${base}pieces/${name}.svg`;
      img.onload = () => {
        this.pieceImages.set(name, img);
        loadedCount++;
        if (loadedCount === pieces.length) {
          this.isLoaded = true;
          this.render();
        }
      };
      img.onerror = () => {
        console.error(`Failed to load piece image: ${base}pieces/${name}.svg`);
      };
    });
  }

  private attachEventListeners(): void {
    this.canvas.addEventListener('mousedown', this.handlePointerDown.bind(this));
    this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
    this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
    window.addEventListener('mousemove', this.handleWindowMouseMove.bind(this));
    window.addEventListener('mouseup', this.handlePointerUp.bind(this));

    // Touch support with touch-action lock
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        const touch = e.touches[0];
        this.startDrag(touch.clientX, touch.clientY);
      }
    }, { passive: false });

    window.addEventListener('touchmove', (e) => {
      if (this.isDragging && e.touches.length === 1) {
        if (e.cancelable) e.preventDefault();
        const touch = e.touches[0];
        this.updateDrag(touch.clientX, touch.clientY);
      }
    }, { passive: false });

    window.addEventListener('touchend', (e) => {
      if (this.isDragging) {
        if (e.changedTouches.length > 0) {
          const touch = e.changedTouches[0];
          this.endDrag(touch.clientX, touch.clientY);
        } else {
          this.endDrag();
        }
      }
    });
  }

  private getCanvasSquare(clientX: number, clientY: number): number | null {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    if (x < 0 || x > rect.width || y < 0 || y > rect.height) return null;

    const file = Math.floor((x / rect.width) * 8);
    const rank = 7 - Math.floor((y / rect.height) * 8);
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
    return rank * 8 + file;
  }

  private handleMouseMove(e: MouseEvent): void {
    if (this.isDragging) return;
    const sq = this.getCanvasSquare(e.clientX, e.clientY);
    if (sq !== this.hoveredSquare) {
      this.hoveredSquare = sq;
      if (this.onHoverCallback) {
        if (sq !== null) {
          const f = sq % 8;
          const r = Math.floor(sq / 8);
          const piece = this.board.squares[sq];
    let masses = this.board.massVector();

    // Live field warping: relocate the dragged piece's mass under the cursor.
    if (this.isDragging && this.dragPiece && this.dragFromSq !== null) {
      const targetSq = this.hoveredSquare ?? this.dragFromSq;
      const carried = masses[this.dragFromSq];
      masses = Float32Array.from(masses);
      masses[this.dragFromSq] = 0;
      if (targetSq !== null) masses[targetSq] += carried;
    }
          const forces = forceField(masses, this.config.eps, this.config.G, this.config.c);
          const fMag = Math.sqrt(forces.fx[sq] * forces.fx[sq] + forces.fy[sq] * forces.fy[sq]);

          this.onHoverCallback({
            square: sq,
            fileChar: String.fromCharCode(97 + f),
            rankNum: r + 1,
            piece,
            mass: piece ? piece.mass : 0,
            potential: 0,
            forceMag: fMag,
          });
        } else {
          this.onHoverCallback(null);
        }
      }
      this.render();
    }
  }

  private handleMouseLeave(): void {
    this.hoveredSquare = null;
    if (this.onHoverCallback) this.onHoverCallback(null);
    this.render();
  }

  private handlePointerDown(e: MouseEvent): void {
    this.startDrag(e.clientX, e.clientY);
  }

  private startDrag(clientX: number, clientY: number): void {
    this.dragStartX = clientX;
    this.dragStartY = clientY;
    this.dragLastClientX = clientX;
    this.dragLastClientY = clientY;

    const rect = this.canvas.getBoundingClientRect();
    this.dragCanvasX = ((clientX - rect.left) / rect.width) * this.canvas.width;
    this.dragCanvasY = ((clientY - rect.top) / rect.height) * this.canvas.height;

    const sq = this.getCanvasSquare(clientX, clientY);
    if (sq === null) return;
    const piece = this.board.squares[sq];

    this.isDragging = true;
    this.dragPiece = piece;
    this.dragFromSq = sq;
    this.render();
  }

  private handleWindowMouseMove(e: MouseEvent): void {
    if (!this.isDragging) return;
    this.updateDrag(e.clientX, e.clientY);
  }

  private updateDrag(clientX: number, clientY: number): void {
    this.dragLastClientX = clientX;
    this.dragLastClientY = clientY;

    const rect = this.canvas.getBoundingClientRect();
    this.dragCanvasX = ((clientX - rect.left) / rect.width) * this.canvas.width;
    this.dragCanvasY = ((clientY - rect.top) / rect.height) * this.canvas.height;
    this.hoveredSquare = this.getCanvasSquare(clientX, clientY);
    this.render();
  }

  private handlePointerUp(e: MouseEvent): void {
    if (!this.isDragging) return;
    this.endDrag(e.clientX, e.clientY);
  }

  private endDrag(clientX?: number, clientY?: number): void {
    if (!this.isDragging) return;

    const finalX = clientX !== undefined ? clientX : this.dragLastClientX;
    const finalY = clientY !== undefined ? clientY : this.dragLastClientY;
    const dragDistance = Math.hypot(finalX - this.dragStartX, finalY - this.dragStartY);
    const targetSq = this.getCanvasSquare(finalX, finalY);

    if (dragDistance < 10) {
      // TAP / CLICK INTERACTION (Tap-to-move)
      if (this.selectedSquare === null) {
        if (targetSq !== null && this.board.squares[targetSq]) {
          this.selectedSquare = targetSq;
        }
      } else {
        if (targetSq === this.selectedSquare) {
          // Deselect
          this.selectedSquare = null;
        } else if (targetSq !== null) {
          // Attempt move from selectedSquare to targetSq
          if (this.onMoveCallback) {
            this.onMoveCallback(this.selectedSquare, targetSq);
          } else {
            const piece = this.board.squares[this.selectedSquare];
            this.board.squares[targetSq] = piece;
            this.board.squares[this.selectedSquare] = null;
            this.lastMove = { from: this.selectedSquare, to: targetSq };
          }
          this.selectedSquare = null;
        }
      }
    } else {
      // DRAG & DROP INTERACTION
      if (targetSq !== null && this.dragFromSq !== null && targetSq !== this.dragFromSq && this.dragPiece) {
        if (this.onMoveCallback) {
          this.onMoveCallback(this.dragFromSq, targetSq);
        } else {
          // Direct local board move
          this.board.squares[targetSq] = this.dragPiece;
          this.board.squares[this.dragFromSq] = null;
          this.lastMove = { from: this.dragFromSq, to: targetSq };
        }
        this.selectedSquare = null;
      }
    }

    this.isDragging = false;
    this.dragPiece = null;
    this.dragFromSq = null;
    this.render();
  }

  private computeLagrangePoints(): { x: number; y: number; label: string }[] {
    const wk = this.board.findKingSquare('w');
    const bk = this.board.findKingSquare('b');
    if (wk === null || bk === null) return [];

    const wx = (wk % 8) + 0.5;
    const wy = Math.floor(wk / 8) + 0.5;
    const bx = (bk % 8) + 0.5;
    const by = Math.floor(bk / 8) + 0.5;

    const dx = bx - wx;
    const dy = by - wy;
    const d = Math.hypot(dx, dy) || 1;
    const ux = dx / d;
    const uy = dy / d;

    const s60 = 0.8660254;
    const points = [
      { x: (wx + bx) / 2, y: (wy + by) / 2, label: 'L1' },
      { x: bx + ux * d * 0.55, y: by + uy * d * 0.55, label: 'L2' },
      { x: wx - ux * d * 0.55, y: wy - uy * d * 0.55, label: 'L3' },
      { x: wx + (ux * 0.5 - uy * s60) * d, y: wy + (uy * 0.5 + ux * s60) * d, label: 'L4' },
      { x: wx + (ux * 0.5 + uy * s60) * d, y: wy + (uy * 0.5 - ux * s60) * d, label: 'L5' },
    ];

    return points.filter((p) => p.x >= 0 && p.x <= 8 && p.y >= 0 && p.y <= 8);
  }

  public render(): void {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    const sqSize = width / 8.0;

    // 1. Draw Board Base Squares
    const sqLight = '#e9eff4';
    const sqDark = '#b8c9d9';

    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const isLight = (r + f) % 2 === 0;
        ctx.fillStyle = isLight ? sqLight : sqDark;
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    // 2. Draw Hover Square Highlight
    if (this.hoveredSquare !== null && !this.isDragging) {
      const f = this.hoveredSquare % 8;
      const r = Math.floor(this.hoveredSquare / 8);
      ctx.fillStyle = 'rgba(36, 72, 184, 0.18)'; // Ultramarine highlight
      ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
    }

    // 3. Draw Selected Square (Tap-to-move) Highlight
    if (this.selectedSquare !== null) {
      const f = this.selectedSquare % 8;
      const r = Math.floor(this.selectedSquare / 8);
      ctx.save();
      ctx.fillStyle = 'rgba(240, 100, 38, 0.25)';
      ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      ctx.strokeStyle = '#f06426';
      ctx.lineWidth = 3;
      ctx.strokeRect(f * sqSize + 1.5, (7 - r) * sqSize + 1.5, sqSize - 3, sqSize - 3);
      ctx.restore();
    }

    // 4. Draw Last Move Highlight
    if (this.lastMove) {
      ctx.fillStyle = 'rgba(240, 100, 38, 0.35)'; // Trajectory Orange
      for (const sq of [this.lastMove.from, this.lastMove.to]) {
        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    const masses = this.board.massVector();

    // 4. Draw Continuous Plummer Potential Heatmap
    if (this.layers.showHeatmap) {
      const { grid, n, p3, p97 } = potentialOnGrid(masses, this.config.eps, this.config.G, this.config.c, 64);
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

            // Deep Ultramarine -> Indigo -> Soft Amber -> Light Magma
            const r = Math.round(18 + t * (230 - 18));
            const g = Math.round(26 + t * (140 - 26));
            const b = Math.round(75 + t * (45 - 75));
            const alpha = Math.round(35 + t * 45); // Subtle, non-obscuring 15-30% opacity

            const idx = (gy * n + gx) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = alpha;
          }
        }
        offCtx.putImageData(imgData, 0, 0);

        ctx.save();
        ctx.drawImage(offCanvas, 0, 0, width, height);
        ctx.restore();
      }

      // 5. Draw Equipotential Contour Lines
      if (this.layers.showContours) {
        const levels: number[] = [];
        const numLevels = 14;
        for (let k = 1; k < numLevels; k++) {
          levels.push(p3 + (span * k) / numLevels);
        }

        const contours = generateContourLines(grid, n, levels);
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.lineWidth = 1.2;

        contours.forEach((segments) => {
          ctx.beginPath();
          segments.forEach((seg) => {
            const x1 = ((seg.x1 + 0.5) / 8.0) * width;
            const y1 = height - ((seg.y1 + 0.5) / 8.0) * height;
            const x2 = ((seg.x2 + 0.5) / 8.0) * width;
            const y2 = height - ((seg.y2 + 0.5) / 8.0) * height;
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
          });
          ctx.stroke();
        });
        ctx.restore();
      }
    }

    // 6. Draw Gravitational Force Vectors (Streamlines)
    if (this.layers.showVectors) {
      const forces = forceField(masses, this.config.eps, this.config.G, this.config.c);
      ctx.save();
      ctx.strokeStyle = 'rgba(240, 100, 38, 0.65)';
      ctx.fillStyle = 'rgba(240, 100, 38, 0.65)';
      ctx.lineWidth = 1.4;

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
        const len = Math.min(sqSize * 0.4, sqSize * 0.15 * Math.log10(1 + mag * 5));

        const endX = cx + normX * len;
        const endY = cy - normY * len; // Invert Y for canvas

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(-normY, normX);
        const head = 4;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - head * Math.cos(angle - Math.PI / 6), endY - head * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(endX - head * Math.cos(angle + Math.PI / 6), endY - head * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }

    // 7. Evaluate Position and Render Tidal Tensors
    const breakdown = evaluatePosition(this.board, this.config);
    if (this.onEvaluateCallback) {
      this.onEvaluateCallback(breakdown);
    }

    if (this.layers.showTidalStress) {
      const wk = this.board.findKingSquare('w');
      const bk = this.board.findKingSquare('b');

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

    // 7b. Draw Retarded Gravitational Wavefronts (finite c)
    if (this.layers.showWavefronts && this.lastMove) {
      const origin = this.lastMove.from;
      const of = origin % 8;
      const or = Math.floor(origin / 8);
      const cx = (of + 0.5) * sqSize;
      const cy = ((7 - or) + 0.5) * sqSize;
      const cSq = this.config.c;

      ctx.save();
      const ripples = [1, 0.62, 0.3];
      ripples.forEach((f, idx) => {
        ctx.beginPath();
        ctx.arc(cx, cy, cSq * f * sqSize, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 105, 255, ${0.5 - idx * 0.14})`;
        ctx.lineWidth = idx === 0 ? 1.6 : 1.2;
        ctx.setLineDash(idx === 0 ? [5, 5] : []);
        ctx.stroke();
      });
      ctx.setLineDash([]);

      // Horizon indicator: squares outside the light cone have not felt the update.
      ctx.strokeStyle = 'rgba(17, 17, 17, 0.22)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (let sq = 0; sq < 64; sq++) {
        if (DIST_64[origin * 64 + sq] <= cSq) continue;
        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.strokeRect(f * sqSize + 1.5, (7 - r) * sqSize + 1.5, sqSize - 3, sqSize - 3);
      }
      ctx.setLineDash([]);
      ctx.restore();
    }

    // 7c. Draw Kinetic Trails (Lorentz escalation)
    if (this.layers.showLorentz && this.lastMove) {
      const from = this.lastMove.from;
      const to = this.lastMove.to;
      const dist = DIST_64[from * 64 + to];
      if (dist >= 2) {
        const ratio = Math.min(0.95, dist / this.config.c);
        const rel = ratio > 0.6;
        const ff = from % 8;
        const fr = Math.floor(from / 8);
        const tf = to % 8;
        const tr = Math.floor(to / 8);
        const sx = (ff + 0.5) * sqSize;
        const sy = ((7 - fr) + 0.5) * sqSize;
        const ex = (tf + 0.5) * sqSize;
        const ey = ((7 - tr) + 0.5) * sqSize;

        ctx.save();
        const strokes = rel ? 4 : 2;
        for (let i = 1; i <= strokes; i++) {
          const t = i / (strokes + 1);
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(sx + (ex - sx) * t, sy + (ey - sy) * t);
          ctx.strokeStyle = `rgba(0, 229, 255, ${(rel ? 0.4 : 0.18) * (1 - t)})`;
          ctx.lineWidth = rel ? 2.2 : 1.2;
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // 7d. Draw Lagrange Equilibrium Points
    if (this.layers.showLagrange) {
      const lp = this.computeLagrangePoints();
      ctx.save();
      lp.forEach((p) => {
        const px = p.x * sqSize;
        const py = (8 - p.y) * sqSize;
        const triangular = p.label === 'L4' || p.label === 'L5';
        const arm = sqSize * 0.18;

        ctx.beginPath();
        ctx.moveTo(px - arm, py);
        ctx.lineTo(px + arm, py);
        ctx.moveTo(px, py - arm);
        ctx.lineTo(px, py + arm);
        ctx.strokeStyle = triangular ? 'rgba(255, 90, 0, 0.6)' : 'rgba(0, 105, 255, 0.65)';
        ctx.lineWidth = 1.4;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fillStyle = triangular ? 'rgba(255, 90, 0, 0.8)' : 'rgba(0, 105, 255, 0.8)';
        ctx.fill();

        ctx.font = '700 9px Sometype Mono, monospace';
        ctx.fillStyle = 'rgba(17, 17, 17, 0.6)';
        ctx.fillText(p.label, px + 3, py - 3);
      });
      ctx.restore();
    }

    // 7e. Draw Accretion Halos (Layer 2)
    if (this.layers.showAccretion) {
      drawAccretionHalos(ctx, this.board, this.accretionExcess, sqSize);
    }

    // 8. Draw Pieces
    for (let sq = 0; sq < 64; sq++) {
      if (this.isDragging && sq === this.dragFromSq) continue;
      const piece = this.board.squares[sq];
      if (!piece) continue;

      const f = sq % 8;
      const r = Math.floor(sq / 8);
      const px = f * sqSize;
      const py = (7 - r) * sqSize;

      const imgKey = `${piece.color}${piece.type}`;
      const img = this.pieceImages.get(imgKey);
      if (img && this.isLoaded) {
        const piecePad = sqSize * 0.08;
        ctx.drawImage(img, px + piecePad, py + piecePad, sqSize - 2 * piecePad, sqSize - 2 * piecePad);
      }

      // Draw Accretion Mass Badge (Layer 2)
      if (this.layers.showAccretion && (piece.mass > (piece.type === 'k' ? 1001.0 : piece.type === 'q' ? 9.2 : 5.2))) {
        ctx.save();
        ctx.font = '600 10px Sometype Mono, monospace';
        ctx.fillStyle = '#f06426';
        ctx.textAlign = 'right';
        ctx.fillText(`${piece.mass.toFixed(1)}m`, px + sqSize - 4, py + sqSize - 4);
        ctx.restore();
      }
    }

    // 9. Draw Dragging Piece
    if (this.isDragging && this.dragPiece) {
      const imgKey = `${this.dragPiece.color}${this.dragPiece.type}`;
      const img = this.pieceImages.get(imgKey);
      if (img && this.isLoaded) {
        const drawSize = sqSize * 1.15;
        ctx.save();
        ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
        ctx.shadowBlur = 12;
        ctx.shadowOffsetY = 6;
        ctx.drawImage(
          img,
          this.dragCanvasX - drawSize / 2,
          this.dragCanvasY - drawSize / 2,
          drawSize,
          drawSize
        );
        ctx.restore();
      }
    }

    // 9b. Draw Capture Matter Stream (Layer 2)
    if (this.captureStream) {
      const t = Math.min(1, (performance.now() - this.captureStream.startTime) / 300);
      drawCaptureStream(ctx, this.captureStream.fromSq, this.captureStream.toSq, sqSize, t);
    }

    // 9c. Draw Relativistic Glow (Lorentz escalation)
    if (this.layers.showLorentz && this.lastMove) {
      const dist = DIST_64[this.lastMove.from * 64 + this.lastMove.to];
      const ratio = Math.min(0.95, dist / this.config.c);
      if (ratio > 0.6) {
        const to = this.lastMove.to;
        const f = to % 8;
        const r = Math.floor(to / 8);
        const cx = (f + 0.5) * sqSize;
        const cy = ((7 - r) + 0.5) * sqSize;
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, sqSize * 0.58, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.7)';
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.restore();
      }
    }

    // 9d. Draw Gravitational Anchors (piece on a Lagrange point)
    if (this.layers.showLagrange) {
      const lp = this.computeLagrangePoints();
      for (let sq = 0; sq < 64; sq++) {
        const piece = this.board.squares[sq];
        if (!piece || (piece.type !== 'n' && piece.type !== 'b')) continue;
        const px = (sq % 8) + 0.5;
        const py = Math.floor(sq / 8) + 0.5;
        if (!lp.some((p) => Math.hypot(p.x - px, p.y - py) < 0.75)) continue;

        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 105, 255, 0.9)';
        ctx.lineWidth = 2;
        ctx.strokeRect(f * sqSize + 2, (7 - r) * sqSize + 2, sqSize - 4, sqSize - 4);
        ctx.font = '700 8px Sometype Mono, monospace';
        ctx.fillStyle = 'rgba(0, 105, 255, 0.95)';
        ctx.fillText('GRAV. ANCHOR', f * sqSize + 3, (7 - r) * sqSize + 11);
        ctx.restore();
      }
    }

    // 10. Board Outer Coordinates (a-h, 1-8)
    ctx.save();
    ctx.font = '600 10px Sometype Mono, monospace';
    ctx.fillStyle = 'rgba(23, 34, 53, 0.55)';

    for (let f = 0; f < 8; f++) {
      const char = String.fromCharCode(97 + f);
      ctx.fillText(char, f * sqSize + 3, height - 3);
    }
    for (let r = 0; r < 8; r++) {
      const num = (r + 1).toString();
      ctx.fillText(num, width - 10, (7 - r) * sqSize + 12);
    }
    ctx.restore();
  }
}
